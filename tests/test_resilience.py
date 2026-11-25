"""Pruebas de reintentos y ritmo, sin esperas reales."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx2
import pytest

from src.tools.resilience import (FailureKind, Pacer, RetryPolicy, SourceError, from_exception,
                                  from_response, with_retries)


def failing(*errors: SourceError) -> Callable[[], str]:
    """Operación que falla en orden y después funciona."""
    queue = list(errors)

    def operation() -> str:
        if queue:
            raise queue.pop(0)
        return "ok"

    return operation


def limited(retry_after: float | None = None) -> SourceError:
    """Fallo por límite de peticiones."""
    return SourceError("x", FailureKind.RATE_LIMITED, retry_after)


def test_backoff_is_exponential_and_capped() -> None:
    waits: list[float] = []
    policy = RetryPolicy(attempts=5, base_seconds=2, max_seconds=10)
    assert with_retries(failing(*[limited() for _ in range(4)]), policy, waits.append,
                        lambda low, high: 0) == "ok"
    assert waits == [2, 4, 8, 10]


def test_jitter_stays_in_range() -> None:
    policy = RetryPolicy(base_seconds=4, jitter=0.25)
    assert policy.delay(1, lambda low, high: low) == 4
    assert policy.delay(1, lambda low, high: high) == 5


def test_retry_after_extends_the_wait() -> None:
    waits: list[float] = []
    with_retries(failing(limited(7)), RetryPolicy(base_seconds=1), waits.append, lambda a, b: 0)
    assert waits == [7]


@pytest.mark.parametrize(("errors", "attempts"), [
    ([limited() for _ in range(3)], 3),
    ([SourceError("x")], 1),
    ([limited(99)], 1),
])
def test_gives_up_with_attempt_count(errors: list[SourceError], attempts: int) -> None:
    with pytest.raises(SourceError) as caught:
        with_retries(failing(*errors), RetryPolicy(attempts=3), lambda s: None)
    assert caught.value.attempts == attempts


@pytest.mark.parametrize(("status", "kind"), [
    (429, FailureKind.RATE_LIMITED), (500, FailureKind.SERVER), (503, FailureKind.SERVER),
    (408, FailureKind.TIMEOUT), (404, FailureKind.OTHER), (301, FailureKind.OTHER),
])
def test_statuses_are_classified(status: int, kind: FailureKind) -> None:
    assert from_response(SourceError, httpx2.Response(status)).kind is kind


@pytest.mark.parametrize(("error", "kind"), [
    (httpx2.ReadTimeout("x"), FailureKind.TIMEOUT), (httpx2.ConnectError("x"), FailureKind.NETWORK),
    (httpx2.RemoteProtocolError("x"), FailureKind.NETWORK), (httpx2.InvalidURL("x"), FailureKind.OTHER),
])
def test_exceptions_are_classified(error: Exception, kind: FailureKind) -> None:
    assert from_exception(SourceError, error).kind is kind


@pytest.mark.parametrize(("value", "low", "high"), [
    ("12", 12, 12), (format_datetime(datetime.now(UTC) + timedelta(seconds=60)), 50, 60),
    (format_datetime(datetime.now(UTC) - timedelta(seconds=60)), 0, 0),
])
def test_retry_after_formats(value: str, low: float, high: float) -> None:
    error = from_response(SourceError, httpx2.Response(429, headers={"retry-after": value}))
    assert error.retry_after is not None and low <= error.retry_after <= high


@pytest.mark.parametrize("value", ["", "pronto", "-1", "²"])
def test_invalid_retry_after_is_ignored(value: str) -> None:
    from src.tools.resilience import _retry_after
    assert _retry_after(value) is None


def test_pause_expires() -> None:
    now = [0.0]
    pacer = Pacer(1, clock=lambda: now[0])
    assert pacer.paused() is None
    pacer.pause(10, FailureKind.TIMEOUT)
    assert pacer.paused() is FailureKind.TIMEOUT
    now[0] = 10
    assert pacer.paused() is None
