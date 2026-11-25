"""Fallos de fuentes externas: clasificación, reintentos y ritmo de peticiones.

Solo se reintentan los fallos transitorios (HTTP 429, 5xx, tiempo agotado y red),
con espera exponencial, variación aleatoria y respeto de ``Retry-After``.
"""

import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import TypeVar

import httpx2
from pydantic import Field

from src.schemas import VeritasModel

ResultT = TypeVar("ResultT")
Sleep = Callable[[float], None]
Clock = Callable[[], float]


class FailureKind(str, Enum):
    """Clase de fallo de una fuente externa."""

    RATE_LIMITED = "límite de peticiones"
    SERVER = "error del servicio"
    TIMEOUT = "tiempo de espera agotado"
    NETWORK = "fallo de conexión"
    OTHER = "respuesta no utilizable"

    @property
    def is_transient(self) -> bool:
        """Indica si repetir la petición más tarde podría funcionar."""
        return self is not FailureKind.OTHER


class SourceError(RuntimeError):
    """Una fuente externa no pudo consultarse."""

    def __init__(
        self,
        message: str,
        kind: FailureKind = FailureKind.OTHER,
        retry_after: float | None = None,
    ) -> None:
        """Guarda la clase de fallo y la espera que pide el servicio, en segundos."""
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after
        self.attempts = 1


ErrorT = TypeVar("ErrorT", bound=SourceError)


def _retry_after(value: str | None) -> float | None:
    """Interpreta ``Retry-After`` en segundos o como fecha HTTP."""
    if not value:
        return None
    text = value.strip()
    if text.isascii() and text.isdigit():
        return float(text)
    try:
        moment = parsedate_to_datetime(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - datetime.now(UTC)).total_seconds())


def from_response(error_type: type[ErrorT], response: httpx2.Response) -> ErrorT:
    """Crea el error de una respuesta HTTP no satisfactoria."""
    status = response.status_code
    if status == 429:
        kind = FailureKind.RATE_LIMITED
    elif status >= 500:
        kind = FailureKind.SERVER
    elif status == 408:
        kind = FailureKind.TIMEOUT
    else:
        kind = FailureKind.OTHER
    return error_type(
        f"Respuesta HTTP {status}.", kind, _retry_after(response.headers.get("retry-after"))
    )


def from_exception(error_type: type[ErrorT], error: Exception) -> ErrorT:
    """Crea el error de un fallo de transporte."""
    if isinstance(error, httpx2.TimeoutException):
        kind = FailureKind.TIMEOUT
    elif isinstance(error, httpx2.NetworkError | httpx2.RemoteProtocolError):
        kind = FailureKind.NETWORK
    else:
        kind = FailureKind.OTHER
    return error_type(f"Fallo de conexión: {type(error).__name__}.", kind)


class RetryPolicy(VeritasModel):
    """Número de intentos y espera exponencial entre ellos."""

    attempts: int = Field(default=3, ge=1, le=6)
    base_seconds: float = Field(default=1.0, gt=0)
    max_seconds: float = Field(default=20.0, gt=0)
    jitter: float = Field(default=0.25, ge=0, le=1, description="Fracción aleatoria añadida.")

    def delay(self, failures: int, uniform: Callable[[float, float], float]) -> float:
        """Espera tras ``failures`` intentos fallidos."""
        seconds = min(self.max_seconds, self.base_seconds * 2 ** (failures - 1))
        return seconds + uniform(0, seconds * self.jitter)


def with_retries(
    operation: Callable[[], ResultT],
    policy: RetryPolicy,
    sleep: Sleep = time.sleep,
    uniform: Callable[[float, float], float] = random.uniform,
) -> ResultT:
    """Ejecuta ``operation`` y reintenta los fallos transitorios.

    Si el servicio pide esperar más de ``max_seconds``, no se insiste antes de
    tiempo: el fallo se devuelve de inmediato.
    """
    for attempt in range(1, policy.attempts + 1):
        try:
            return operation()
        except SourceError as error:
            error.attempts = attempt
            too_long = error.retry_after is not None and error.retry_after > policy.max_seconds
            if not error.kind.is_transient or too_long or attempt == policy.attempts:
                raise
            sleep(max(policy.delay(attempt, uniform), error.retry_after or 0.0))
    raise AssertionError("inalcanzable")


class Pacer:
    """Espacia peticiones y suspende un servicio que ha dejado de responder.

    Se comparte entre auditorías del mismo proceso y es seguro entre hilos.
    """

    def __init__(
        self, interval: float, sleep: Sleep = time.sleep, clock: Clock = time.monotonic
    ) -> None:
        """Fija la separación mínima en segundos."""
        self._interval = interval
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next_request = float("-inf")
        self._paused_until = float("-inf")
        self._pause_kind = FailureKind.OTHER

    def wait(self) -> None:
        """Espera el turno de la siguiente petición."""
        with self._lock:
            delay = self._next_request - self._clock()
            if delay > 0:
                self._sleep(delay)
            self._next_request = self._clock() + self._interval

    def pause(self, seconds: float, kind: FailureKind) -> None:
        """Suspende el servicio durante ``seconds``."""
        with self._lock:
            self._paused_until = max(self._paused_until, self._clock() + seconds)
            self._pause_kind = kind

    def paused(self) -> FailureKind | None:
        """Motivo de la suspensión vigente, si la hay."""
        with self._lock:
            return self._pause_kind if self._clock() < self._paused_until else None
