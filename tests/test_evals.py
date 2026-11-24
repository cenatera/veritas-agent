"""Pruebas de la evaluación, sin red ni coste."""

from collections import Counter
from pathlib import Path

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from evals import run as runner
from evals.metrics import (CaseResult, Category, EvalCase, Outcome, load_cases, score, summarize,
                           wilson)
from evals.run import LIMITS, Meter, fingerprint, main, run_cases
from src.agents.llm import ModelOutputError
from src.costs import estimate_high_usd
from src.orchestrator import AuditResult
from src.schemas import Verdict
from tests.test_orchestrator import CLAIMS, FakeProvider, FakeVerifier, api_error, audit

CASE = EvalCase.model_validate({
    "id": "x1", "category": "cierta", "expected": "Confirmada", "text": "Uno mide 1.",
    "reference": {"url": "https://example.org", "excerpt": "Uno mide uno."}})
OTHER = CASE.model_copy(update={"id": "x2"})
HIGH = estimate_high_usd(CASE.text, LIMITS)


def good(text: str = "") -> AuditResult:
    """Auditoría simulada que confirma una afirmación."""
    return audit([FakeProvider("a", ["1"])])


def test_dataset_is_valid_and_balanced() -> None:
    cases = load_cases()
    assert 20 <= len(cases) <= 30
    assert set(Counter(c.category for c in cases)) == set(Category)
    assert len({c.text for c in cases}) == len(cases)
    for case in cases:
        assert (case.category is Category.UNVERIFIABLE) == (case.reference is None)


def test_conclusive_case_needs_reference() -> None:
    with pytest.raises(ValidationError, match="referencia"):
        EvalCase.model_validate({**CASE.model_dump(mode="json"), "reference": None})


@pytest.mark.parametrize(("expected", "providers", "outcome"), [
    ("Confirmada", ["1"], Outcome.CORRECT), ("Refutada", ["1"], Outcome.WRONG),
    ("Confirmada", [], Outcome.ABSTAINED),
])
def test_scoring(expected: str, providers: list[str], outcome: Outcome) -> None:
    case = CASE.model_copy(update={"expected": Verdict(expected)})
    assert score(case, audit([FakeProvider("a", providers)]), "f", 0.1).outcome is outcome


@pytest.mark.parametrize("claims", [(), CLAIMS[:2]])
def test_unexpected_extraction_is_not_a_hit(claims: tuple) -> None:
    result = score(CASE, audit([FakeProvider("a", ["1"])], claims=claims), "f", 0)
    assert result.outcome is Outcome.EXTRACTION and result.predicted is None


def test_failed_verification_is_incomplete() -> None:
    failed = audit([FakeProvider("a", ["1"])], verifier=FakeVerifier({"c1": ModelOutputError("x")}))
    assert score(CASE, failed, "f", 0).outcome is Outcome.INCOMPLETE


def test_search_problems_are_flagged() -> None:
    result = score(CASE, audit([FakeProvider("a", ["1"]), FakeProvider("b", None)]), "f", 0)
    assert result.outcome is Outcome.CORRECT and result.degraded


def test_wilson_interval() -> None:
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.490, abs=0.001) and high == pytest.approx(0.943, abs=0.001)
    assert wilson(0, 5)[0] == 0 and wilson(5, 5)[1] == 1
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_summary_shows_everything() -> None:
    result = CaseResult(case_id="x1", fingerprint="f", outcome=Outcome.WRONG,
                        predicted=Verdict.REFUTED, usd=0.25)
    text = summarize([CASE, OTHER], [result])
    assert "1 de 2" in text and "no ejecutado" in text and "0/1" in text and "0.25 USD" in text
    assert "0 de 2" in summarize([CASE, OTHER], [])


def test_budget_prevents_starting_a_case(tmp_path: Path) -> None:
    calls: list[str] = []
    meter = Meter()

    def spend(text: str) -> AuditResult:
        calls.append(text)
        meter.usd += 0.2
        return good()

    assert run_cases([CASE], spend, meter, HIGH - 0.01, tmp_path) == [] and calls == []
    results = run_cases([CASE, OTHER], spend, meter, HIGH + 0.1, tmp_path)
    assert len(calls) == 1 and results[0].usd == pytest.approx(0.2)


def test_results_are_reused_until_the_case_changes(tmp_path: Path) -> None:
    calls: list[str] = []

    def count(text: str) -> AuditResult:
        calls.append(text)
        return good()

    run_cases([CASE], count, Meter(), 10, tmp_path)
    run_cases([CASE], count, Meter(), 10, tmp_path)
    assert len(calls) == 1
    changed = CASE.model_copy(update={"text": "Otro."})
    assert fingerprint(changed) != fingerprint(CASE)
    run_cases([changed], count, Meter(), 10, tmp_path)
    assert len(calls) == 2


def test_incomplete_cases_are_retried(tmp_path: Path) -> None:
    def fail(text: str) -> AuditResult:
        raise api_error(anthropic.RateLimitError, 429)

    assert run_cases([CASE], fail, Meter(), 10, tmp_path)[0].outcome is Outcome.INCOMPLETE
    assert run_cases([CASE], good, Meter(), 10, tmp_path)[0].outcome is Outcome.CORRECT


def test_bad_key_stops_the_run(tmp_path: Path) -> None:
    def fail(text: str) -> AuditResult:
        raise api_error(anthropic.AuthenticationError, 401)

    with pytest.raises(anthropic.AuthenticationError):
        run_cases([CASE], fail, Meter(), 10, tmp_path)


def test_meter_counts_only_successful_responses() -> None:
    meter = Meter()
    meter.record(httpx2.Response(200, json={"usage": {"input_tokens": 1_000_000, "output_tokens": 100_000}}))
    meter.record(httpx2.Response(429, json={}))
    assert meter.usd == pytest.approx(6.0)


def test_default_mode_never_calls_the_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner.anthropic, "Anthropic", lambda **k: pytest.fail("Llamada a la API"))
    assert main([]) == 0
    assert "No se ha llamado" in capsys.readouterr().out
    for arguments in (["--run"], ["--run", "--max-usd", "0"], ["--run", "--max-usd", "nan"]):
        with pytest.raises(SystemExit):
            main(arguments)
