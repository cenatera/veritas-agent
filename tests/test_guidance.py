"""Pruebas de las explicaciones al lector."""

from typing import Any

import pytest

from src.guidance import Cause, build_notices, explain_claim, explain_no_claims, looks_like_question
from src.orchestrator import AuditLimits, ClaimStatus, ClaimTrace, Issue, IssueKind
from src.schemas import ClaimAssessment, Verdict

LIMITS = AuditLimits(max_claims=3, max_sources=2, max_source_chars=50_000)
NOTHING = ClaimAssessment(claim_id="c1", verdict=Verdict.INSUFFICIENT, rationale="x")


def trace(status: ClaimStatus, *kinds: IssueKind, **changes: Any) -> ClaimTrace:
    """Recorrido ficticio."""
    return ClaimTrace(claim_id="c1", status=status,
                      issues=tuple(Issue(kind=k, origin="Búsqueda en gdelt") for k in kinds), **changes)


@pytest.mark.parametrize(("item", "cause"), [
    (trace(ClaimStatus.SKIPPED), Cause.CLAIM_LIMIT),
    (trace(ClaimStatus.FAILED, IssueKind.VERIFIER), Cause.VERIFIER),
    (trace(ClaimStatus.NO_SOURCES, IssueKind.RATE_LIMITED), Cause.OUTAGE),
    (trace(ClaimStatus.NO_SOURCES, IssueKind.UNUSABLE, IssueKind.TIMEOUT), Cause.OUTAGE),
    (trace(ClaimStatus.NO_SOURCES, IssueKind.NO_RESULTS), Cause.NO_RESULTS),
    (trace(ClaimStatus.NO_SOURCES, IssueKind.COPY), Cause.UNUSABLE),
    (trace(ClaimStatus.CHECKED, source_ids=("s1",)), Cause.NO_EVIDENCE),
    (trace(ClaimStatus.CHECKED, proposed_verdict=Verdict.SUPPORTED), Cause.ADJUSTED),
])
def test_every_cause_is_identified(item: ClaimTrace, cause: Cause) -> None:
    guidance = explain_claim(NOTHING, item, 5, LIMITS)
    assert guidance is not None and guidance.cause == cause.value and guidance.action
    if item.status is not ClaimStatus.SKIPPED:
        assert "no equivale a que la afirmación sea falsa" in guidance.explanation


def test_conclusive_verdict_needs_no_explanation() -> None:
    confirmed = ClaimAssessment.model_validate({
        "claim_id": "c1", "verdict": "Confirmada", "confidence": "media", "rationale": "x",
        "evidence": [{"source_id": "s", "excerpt": "uno dos tres", "stance": "respalda"}]})
    assert explain_claim(confirmed, trace(ClaimStatus.CHECKED), 1, LIMITS) is None


@pytest.mark.parametrize(("total", "expected"), [(5, "a 5 en «Alcance»"), (14, "divida el texto")])
def test_limit_action_is_feasible(total: int, expected: str) -> None:
    guidance = explain_claim(NOTHING, trace(ClaimStatus.SKIPPED), total, LIMITS)
    assert guidance is not None and expected in guidance.action and "máximo de 3" in guidance.explanation


def test_actions_respect_the_sliders() -> None:
    low = explain_claim(NOTHING, trace(ClaimStatus.CHECKED), 1, LIMITS)
    top = explain_claim(NOTHING, trace(ClaimStatus.CHECKED), 1, AuditLimits(max_sources=4))
    large = explain_claim(NOTHING, trace(ClaimStatus.NO_SOURCES, IssueKind.TOO_LARGE), 1, LIMITS)
    assert low and top and large
    assert "Fuentes por afirmación" in low.action and "Fuentes por afirmación" not in top.action
    assert "Caracteres de fuentes" in large.action


def test_partial_outage_is_mentioned_after_checking() -> None:
    guidance = explain_claim(NOTHING, trace(ClaimStatus.CHECKED, IssueKind.SERVER), 1, None)
    assert guidance and "búsqueda fue incompleta" in guidance.explanation
    assert "Espere" in guidance.action


@pytest.mark.parametrize(("text", "expected"), [
    ("¿Cuánto mide?", True), ("Who won?", True), ("¿Es cierto? ¿Cuándo fue?", True),
    ("«¿Cuánto mide?»", True), ("¿Cuánto mide? Mide 10 m.", False), ("Mide 10 m.", False), ("", False),
])
def test_question_detection(text: str, expected: bool) -> None:
    assert looks_like_question(text) is expected


def test_no_claims_explanations_differ() -> None:
    causes = {explain_no_claims("¿Qué?", 0).cause, explain_no_claims("Es bonito.", 0).cause,
              explain_no_claims("Mide 1.", 2).cause}
    assert len(causes) == 3


def test_notices_are_deduplicated() -> None:
    outage = trace(ClaimStatus.CHECKED, IssueKind.RATE_LIMITED)
    notices = build_notices([outage, outage, trace(ClaimStatus.SKIPPED)], LIMITS)
    assert len(notices) == 2 and notices[0].text.count("gdelt") == 1
    assert build_notices([trace(ClaimStatus.CHECKED)], LIMITS) == ()
