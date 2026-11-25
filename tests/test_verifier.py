"""Pruebas del verificador con el SDK real y transporte HTTP simulado."""

import json
from typing import Any

import anthropic
import httpx2
import pytest

from src.agents.llm import ModelOutputError
from src.agents.verifier import ClaimVerifier, VerificationDraft, VerificationResult, check_draft
from src.schemas import AuditReport, Claim, RetrievedSource

CLAIM = Claim(id="c1", statement="El puente mide 10 metros.", search_query="puente", quote="mide 10 metros", start=10)
YES, NO = "El puente mide 10 metros en total.", "El puente no mide 10 metros, sino 12."


def source(id_: str, host: str, content: str = f"{YES} {NO}") -> RetrievedSource:
    """Fuente ficticia."""
    return RetrievedSource.model_validate({
        "id": id_, "url": f"https://{host}/a", "content": content,
        "retrieved_at": "2026-01-01T00:00:00Z"})


SOURCES = [source("s1", "uno.test"), source("s2", "dos.test")]


def ev(stance: str = "respalda", excerpt: str | None = None, source_id: str = "s1") -> dict[str, str]:
    """Evidencia ficticia."""
    return {"source_id": source_id, "stance": stance,
            "excerpt": excerpt if excerpt is not None else (YES if stance == "respalda" else NO)}


def draft(**changes: Any) -> dict[str, Any]:
    """Propuesta válida con los cambios indicados."""
    return {"evidence": [ev(), ev(source_id="s2")], "related": [], "verdict": "Confirmada",
            "confidence": "alta", "rationale": "Ambas fuentes lo indican.", **changes}


def check(sources: list[RetrievedSource] = SOURCES, **changes: Any) -> VerificationResult:
    """Comprueba una propuesta."""
    return check_draft(CLAIM, sources, VerificationDraft.model_validate(draft(**changes)))


def verifier(text: str, stop: str = "end_turn") -> tuple[ClaimVerifier, list[dict[str, Any]]]:
    """Verificador con respuesta fija y registro de peticiones."""
    sent: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop,
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

    return ClaimVerifier(anthropic.Anthropic(api_key="test", max_retries=0, http_client=httpx2.Client(
        transport=httpx2.MockTransport(handle)))), sent


def test_verifies_sends_sources_and_fits_report() -> None:
    agent, sent = verifier(json.dumps(draft()))
    result = agent.verify(CLAIM, SOURCES)
    assert result.assessment.verdict.value == "Confirmada"
    assert result.assessment.confidence is not None and result.assessment.confidence.value == "alta"
    blocks = sent[0]["messages"][0]["content"]
    assert [b["title"] for b in blocks[:2]] == ["s1", "s2"]
    assert blocks[0]["source"]["data"] == SOURCES[0].content
    assert CLAIM.statement in blocks[2]["text"]
    AuditReport(document="El puente mide 10 metros.", claims=(CLAIM,),
                sources=tuple(SOURCES), assessments=(result.assessment,))


def test_no_sources_makes_no_request() -> None:
    agent, sent = verifier("{}")
    result = agent.verify(CLAIM, [])
    assert result.assessment.verdict.value == "Sin evidencia suficiente"
    assert result.proposed_verdict is None and sent == []


def test_no_sources_needs_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ClaimVerifier().verify(CLAIM, [])


def test_duplicate_sources_are_rejected() -> None:
    agent, _ = verifier("{}")
    with pytest.raises(ValueError, match="duplicados"):
        agent.verify(CLAIM, [SOURCES[0], SOURCES[0]])


@pytest.mark.parametrize(("text", "stop"), [("{", "max_tokens"), ("", "refusal"), ("{}", "end_turn")])
def test_unusable_response_fails_loudly(text: str, stop: str) -> None:
    agent, _ = verifier(text, stop)
    with pytest.raises(ModelOutputError):
        agent.verify(CLAIM, SOURCES)


@pytest.mark.parametrize(("item", "reason"), [
    (ev(source_id="s9"), "no existe"),
    (ev(excerpt="El puente mide 99 metros"), "literalmente"),
    (ev(excerpt="puente mide"), "breve"),
])
def test_invented_evidence_becomes_insufficient(item: dict[str, str], reason: str) -> None:
    result = check(evidence=[item])
    assert result.assessment.verdict.value == "Sin evidencia suficiente"
    assert result.assessment.confidence is None
    assert result.assessment.evidence == ()
    assert reason in result.rejected[0].reason
    assert result.proposed_verdict is not None and result.proposed_verdict.value == "Confirmada"


@pytest.mark.parametrize(("verdict", "evidence", "expected"), [
    ("Refutada", [ev("contradice")], "Refutada"),
    ("En disputa", [ev(), ev("contradice")], "En disputa"),
    ("Sin evidencia suficiente", [ev()], "Sin evidencia suficiente"),
    ("Confirmada", [], "Sin evidencia suficiente"),
    ("Refutada", [ev()], "Sin evidencia suficiente"),
    ("Confirmada", [ev(), ev("contradice")], "En disputa"),
    ("En disputa", [ev()], "Sin evidencia suficiente"),
    ("En disputa", [ev(), ev("contradice", "inventado del todo")], "Sin evidencia suficiente"),
])
def test_verdict_must_match_checked_evidence(
    verdict: str, evidence: list[dict[str, str]], expected: str
) -> None:
    assert check(verdict=verdict, evidence=evidence).assessment.verdict.value == expected


def test_same_excerpt_cannot_support_and_refute() -> None:
    result = check(verdict="En disputa", evidence=[ev(), ev("contradice", YES)])
    assert result.assessment.verdict.value == "Sin evidencia suficiente"
    assert len(result.rejected) == 2


@pytest.mark.parametrize(("hosts", "expected"), [
    (("uno.test", "dos.test"), "alta"),
    (("uno.test", "www.uno.test"), "media"),
    (("es.wikipedia.org", "en.wikipedia.org"), "media"),
])
def test_high_confidence_needs_two_publishers(hosts: tuple[str, str], expected: str) -> None:
    result = check([source("s1", hosts[0]), source("s2", hosts[1])])
    assert result.assessment.confidence is not None
    assert result.assessment.confidence.value == expected


def test_rejected_evidence_lowers_high_confidence() -> None:
    result = check(evidence=[ev(), ev(source_id="s2"), ev(source_id="s9")])
    assert result.assessment.confidence is not None
    assert result.assessment.confidence.value == "media"


def test_missing_confidence_is_insufficient() -> None:
    assert check(confidence=None).assessment.verdict.value == "Sin evidencia suficiente"


@pytest.mark.parametrize("rationale", ["", "Ver https://inventada.test", "Se midió en 1999."])
def test_ungrounded_rationale_is_replaced(rationale: str) -> None:
    result = check(rationale=rationale)
    assert "descartada" in result.assessment.rationale
    assert result.rejected_rationale is not None
    assert result.rejected_rationale.text == rationale


def test_grounded_rationale_is_kept() -> None:
    result = check(rationale="Mide 10 metros.")
    assert result.assessment.rationale == "Mide 10 metros."
    assert result.rejected_rationale is None


def test_rationale_without_evidence_gets_useful_text() -> None:
    result = check(verdict="Sin evidencia suficiente", evidence=[], confidence=None,
                   rationale="Solo consta algo de 1999.")
    assert "no aportaron extractos" in result.assessment.rationale
    assert result.rejected_rationale is not None and "1999" in result.rejected_rationale.reason


CONTEXT = "El puente se abrió en 1950 al público."


def related(excerpt: str = CONTEXT, source_id: str = "s1") -> dict[str, str]:
    """Contexto ficticio."""
    return {"source_id": source_id, "excerpt": excerpt}


def check_context(**changes: Any) -> VerificationResult:
    """Comprueba una propuesta con contexto disponible en las fuentes."""
    sources = [source("s1", "uno.test", f"{YES} {NO} {CONTEXT}"), SOURCES[1]]
    return check(sources, **changes)


@pytest.mark.parametrize(("verdict", "evidence"), [
    ("Sin evidencia suficiente", []), ("Refutada", [ev("contradice")]),
    ("En disputa", [ev(), ev("contradice")]),
])
def test_literal_context_is_kept_without_changing_the_verdict(
    verdict: str, evidence: list[dict[str, str]]
) -> None:
    result = check_context(verdict=verdict, evidence=evidence, related=[related()])
    assert result.assessment.verdict.value == verdict
    assert result.assessment.related[0].excerpt == CONTEXT


def test_context_alone_never_confirms() -> None:
    result = check_context(verdict="Confirmada", evidence=[], related=[related()])
    assert result.assessment.verdict.value == "Sin evidencia suficiente"


@pytest.mark.parametrize(("item", "reason"), [
    (related("Dato inventado por completo"), "literalmente"), (related(source_id="s9"), "no existe"),
    (related("se abrió"), "breve"), (related(NO), "ya se propuso"),
])
def test_unverifiable_context_is_rejected(item: dict[str, str], reason: str) -> None:
    result = check_context(verdict="Refutada", evidence=[ev("contradice")], related=[item])
    assert result.assessment.related == () and reason in result.rejected_related[0].reason


def test_context_is_dropped_when_confirmed_and_capped() -> None:
    assert check_context(related=[related()]).assessment.related == ()
    many = [related(), related(CONTEXT[:30]), related(CONTEXT[:25]), related(CONTEXT[:20]), related()]
    result = check_context(verdict="Sin evidencia suficiente", evidence=[], related=many)
    assert len(result.assessment.related) == 3 and len(result.rejected_related) == 2


def test_rationale_may_use_figures_from_context() -> None:
    result = check_context(verdict="Sin evidencia suficiente", evidence=[], confidence=None,
                           related=[related()], rationale="Solo consta que se abrió en 1950.")
    assert result.rejected_rationale is None


def test_report_rejects_untraceable_context() -> None:
    result = check_context(verdict="Refutada", evidence=[ev("contradice")], related=[related()])
    with pytest.raises(ValueError, match="Extracto ausente"):
        AuditReport(document="El puente mide 10 metros.", claims=(CLAIM,), sources=tuple(SOURCES),
                    assessments=(result.assessment,))
