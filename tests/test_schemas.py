"""Pruebas de los contratos de datos. Todos los datos son ficticios."""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import AuditReport, ClaimAssessment, RetrievedSource

SOURCE: dict[str, Any] = {
    "id": "s1",
    "url": "https://example.org/dato",
    "retrieved_at": "2026-01-01T00:00:00Z",
    "content": "El puente mide 10 metros. El puente no mide 10 metros.",
}
SUPPORT = {"source_id": "s1", "excerpt": "El puente mide 10 metros", "stance": "respalda"}
REFUTE = {"source_id": "s1", "excerpt": "no mide 10 metros", "stance": "contradice"}
CLAIM = {"id": "c1", "statement": "El puente mide 10 m.", "search_query": "puente", "quote": "mide 10 m", "start": 10}


def assessment(**changes: Any) -> dict[str, Any]:
    """Veredicto válido con los cambios indicados."""
    base = {"claim_id": "c1", "verdict": "Confirmada", "confidence": "alta",
            "evidence": [SUPPORT], "rationale": "La fuente lo afirma."}
    return {**base, **changes}


def report(**changes: Any) -> dict[str, Any]:
    """Informe válido con los cambios indicados."""
    base = {"document": "El puente mide 10 m de largo.", "claims": [CLAIM],
            "sources": [SOURCE], "assessments": [assessment()]}
    return {**base, **changes}


def test_valid_report_survives_json_round_trip() -> None:
    parsed = AuditReport.model_validate(report())
    assert AuditReport.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(("verdict", "evidence"), [
    ("Confirmada", [SUPPORT]), ("Refutada", [REFUTE]), ("En disputa", [SUPPORT, REFUTE]),
])
def test_conclusive_verdict_accepts_matching_evidence(verdict: str, evidence: list[Any]) -> None:
    ClaimAssessment.model_validate(assessment(verdict=verdict, evidence=evidence))


@pytest.mark.parametrize(("verdict", "evidence"), [
    ("Confirmada", []), ("Refutada", []), ("En disputa", []),
    ("Confirmada", [REFUTE]), ("Refutada", [SUPPORT]), ("En disputa", [SUPPORT]),
    ("Confirmada", [SUPPORT, REFUTE]), ("Refutada", [SUPPORT, REFUTE]),
])
def test_conclusive_verdict_rejects_unmatched_evidence(verdict: str, evidence: list[Any]) -> None:
    with pytest.raises(ValidationError, match="no sostiene"):
        ClaimAssessment.model_validate(assessment(verdict=verdict, evidence=evidence))


def test_insufficient_needs_neither_evidence_nor_confidence() -> None:
    result = ClaimAssessment.model_validate(
        assessment(verdict="Sin evidencia suficiente", confidence=None, evidence=[]))
    assert result.verdict.value == "Sin evidencia suficiente"


@pytest.mark.parametrize("changes", [
    {"verdict": "Sin evidencia suficiente", "evidence": []},
    {"confidence": None}, {"verdict": "Probable"}, {"inventado": 1},
])
def test_assessment_rejects_invalid_fields(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ClaimAssessment.model_validate(assessment(**changes))


@pytest.mark.parametrize(("changes", "message"), [
    ({"claims": [{**CLAIM, "quote": "mide 99 m"}]}, "no está en el documento"),
    ({"claims": [{**CLAIM, "start": 0}]}, "no está en el documento"),
    ({"sources": []}, "Fuente inexistente"),
    ({"sources": [SOURCE, SOURCE]}, "fuente duplicados"),
    ({"claims": [CLAIM, CLAIM]}, "afirmación duplicados"),
    ({"assessments": []}, "exactamente un veredicto"),
    ({"assessments": [assessment(), assessment()]}, "exactamente un veredicto"),
    ({"assessments": [assessment(evidence=[{**SUPPORT, "excerpt": "Inventado"}])]},
     "Extracto ausente"),
])
def test_report_rejects_untraceable_data(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AuditReport.model_validate(report(**changes))


@pytest.mark.parametrize("changes", [
    {"url": "no-es-url"}, {"url": "ftp://example.org"},
    {"retrieved_at": "2026-01-01T00:00:00"}, {"published_on": "ayer"},
])
def test_source_rejects_invalid_provenance(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RetrievedSource.model_validate({**SOURCE, **changes})


def test_models_are_immutable() -> None:
    parsed = AuditReport.model_validate(report())
    with pytest.raises(ValidationError):
        parsed.document = "otro"  # type: ignore[misc]
