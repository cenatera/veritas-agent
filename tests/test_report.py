"""Pruebas del informe ejecutivo con datos ficticios."""

from datetime import UTC, datetime
from typing import Any

import pytest

from src.orchestrator import AuditResult
from src.report import render_markdown

ISSUE = {"kind": "La fuente limitó temporalmente las peticiones", "origin": "Búsqueda en gdelt",
         "detail": "Respuesta HTTP 429.", "attempts": 3}
NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
EXCERPT = "El puente mide 10 metros."


def result(**changes: Any) -> AuditResult:
    """Auditoría con una afirmación confirmada y otra sin comprobar."""
    trace = {"claim_id": "c1", "status": "Contrastada", "source_ids": ["s1"],
             "proposed_verdict": "Confirmada", **changes.pop("trace", {})}
    source = {"id": "s1", "url": "https://example.org/a", "title": "Puentes",
              "published_on": "2024-03-05", "retrieved_at": "2026-01-01T00:00:00Z",
              "content": EXCERPT, **changes.pop("source", {})}
    evidence = {"source_id": "s1", "excerpt": changes.pop("excerpt", EXCERPT), "stance": "respalda"}
    other = {**source, "id": "s2", "url": "https://example.net/b", "title": "Otra"}
    return AuditResult.model_validate({
        "report": {
            "document": "Mide 10 m. Es de 1900.",
            "claims": [
                {"id": "c1", "statement": "El puente mide 10 m.", "quote": "Mide 10 m",
                 "start": 0, "search_query": "puente"},
                {"id": "c2", "statement": "El puente es de 1900.", "quote": "Es de 1900",
                 "start": 11, "search_query": "puente"}],
            "sources": [source, other],
            "assessments": [
                {"claim_id": "c1", "verdict": "Confirmada", "confidence": "media",
                 "evidence": [evidence], "rationale": "Consta."},
                {"claim_id": "c2", "verdict": "Sin evidencia suficiente", "rationale": "Límite."}],
        },
        "traces": [trace, {"claim_id": "c2", "status": "No comprobada por límite"}],
        **changes})


def test_report_contains_verdicts_citations_and_dates() -> None:
    text = render_markdown(result(), NOW)
    assert "Generado: 2026-01-02 03:04 UTC" in text
    assert "Se contrastó **1 de 2 afirmaciones**" in text
    assert "1 confirmada, 0 refutadas, 0 en disputa y 1 sin evidencia suficiente" in text
    assert f"«{EXCERPT}» — [Puentes](<https://example.org/a>) (publicada: 2024-03-05" in text
    assert "| 2 | El puente es de 1900. | ○ Sin evidencia suficiente | — | No comprobada por límite |" in text
    assert "Alcance limitado" in text and "a 2 en «Alcance»" in text
    assert text.index("no citadas") < text.index("example.net/b")


def test_rendering_is_deterministic() -> None:
    assert render_markdown(result(), NOW) == render_markdown(result(), NOW)


def test_every_url_belongs_to_a_source() -> None:
    import re
    audit = result()
    urls = set(re.findall(r"\]\(<([^>]+)>\)", render_markdown(audit, NOW)))
    assert urls == {str(s.url) for s in audit.report.sources}


def test_missing_date_is_stated() -> None:
    assert "fecha de publicación no declarada" in render_markdown(result(source={"published_on": None}), NOW)


def test_issues_and_adjustments_are_visible() -> None:
    text = render_markdown(result(trace={
        "issues": [ISSUE], "proposed_verdict": "Refutada",
        "rejected_evidence": [{"reason": "La fuente no existe.", "evidence": {
            "source_id": "s9", "excerpt": "texto inventado", "stance": "respalda"}}]}), NOW)
    assert "HTTP 429" in text and "cobertura" in text and "Intentos: 3" in text
    assert "limitó temporalmente las peticiones" in text
    assert "no equivale a que la afirmación sea falsa" in text
    assert "ajustado: Refutada" in text and "La fuente no existe" in text
    assert "texto inventado" not in text


def test_rejected_rationale_is_never_published() -> None:
    text = render_markdown(result(trace={"rejected_rationale": {
        "text": "Ver https://inventada.test", "reason": "Contiene una URL."}}), NOW)
    assert "inventada" not in text and "Contiene una URL." in text


@pytest.mark.parametrize("hostile", [
    "<script>alert(1)</script>", "[clic](https://malo.test)", "![x](https://malo.test/i.png)",
    "a | b | c", "uno\n\n# Título falso", "`código`", "https://malo.test", "www.malo.test",
    "mailto:a@malo.test",
    "cuesta $5 y $9",
])
def test_source_text_cannot_inject_markup(hostile: str) -> None:
    text = render_markdown(result(
        source={"content": hostile, "title": hostile}, excerpt=hostile), NOW)
    for raw in ("<script", "](https://malo", "\n# Título falso", "a | b", "`código`",
                "https://malo", "www.malo", " $5", "a@malo"):
        assert raw not in text


def test_web_addresses_in_quotes_are_visibly_disabled() -> None:
    text = render_markdown(result(source={"content": "Ver https://malo.test"},
                                  excerpt="Ver https://malo.test"), NOW)
    assert r"https\[:\]//malo.test" in text and "&#58;" not in text


def test_url_cannot_break_the_link() -> None:
    text = render_markdown(result(source={"url": "https://example.org/a_(b)"}), NOW)
    assert "](<https://example.org/a_%28b%29>)" in text


def test_empty_audit_and_rejected_claims() -> None:
    audit = AuditResult.model_validate({
        "report": {"document": "Hola.", "claims": [], "sources": [], "assessments": []},
        "traces": [], "rejected_claims": [{"reason": "La cita no aparece.", "draft": {
            "statement": "dato fabricado", "quote": "x", "search_query": "x"}}]})
    text = render_markdown(audit, NOW)
    assert "ninguna afirmación comprobable" in text and "La cita no aparece." in text
    assert "dato fabricado" not in text and "## Detalle" not in text and "propuso 1 afirmación" in text


def test_naive_date_is_rejected() -> None:
    with pytest.raises(ValueError, match="zona horaria"):
        render_markdown(result(), datetime(2026, 1, 1))


def test_related_context_is_quoted_and_labelled() -> None:
    audit = result()
    data = audit.model_dump(mode="json")
    data["report"]["assessments"][1]["related"] = [{"source_id": "s1", "excerpt": EXCERPT}]
    text = render_markdown(AuditResult.model_validate(data), NOW)
    section = text[text.index("### 2."):]
    assert "Contexto relacionado" in section and "No respaldan ni contradicen" in section
    assert f"«{EXCERPT}» — [Puentes](<https://example.org/a>)" in section


def test_evidence_is_grouped_by_stance() -> None:
    text = render_markdown(result(), NOW)
    assert "**Evidencia a favor**" in text and "Evidencia en contra" not in text


def test_question_gets_reformulation_advice() -> None:
    audit = AuditResult.model_validate({
        "report": {"document": "¿Cuánto mide el puente?", "claims": [], "sources": [],
                   "assessments": []}, "traces": []})
    text = render_markdown(audit, NOW)
    assert "es una pregunta" in text and "Escriba como afirmación" in text
