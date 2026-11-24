"""Informe ejecutivo en Markdown.

Se compone sin modelos de lenguaje, solo con datos ya validados. El texto ajeno se
escapa para que una fuente no pueda inyectar formato, enlaces ni HTML. Las
justificaciones descartadas no se publican: solo constan en el registro JSON.
"""

import re
from collections import Counter
from datetime import UTC, datetime

from src.orchestrator import AuditResult, ClaimStatus, ClaimTrace
from src.schemas import Claim, ClaimAssessment, RetrievedSource, Verdict

METHOD = (
    "Claude propone afirmaciones, extractos y veredictos. El código comprueba que cada "
    "cita sea literal y que la evidencia sea compatible con el veredicto. Un extracto "
    "literal puede estar fuera de contexto y una fuente puede equivocarse: este informe "
    "apoya la revisión editorial, no la sustituye. En los textos citados, las direcciones "
    "web y de correo se desactivan con corchetes, como https\\[:\\]//."
)


def _text(value: str) -> str:
    """Texto en una línea, sin significado especial en Markdown o HTML."""
    flat = " ".join(value.split())
    flat = flat.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    flat = re.sub(r"([\\`*_{}\[\]()|~#!$])", r"\\\1", flat)
    # Los visores enlazan URL y correos aunque usen entidades: la marca es visible.
    flat = re.sub(r"(?i)(https?|ftp)://", r"\1\\[:\\]//", flat)
    flat = re.sub(r"(?i)\bwww\.", r"www\\[.\\]", flat)
    flat = re.sub(r"(?<=\w)@(?=\w)", r"\\[@\\]", flat)
    return re.sub(r"^([-+=]|\d+\.)", r"\\\1", flat)


def _link(source: RetrievedSource) -> str:
    """Enlace a la URL descargada, con título y fechas."""
    url = str(source.url)
    for character in "<>()[]\\ |":
        url = url.replace(character, f"%{ord(character):02X}")
    published = (
        f"publicada: {source.published_on.isoformat()}"
        if source.published_on
        else "fecha de publicación no declarada"
    )
    title = _text(source.title or source.url.host or "Fuente")
    retrieved = source.retrieved_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"[{title}](<{url}>) ({published}; consultada: {retrieved})"


def _confidence(assessment: ClaimAssessment) -> str:
    """Nivel de confianza, o una raya si no procede."""
    return assessment.confidence.value if assessment.confidence else "—"


def _summary(result: AuditResult) -> list[str]:
    """Recuento por veredicto, avisos y tabla."""
    report = result.report
    checked = sum(trace.status is ClaimStatus.CHECKED for trace in result.traces)
    total = len(report.claims)
    lines = ["## Resumen", "", f"Afirmaciones contrastadas: **{checked} de {total}**.", ""]
    if not total:
        return [*lines, "No se aceptó ninguna afirmación verificable.", ""]

    counts = Counter(item.verdict for item in report.assessments)
    lines += [f"- {verdict.value}: {counts[verdict]}" for verdict in Verdict]
    if checked < total:
        lines += ["", "> **Aviso:** hay afirmaciones sin contrastar. Su etiqueta «Sin evidencia "
                  "suficiente» no significa que se buscara y no existiera evidencia."]
    if any(trace.issues for trace in result.traces):
        lines += ["", "> **Aviso:** hubo incidencias de búsqueda o descarga; la cobertura "
                  "puede ser incompleta."]
    lines += ["", "| N.º | Afirmación | Veredicto | Confianza | Estado |",
              "|---|---|---|---|---|"]
    traces = {trace.claim_id: trace for trace in result.traces}
    by_claim = {item.claim_id: item for item in report.assessments}
    for number, claim in enumerate(report.claims, 1):
        item = by_claim[claim.id]
        lines.append(
            f"| {number} | {_text(claim.statement)} | {item.verdict.value} | "
            f"{_confidence(item)} | {traces[claim.id].status.value} |"
        )
    return [*lines, ""]


def _detail(
    number: int,
    claim: Claim,
    item: ClaimAssessment,
    trace: ClaimTrace,
    sources: dict[str, RetrievedSource],
) -> list[str]:
    """Ficha de una afirmación."""
    lines = [
        f"### {number}. {_text(claim.statement)}", "",
        f"- **Veredicto:** {item.verdict.value}",
        f"- **Confianza:** {_confidence(item)}",
        f"- **Estado:** {trace.status.value}",
        f"- **Cita original:** «{_text(claim.quote)}»",
        f"- **Justificación:** {_text(item.rationale)}", "",
    ]
    if item.evidence:
        lines += ["**Evidencia**", ""]
        lines += [
            f"- *{evidence.stance.value.capitalize()}:* «{_text(evidence.excerpt)}» — "
            f"{_link(sources[evidence.source_id])}"
            for evidence in item.evidence
        ]
        lines.append("")
    notes = [_text(issue) for issue in trace.issues]
    if trace.proposed_verdict is not None and trace.proposed_verdict is not item.verdict:
        notes.append(f"Veredicto propuesto y ajustado: {trace.proposed_verdict.value}.")
    notes += [f"Evidencia descartada: {_text(r.reason)}" for r in trace.rejected_evidence]
    if trace.rejected_rationale:
        notes.append(f"Justificación descartada: {_text(trace.rejected_rationale.reason)}")
    if notes:
        lines += ["**Incidencias**", "", *(f"- {note}" for note in notes), ""]
    return lines


def render_markdown(result: AuditResult, generated_at: datetime | None = None) -> str:
    """Compone el informe ejecutivo de una auditoría.

    Raises:
        ValueError: ``generated_at`` no tiene zona horaria.
    """
    moment = generated_at or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("La fecha del informe requiere zona horaria.")

    report = result.report
    sources = {source.id: source for source in report.sources}
    by_claim = {item.claim_id: item for item in report.assessments}
    traces = {trace.claim_id: trace for trace in result.traces}
    lines = ["# Informe de verificación", "",
             f"Generado: {moment.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')}", ""]
    lines += _summary(result)
    if report.claims:
        lines += ["## Detalle", ""]
    for number, claim in enumerate(report.claims, 1):
        lines += _detail(number, claim, by_claim[claim.id], traces[claim.id], sources)

    cited = {e.source_id for item in report.assessments for e in item.evidence}
    unused = [source for source in report.sources if source.id not in cited]
    if unused:
        lines += ["## Fuentes consultadas y no citadas", ""]
        lines += [*(f"- {_link(source)}" for source in unused), ""]
    if result.rejected_claims:
        lines += ["## Propuestas de afirmación descartadas", ""]
        lines += [f"- {_text(item.reason)}" for item in result.rejected_claims]
        lines.append("")
    return "\n".join([*lines, "## Método y límites", "", METHOD, ""])
