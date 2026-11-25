"""Informe ejecutivo en Markdown.

La misma vista alimenta el archivo descargable y la interfaz. Se compone sin modelos de lenguaje, solo con datos ya validados. El texto ajeno se
escapa para que una fuente no pueda inyectar formato, enlaces ni HTML. Las
justificaciones descartadas no se publican: solo constan en el registro JSON.
"""

import re
from collections import Counter
from datetime import UTC, datetime

from pydantic import Field

from src.guidance import Guidance, build_notices, explain_claim, explain_no_claims
from src.orchestrator import AuditResult, ClaimStatus, ClaimTrace
from src.schemas import (
    Claim,
    ClaimAssessment,
    RetrievedSource,
    Stance,
    VeritasModel,
    Verdict,
)

SYMBOLS = {
    Verdict.SUPPORTED: "✓",
    Verdict.REFUTED: "✗",
    Verdict.DISPUTED: "≠",
    Verdict.INSUFFICIENT: "○",
}
STANCE_TITLES = {
    Stance.SUPPORTS: "Evidencia a favor",
    Stance.REFUTES: "Evidencia en contra",
}

METHOD = (
    "Claude propone afirmaciones, extractos y veredictos. El código comprueba que cada "
    "cita sea literal y que la evidencia sea compatible con el veredicto. El contexto "
    "relacionado también es literal, pero no influye en el veredicto. Un extracto "
    "literal puede estar fuera de contexto y una fuente puede equivocarse: este informe "
    "apoya la revisión editorial, no la sustituye. En los textos citados, las direcciones "
    "web y de correo se desactivan con corchetes, como https\\[:\\]//."
)


class ClaimView(VeritasModel):
    """Ficha de una afirmación; los textos ya son Markdown seguro."""

    number: int = Field(ge=1)
    title: str
    verdict: Verdict
    confidence: str
    status: str
    body: str


class ReportView(VeritasModel):
    """Informe por secciones, en Markdown seguro."""

    title: str = "Informe de verificación"
    generated: str
    lead: str
    counts: dict[Verdict, int]
    notices: tuple[str, ...] = ()
    guidance: str | None = None
    table: str = ""
    claims: tuple[ClaimView, ...] = ()
    appendix: str


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


def _guidance(guidance: Guidance) -> str:
    """Causa, explicación y acción, en tres párrafos."""
    return "\n\n".join([
        f"**Causa:** {_text(guidance.cause)}.",
        f"**Qué ocurrió:** {_text(guidance.explanation)}",
        f"**Qué puede hacer:** {_text(guidance.action)}",
    ])


def _plural(count: int, singular: str, plural: str) -> str:
    """Cantidad con su sustantivo concordado."""
    return f"{count} {singular if count == 1 else plural}"


def _lead(result: AuditResult) -> str:
    """Frase inicial con el resultado general."""
    total = len(result.report.claims)
    if not total:
        return "No se identificó ninguna afirmación comprobable en el texto."
    checked = sum(trace.status is ClaimStatus.CHECKED for trace in result.traces)
    counts = Counter(item.verdict for item in result.report.assessments)
    parts = [
        _plural(counts[Verdict.SUPPORTED], "confirmada", "confirmadas"),
        _plural(counts[Verdict.REFUTED], "refutada", "refutadas"),
        f"{counts[Verdict.DISPUTED]} en disputa",
        f"{counts[Verdict.INSUFFICIENT]} sin evidencia suficiente",
    ]
    verb = "Se contrastó" if checked == 1 else "Se contrastaron"
    return (
        f"{verb} **{checked} de {_plural(total, 'afirmación', 'afirmaciones')}**. "
        f"Resultado: {', '.join(parts[:-1])} y {parts[-1]}."
    )


def _table(result: AuditResult) -> str:
    """Tabla de un vistazo."""
    lines = ["| N.º | Afirmación | Veredicto | Confianza | Estado |",
             "|:-:|:--|:--|:--|:--|"]
    traces = {trace.claim_id: trace for trace in result.traces}
    by_claim = {item.claim_id: item for item in result.report.assessments}
    for number, claim in enumerate(result.report.claims, 1):
        item = by_claim[claim.id]
        lines.append(
            f"| {number} | {_text(claim.statement)} | {SYMBOLS[item.verdict]} "
            f"{item.verdict.value} | {_confidence(item)} | {traces[claim.id].status.value} |"
        )
    return "\n".join(lines)


def _claim_body(
    claim: Claim,
    item: ClaimAssessment,
    trace: ClaimTrace,
    sources: dict[str, RetrievedSource],
    guidance: Guidance | None,
) -> str:
    """Cuerpo de la ficha de una afirmación."""
    blocks = [f"**En el texto:** «{_text(claim.quote)}»"]
    if guidance:
        blocks.append(_guidance(guidance))
    if trace.status is ClaimStatus.CHECKED:
        blocks.append(f"**Justificación:** {_text(item.rationale)}")
    for stance, title in STANCE_TITLES.items():
        quotes = [
            f"- «{_text(e.excerpt)}» — {_link(sources[e.source_id])}"
            for e in item.evidence
            if e.stance is stance
        ]
        if quotes:
            note = " (no bastó para un veredicto)" if item.verdict is Verdict.INSUFFICIENT else ""
            blocks.append("\n".join([f"**{title}{note}**", "", *quotes]))
    if item.related:
        blocks.append("\n".join([
            "**Contexto relacionado** — citas literales sobre el tema, seleccionadas por el "
            "modelo. No respaldan ni contradicen la afirmación.",
            "",
            *(f"- «{_text(r.excerpt)}» — {_link(sources[r.source_id])}" for r in item.related),
        ]))
    notes = [_text(issue.describe()) for issue in trace.issues]
    if trace.proposed_verdict is not None and trace.proposed_verdict is not item.verdict:
        notes.append(f"Veredicto propuesto y ajustado: {trace.proposed_verdict.value}.")
    notes += [f"Evidencia descartada: {_text(r.reason)}" for r in trace.rejected_evidence]
    notes += [f"Contexto descartado: {_text(r.reason)}" for r in trace.rejected_related]
    if trace.rejected_rationale:
        notes.append(f"Justificación descartada: {_text(trace.rejected_rationale.reason)}")
    if notes:
        blocks.append("\n".join(["**Incidencias**", "", *(f"- {note}" for note in notes)]))
    return "\n\n".join(blocks)


def _appendix(result: AuditResult) -> str:
    """Fuentes no citadas, descartes y método."""
    used = {
        e.source_id for item in result.report.assessments for e in (*item.evidence, *item.related)
    }
    blocks = []
    unused = [source for source in result.report.sources if source.id not in used]
    if unused:
        blocks.append("\n".join(["## Fuentes consultadas y no citadas", "",
                                 *(f"- {_link(source)}" for source in unused)]))
    if result.rejected_claims:
        blocks.append("\n".join(["## Propuestas de afirmación descartadas", "",
                                 *(f"- {_text(r.reason)}" for r in result.rejected_claims)]))
    blocks.append(f"## Método y límites\n\n{METHOD}")
    return "\n\n".join(blocks)


def build_view(result: AuditResult, generated_at: datetime | None = None) -> ReportView:
    """Compone las secciones del informe.

    Raises:
        ValueError: ``generated_at`` no tiene zona horaria.
    """
    moment = generated_at or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("La fecha del informe requiere zona horaria.")

    report = result.report
    total = len(report.claims)
    sources = {source.id: source for source in report.sources}
    by_claim = {item.claim_id: item for item in report.assessments}
    traces = {trace.claim_id: trace for trace in result.traces}
    counts = Counter(item.verdict for item in report.assessments)
    claims = []
    for number, claim in enumerate(report.claims, 1):
        item, trace = by_claim[claim.id], traces[claim.id]
        claims.append(ClaimView(
            number=number,
            title=_text(claim.statement),
            verdict=item.verdict,
            confidence=_confidence(item),
            status=trace.status.value,
            body=_claim_body(
                claim, item, trace, sources, explain_claim(item, trace, total, result.limits)
            ),
        ))
    return ReportView(
        generated=moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        lead=_lead(result),
        counts={verdict: counts[verdict] for verdict in Verdict},
        notices=tuple(
            f"**{_text(notice.title)}.** {_text(notice.text)}"
            for notice in build_notices(result.traces, result.limits)
        ),
        guidance=None if total else _guidance(
            explain_no_claims(report.document, len(result.rejected_claims))
        ),
        table=_table(result) if total else "",
        claims=tuple(claims),
        appendix=_appendix(result),
    )


def view_to_markdown(view: ReportView) -> str:
    """Une las secciones en un único documento Markdown."""
    blocks = [f"# {view.title}", f"Generado: {view.generated}", "## Resumen ejecutivo", view.lead]
    if view.guidance:
        blocks.append(view.guidance)
    blocks += [f"> {notice}" for notice in view.notices]
    if view.claims:
        blocks += [view.table, "## Detalle por afirmación"]
    for claim in view.claims:
        blocks += [
            f"### {claim.number}. {claim.title}",
            f"**{SYMBOLS[claim.verdict]} {claim.verdict.value}** · "
            f"Confianza: {claim.confidence} · Estado: {claim.status}",
            claim.body,
        ]
    return "\n\n".join([*blocks, view.appendix]) + "\n"


def render_markdown(result: AuditResult, generated_at: datetime | None = None) -> str:
    """Compone el informe ejecutivo completo.

    Raises:
        ValueError: ``generated_at`` no tiene zona horaria.
    """
    return view_to_markdown(build_view(result, generated_at))
