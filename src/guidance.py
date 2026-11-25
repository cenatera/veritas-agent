"""Explicaciones para el lector: qué ocurrió y qué puede hacerse.

Los textos son fijos y se eligen según datos ya comprobados; ningún modelo
interviene, de modo que no pueden introducir hechos sobre la afirmación.
"""

import re
from collections.abc import Sequence
from enum import Enum

from pydantic import Field

from src.orchestrator import (
    MAX_CLAIMS,
    AuditLimits,
    ClaimStatus,
    ClaimTrace,
    IssueKind,
)
from src.schemas import ClaimAssessment, VeritasModel, Verdict

MAX_SOURCES = 4
MAX_SOURCE_CHARS = 200_000
NOT_FALSE = "«Sin evidencia suficiente» no equivale a que la afirmación sea falsa."


class Cause(str, Enum):
    """Motivo por el que una afirmación carece de veredicto concluyente."""

    CLAIM_LIMIT = "Tope de afirmaciones alcanzado"
    OUTAGE = "Incidencia de red o de las fuentes"
    NO_RESULTS = "La búsqueda no encontró páginas"
    UNUSABLE = "Las páginas encontradas no pudieron utilizarse"
    VERIFIER = "La verificación no pudo completarse"
    ADJUSTED = "La evidencia no sostiene el veredicto propuesto"
    NO_EVIDENCE = "Falta de evidencia literal en las fuentes consultadas"


class Guidance(VeritasModel):
    """Causa, explicación y acción recomendada."""

    cause: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    action: str = Field(min_length=1)


class Notice(VeritasModel):
    """Aviso general del informe."""

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


def _limit_action(total: int) -> str:
    """Acción para una afirmación que quedó fuera del tope."""
    if total <= MAX_CLAIMS:
        return (
            f"Aumente «Máximo de afirmaciones» a {total} en «Alcance» y vuelva a verificar. "
            "El coste estimado aumentará."
        )
    return (
        f"El máximo disponible es {MAX_CLAIMS}: divida el texto en partes y verifíquelas "
        "por separado."
    )


def explain_claim(
    assessment: ClaimAssessment, trace: ClaimTrace, total: int, limits: AuditLimits | None
) -> Guidance | None:
    """Explica un resultado no concluyente; ``None`` si el veredicto es concluyente."""
    if assessment.verdict is not Verdict.INSUFFICIENT:
        return None

    kinds = {issue.kind for issue in trace.issues}
    outage = any(kind.is_outage for kind in kinds)
    if trace.status is ClaimStatus.SKIPPED:
        cap = f" de {limits.max_claims}" if limits else ""
        return Guidance(
            cause=Cause.CLAIM_LIMIT.value,
            explanation=f"Se extrajeron {total} afirmaciones y se alcanzó el máximo{cap} fijado "
            "en «Alcance». No se buscaron fuentes para esta afirmación.",
            action=_limit_action(total),
        )
    if trace.status is ClaimStatus.FAILED:
        return Guidance(
            cause=Cause.VERIFIER.value,
            explanation="Se obtuvieron fuentes, pero el modelo verificador no completó una "
            f"respuesta utilizable. {NOT_FALSE}",
            action="Vuelva a verificar. Si se repite, reduzca las fuentes o los caracteres "
            "por afirmación en «Alcance».",
        )
    if trace.status is ClaimStatus.NO_SOURCES:
        if outage:
            return Guidance(
                cause=Cause.OUTAGE.value,
                explanation="No se obtuvo ninguna fuente porque al menos una limitó "
                f"temporalmente las peticiones o falló al responder. {NOT_FALSE}",
                action="Espere unos minutos y vuelva a verificar. Las incidencias indican qué "
                "fuente falló.",
            )
        if IssueKind.NO_RESULTS in kinds:
            return Guidance(
                cause=Cause.NO_RESULTS.value,
                explanation=f"Ningún buscador devolvió páginas para esta consulta. {NOT_FALSE}",
                action="Reformule la afirmación con nombres completos, lugar y fecha, o pruebe "
                "otro idioma de Wikipedia.",
            )
        action = "Contraste la afirmación manualmente o reformúlela con datos más precisos."
        if IssueKind.TOO_LARGE in kinds and limits and limits.max_source_chars < MAX_SOURCE_CHARS:
            action = "Aumente «Caracteres de fuentes por afirmación» en «Alcance» y vuelva a verificar."
        return Guidance(
            cause=Cause.UNUSABLE.value,
            explanation="Se encontraron páginas, pero ninguna pudo utilizarse; las incidencias "
            f"detallan el motivo. {NOT_FALSE}",
            action=action,
        )

    proposed = trace.proposed_verdict
    if proposed is not None and proposed is not Verdict.INSUFFICIENT:
        cause = Cause.ADJUSTED
        explanation = (
            f"El modelo propuso «{proposed.value}», pero los extractos comprobados no sostienen "
            "ese veredicto y no se acepta."
        )
    else:
        cause = Cause.NO_EVIDENCE
        count = len(trace.source_ids)
        noun = "fuente consultada" if count == 1 else "fuentes consultadas"
        explanation = (
            f"Se contrastó con {count} {noun}, sin obtener extractos literales que permitan "
            "confirmarla o refutarla."
        )
    actions = []
    if outage:
        explanation += " Además, alguna fuente falló, por lo que la búsqueda fue incompleta."
        actions.append("Espere unos minutos y vuelva a verificar.")
    if limits and limits.max_sources < MAX_SOURCES:
        actions.append("Aumente «Fuentes por afirmación» en «Alcance».")
    actions.append("Precise nombres, lugar y fecha, o contraste la afirmación manualmente.")
    return Guidance(
        cause=cause.value, explanation=f"{explanation} {NOT_FALSE}", action=" ".join(actions)
    )


def looks_like_question(document: str) -> bool:
    """Indica si todas las frases del texto son preguntas."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", document) if part.strip()]
    return bool(sentences) and all(
        sentence.startswith("¿") or sentence.rstrip("»\"') ").endswith("?")
        for sentence in sentences
    )


def explain_no_claims(document: str, rejected: int) -> Guidance:
    """Explica por qué no hay afirmaciones y cómo reformular el texto."""
    if looks_like_question(document):
        return Guidance(
            cause="El texto es una pregunta, no una afirmación",
            explanation="El sistema contrasta afirmaciones que pueden ser ciertas o falsas. Una "
            "pregunta no afirma ningún hecho, así que no hay nada que confirmar o refutar.",
            action="Escriba como afirmación la respuesta que quiera comprobar. Por ejemplo, en "
            "lugar de «¿Cuándo se inauguró el puente?», escriba «El puente se inauguró en "
            "[año]», con el nombre y el dato concretos.",
        )
    if rejected:
        return Guidance(
            cause="Ninguna propuesta superó la comprobación",
            explanation=f"El modelo propuso {rejected} "
            f"{'afirmación, que no pudo' if rejected == 1 else 'afirmaciones, que no pudieron'} "
            "vincularse literalmente con el texto. Por rigor, no se contrastó nada.",
            action="Simplifique el texto: una idea por frase, con los nombres y las cifras "
            "escritos de forma explícita. Después, vuelva a verificar.",
        )
    return Guidance(
        cause="No se encontraron hechos comprobables",
        explanation="El texto puede contener opiniones, predicciones, preguntas o generalidades, "
        "que no pueden contrastarse con fuentes.",
        action="Escriba una afirmación factual con quién, qué, dónde y cuándo. Por ejemplo: "
        "«[Persona o entidad] hizo [acción concreta] en [lugar] el [fecha]».",
    )


def build_notices(
    traces: Sequence[ClaimTrace], limits: AuditLimits | None
) -> tuple[Notice, ...]:
    """Avisos generales que condicionan la lectura del informe."""
    notices = []
    outages = dict.fromkeys(
        (issue.kind, issue.origin)
        for trace in traces
        for issue in trace.issues
        if issue.kind.is_outage
    )
    if outages:
        names = "; ".join(f"{origin}: {kind.value.lower()}" for kind, origin in outages)
        notices.append(Notice(
            title="Fuentes con incidencias",
            text="Algunas limitaron temporalmente las peticiones o "
            "fallaron al responder, por lo que la cobertura puede ser incompleta. "
            f"{NOT_FALSE} Detalle — {names}.",
        ))
    skipped = sum(trace.status is ClaimStatus.SKIPPED for trace in traces)
    if skipped:
        cap = f" ({limits.max_claims})" if limits else ""
        noun = "afirmación no se comprobó" if skipped == 1 else "afirmaciones no se comprobaron"
        notices.append(Notice(
            title="Alcance limitado",
            text=f"{skipped} {noun} por el máximo fijado en «Alcance»{cap}. "
            f"{_limit_action(len(traces))}",
        ))
    if any(trace.status is ClaimStatus.FAILED for trace in traces):
        notices.append(Notice(
            title="Verificación incompleta",
            text=f"El modelo verificador no completó alguna afirmación. {NOT_FALSE}",
        ))
    return tuple(notices)
