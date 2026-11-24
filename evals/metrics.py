"""Casos, puntuación y métricas de la evaluación, sin red ni modelos."""

import json
import math
from collections import Counter
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pydantic import Field, HttpUrl, model_validator

from src.orchestrator import AuditResult, ClaimStatus
from src.schemas import VeritasModel, Verdict

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


class Category(str, Enum):
    """Tipo de afirmación."""

    TRUE = "cierta"
    FALSE = "falsa"
    MISLEADING = "engañosa"
    OUTDATED = "desactualizada"
    UNVERIFIABLE = "no comprobable"


class Reference(VeritasModel):
    """Respaldo humano del veredicto esperado; el pipeline no lo recibe."""

    url: HttpUrl
    excerpt: str = Field(min_length=10)


class EvalCase(VeritasModel):
    """Afirmación con veredicto conocido."""

    id: str = Field(min_length=1)
    category: Category
    expected: Verdict
    text: str = Field(min_length=1)
    reference: Reference | None = None

    @model_validator(mode="after")
    def _reference_matches_verdict(self) -> "EvalCase":
        """Solo los casos no comprobables carecen de referencia."""
        if (self.expected is Verdict.INSUFFICIENT) != (self.reference is None):
            raise ValueError("Todo veredicto concluyente requiere una referencia.")
        return self


class Outcome(str, Enum):
    """Puntuación de un caso."""

    CORRECT = "acierto"
    ABSTAINED = "abstención"
    WRONG = "error"
    EXTRACTION = "extracción inesperada"
    INCOMPLETE = "sin completar"


class CaseResult(VeritasModel):
    """Resultado puntuado de un caso."""

    case_id: str
    fingerprint: str
    outcome: Outcome
    predicted: Verdict | None = None
    degraded: bool = Field(default=False, description="Hubo fallos de búsqueda o descarga.")
    usd: float = Field(default=0.0, ge=0)
    error: str | None = None


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    """Carga los casos y rechaza identificadores repetidos."""
    cases = [EvalCase.model_validate(item) for item in json.loads(path.read_text("utf-8"))]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Hay identificadores de caso duplicados.")
    return cases


def score(case: EvalCase, audit: AuditResult, fingerprint: str, usd: float) -> CaseResult:
    """Puntúa una auditoría. Un error es afirmar algo concluyente e incorrecto."""
    predicted: Verdict | None = None
    degraded = any(trace.issues for trace in audit.traces)
    if len(audit.report.claims) != 1:
        outcome = Outcome.EXTRACTION
    elif audit.traces[0].status in {ClaimStatus.FAILED, ClaimStatus.SKIPPED}:
        outcome = Outcome.INCOMPLETE
    else:
        predicted = audit.report.assessments[0].verdict
        if predicted is case.expected:
            outcome = Outcome.CORRECT
        elif predicted is Verdict.INSUFFICIENT:
            outcome = Outcome.ABSTAINED
        else:
            outcome = Outcome.WRONG
    return CaseResult(case_id=case.id, fingerprint=fingerprint, outcome=outcome,
                      predicted=predicted, degraded=degraded, usd=usd)


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Intervalo de Wilson al 95 % para una proporción."""
    if total <= 0:
        raise ValueError("El total debe ser positivo.")
    z = 1.96
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / (1 + z * z / total)
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(cases: Sequence[EvalCase], results: Sequence[CaseResult]) -> str:
    """Resumen Markdown. Los casos no ejecutados nunca se ocultan."""
    by_id = {result.case_id: result for result in results}
    done = [case for case in cases if case.id in by_id]
    counts = Counter(by_id[case.id].outcome for case in done)
    lines = ["# Evaluación", "", f"Casos ejecutados: {len(done)} de {len(cases)}.", ""]
    lines += [f"- {outcome.value}: {counts[outcome]}" for outcome in Outcome]
    lines.append(f"- con incidencias de búsqueda o descarga: "
                 f"{sum(by_id[case.id].degraded for case in done)}")
    if done:
        hits = counts[Outcome.CORRECT]
        low, high = wilson(hits, len(done))
        lines += ["", f"Aciertos: {hits}/{len(done)} ({hits / len(done):.0%}; "
                  f"IC 95 %: {low:.0%}–{high:.0%})."]
        conclusive = [case for case in done if by_id[case.id].predicted
                      not in {None, Verdict.INSUFFICIENT}]
        if conclusive:
            right = sum(by_id[case.id].outcome is Outcome.CORRECT for case in conclusive)
            lines.append(f"Precisión de los veredictos concluyentes: {right}/{len(conclusive)}.")
    lines += ["", "| Categoría | " + " | ".join(o.value for o in Outcome) + " |",
              "|---|" + "---|" * len(Outcome)]
    for category in Category:
        row = Counter(by_id[c.id].outcome for c in done if c.category is category)
        lines.append(f"| {category.value} | " + " | ".join(str(row[o]) for o in Outcome) + " |")
    lines += ["", "| Caso | Esperado | Obtenido | Resultado |", "|---|---|---|---|"]
    for case in cases:
        result = by_id.get(case.id)
        predicted = result.predicted.value if result and result.predicted else "—"
        lines.append(f"| {case.id} | {case.expected.value} | {predicted} | "
                     f"{result.outcome.value if result else 'no ejecutado'} |")
    spent = sum(result.usd for result in results)
    return "\n".join([*lines, "", f"Coste calculado: {spent:.2f} USD.", ""])
