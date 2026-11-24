"""Contratos de datos que intercambian los agentes de Veritas-Agent.

Los modelos son inmutables y convierten la regla de «cero alucinaciones» en
invariantes: toda cita debe existir literalmente en un texto conservado y todo
veredicto concluyente debe apoyarse en evidencia. Comprueban coherencia interna;
que una fuente sea real depende de que solo ``src/tools`` cree
``RetrievedSource``, a partir de respuestas HTTP reales.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Verdict(str, Enum):
    """Resultado editorial de contrastar una afirmación."""

    SUPPORTED = "Confirmada"
    REFUTED = "Refutada"
    DISPUTED = "En disputa"
    INSUFFICIENT = "Sin evidencia suficiente"


class Stance(str, Enum):
    """Relación entre una evidencia y la afirmación."""

    SUPPORTS = "respalda"
    REFUTES = "contradice"


class Confidence(str, Enum):
    """Nivel de confianza de un veredicto concluyente."""

    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


class VeritasModel(BaseModel):
    """Base estricta: inmutable y sin campos desconocidos."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Claim(VeritasModel):
    """Afirmación verificable, anclada a su posición en el texto auditado."""

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1, description="Afirmación autocontenida.")
    search_query: str = Field(min_length=1, description="Palabras clave para buscar fuentes.")
    quote: str = Field(min_length=1, description="Cita literal del original.")
    start: int = Field(ge=0, description="Posición inicial de la cita.")


class RetrievedSource(VeritasModel):
    """Fuente descargada por una herramienta, con el texto obtenido."""

    id: str = Field(min_length=1)
    url: HttpUrl
    title: Optional[str] = None
    published_on: Optional[date] = Field(
        default=None, description="``None`` si la fuente no declara la fecha."
    )
    retrieved_at: AwareDatetime
    content: str = Field(min_length=1)


class Evidence(VeritasModel):
    """Extracto literal de una fuente; la URL se resuelve por ``source_id``."""

    source_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, pattern=r"\S")
    stance: Stance


class ClaimAssessment(VeritasModel):
    """Veredicto sobre una afirmación y la evidencia que lo sostiene."""

    claim_id: str = Field(min_length=1)
    verdict: Verdict
    confidence: Optional[Confidence] = None
    evidence: tuple[Evidence, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _verdict_matches_evidence(self) -> "ClaimAssessment":
        """Exige que veredicto, evidencia y confianza sean coherentes."""
        stances = {item.stance for item in self.evidence}
        required = {
            Verdict.SUPPORTED: {Stance.SUPPORTS},
            Verdict.REFUTED: {Stance.REFUTES},
            Verdict.DISPUTED: {Stance.SUPPORTS, Stance.REFUTES},
        }
        if self.verdict is Verdict.INSUFFICIENT:
            if self.confidence is not None:
                raise ValueError("'Sin evidencia suficiente' no admite confianza.")
            return self
        if self.confidence is None:
            raise ValueError("Un veredicto concluyente requiere confianza.")
        if required[self.verdict] != stances:
            raise ValueError("La evidencia no sostiene el veredicto.")
        return self


class AuditReport(VeritasModel):
    """Informe completo; valida la trazabilidad de extremo a extremo."""

    document: str = Field(min_length=1)
    claims: tuple[Claim, ...]
    sources: tuple[RetrievedSource, ...]
    assessments: tuple[ClaimAssessment, ...]

    @model_validator(mode="after")
    def _is_traceable(self) -> "AuditReport":
        """Comprueba identificadores, citas literales y cobertura."""
        claim_ids = [claim.id for claim in self.claims]
        assessed_ids = [item.claim_id for item in self.assessments]
        sources = {source.id: source for source in self.sources}

        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("Hay identificadores de afirmación duplicados.")
        if len(sources) != len(self.sources):
            raise ValueError("Hay identificadores de fuente duplicados.")
        if sorted(claim_ids) != sorted(assessed_ids):
            raise ValueError("Cada afirmación requiere exactamente un veredicto.")

        for claim in self.claims:
            if not self.document.startswith(claim.quote, claim.start):
                raise ValueError(f"La cita de {claim.id} no está en el documento.")
        for assessment in self.assessments:
            for item in assessment.evidence:
                source = sources.get(item.source_id)
                if source is None:
                    raise ValueError(f"Fuente inexistente: {item.source_id}.")
                if item.excerpt not in source.content:
                    raise ValueError(f"Extracto ausente en la fuente {source.id}.")
        return self
