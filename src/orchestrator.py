"""Orquestador: extracción, búsqueda, descarga y verificación.

Ningún fallo se convierte en silencio. Cada afirmación conserva su estado y las
incidencias, de modo que «no se pudo comprobar» nunca se confunde con «se buscó y
no hay evidencia suficiente».
"""

import re
from collections.abc import Callable, Sequence
from enum import Enum
from itertools import chain, zip_longest
from typing import Protocol

import anthropic
from pydantic import Field

from src.agents.extractor import ClaimExtractor, ExtractionResult, RejectedClaim
from src.agents.llm import ModelOutputError
from src.agents.verifier import (
    ClaimVerifier,
    RejectedEvidence,
    RejectedRationale,
    RejectedRelated,
    VerificationResult,
)
from src.schemas import (
    AuditReport,
    Claim,
    ClaimAssessment,
    RetrievedSource,
    VeritasModel,
    Verdict,
)
from src.tools.fetcher import FetchError, PageFetcher
from src.tools.resilience import FailureKind
from src.tools.search import (
    GdeltSearch,
    SearchHit,
    SearchOutcome,
    SearchProvider,
    WikipediaSearch,
    search_sources,
)

# Máximo que ofrece la interfaz en «Alcance».
MAX_CLAIMS = 10
SHINGLE_WORDS = 8
MIN_COPY_WORDS = 40
COPY_RATIO = 0.5


class Extractor(Protocol):
    """Extrae afirmaciones de un documento."""

    def extract(self, document: str) -> ExtractionResult: ...


class Verifier(Protocol):
    """Contrasta una afirmación con fuentes."""

    def verify(self, claim: Claim, sources: Sequence[RetrievedSource]) -> VerificationResult: ...


class Fetcher(Protocol):
    """Descarga una página."""

    def fetch(self, url: str) -> RetrievedSource: ...


class ClaimStatus(str, Enum):
    """Hasta dónde llegó la comprobación de una afirmación."""

    CHECKED = "Contrastada"
    NO_SOURCES = "Sin fuentes utilizables"
    SKIPPED = "No comprobada por límite"
    FAILED = "No comprobada por fallo"


class IssueKind(str, Enum):
    """Clase de incidencia registrada durante una auditoría."""

    RATE_LIMITED = "La fuente limitó temporalmente las peticiones"
    SERVER = "La fuente falló al responder"
    TIMEOUT = "La fuente no respondió a tiempo"
    NETWORK = "No se pudo conectar con la fuente"
    UNUSABLE = "La fuente no ofreció contenido utilizable"
    NO_RESULTS = "La búsqueda no devolvió resultados"
    COPY = "La página reproduce el documento auditado"
    TOO_LARGE = "La página supera el límite de texto"
    VERIFIER = "El modelo verificador no completó la respuesta"

    @property
    def is_outage(self) -> bool:
        """Indica un fallo temporal de una fuente, ajeno a su contenido."""
        return self in {
            IssueKind.RATE_LIMITED,
            IssueKind.SERVER,
            IssueKind.TIMEOUT,
            IssueKind.NETWORK,
        }


_ISSUE_BY_FAILURE = {
    FailureKind.RATE_LIMITED: IssueKind.RATE_LIMITED,
    FailureKind.SERVER: IssueKind.SERVER,
    FailureKind.TIMEOUT: IssueKind.TIMEOUT,
    FailureKind.NETWORK: IssueKind.NETWORK,
    FailureKind.OTHER: IssueKind.UNUSABLE,
}


class Issue(VeritasModel):
    """Incidencia con su origen y el detalle técnico."""

    kind: IssueKind
    origin: str = Field(min_length=1)
    detail: str = ""
    attempts: int = Field(default=1, ge=0, description="Cero si no llegó a consultarse.")

    def describe(self) -> str:
        """Descripción en una línea."""
        parts = [f"{self.origin}: {self.kind.value.lower()}."]
        if self.detail:
            parts.append(self.detail)
        if self.attempts > 1:
            parts.append(f"Intentos: {self.attempts}.")
        return " ".join(parts)


class ClaimTrace(VeritasModel):
    """Recorrido de una afirmación, para auditoría editorial."""

    claim_id: str = Field(min_length=1)
    status: ClaimStatus
    source_ids: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()
    proposed_verdict: Verdict | None = None
    rejected_evidence: tuple[RejectedEvidence, ...] = ()
    rejected_related: tuple[RejectedRelated, ...] = ()
    rejected_rationale: RejectedRationale | None = None


class AuditLimits(VeritasModel):
    """Límites que acotan el coste de una auditoría."""

    max_claims: int = Field(default=10, ge=1)
    max_sources: int = Field(default=4, ge=1)
    max_source_chars: int = Field(
        default=200_000, ge=1, description="Total por afirmación; nunca se recorta una fuente."
    )
    hits_per_provider: int = Field(default=5, ge=1, le=50)


class AuditResult(VeritasModel):
    """Informe validado y registro de todo lo descartado o fallido."""

    report: AuditReport
    traces: tuple[ClaimTrace, ...]
    rejected_claims: tuple[RejectedClaim, ...] = ()
    limits: AuditLimits | None = None


def _shingles(text: str) -> set[tuple[str, ...]]:
    """Secuencias de palabras consecutivas, sin mayúsculas ni puntuación."""
    words = re.findall(r"\w+", text.lower())
    return {
        tuple(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)
    }


def is_copy(document: str, content: str) -> bool:
    """Indica si una fuente reproduce gran parte del documento auditado.

    Los textos breves no se evalúan: una coincidencia no indica copia.
    """
    if len(re.findall(r"\w+", document)) < MIN_COPY_WORDS:
        return False
    original = _shingles(document)
    return len(original & _shingles(content)) / len(original) >= COPY_RATIO


def _provider_name(name: str) -> str:
    """Nombre legible de un buscador."""
    if name.startswith("wikipedia-"):
        return f"Wikipedia ({name.removeprefix('wikipedia-')})"
    return "GDELT" if name == GdeltSearch.name else name


def _interleave(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Alterna los buscadores para diversificar las fuentes."""
    groups: dict[str, list[SearchHit]] = {}
    for hit in hits:
        groups.setdefault(hit.provider, []).append(hit)
    return [hit for hit in chain.from_iterable(zip_longest(*groups.values())) if hit is not None]


class _Run:
    """Estado de una auditoría: documento, cachés y fuentes obtenidas."""

    def __init__(
        self,
        document: str,
        fetcher: Fetcher,
        providers: Sequence[SearchProvider],
        limits: AuditLimits,
    ) -> None:
        self.document = document
        self.sources: dict[str, RetrievedSource] = {}
        self._fetcher = fetcher
        self._providers = providers
        self._limits = limits
        self._searches: dict[str, SearchOutcome] = {}
        self._pages: dict[str, RetrievedSource | Issue] = {}

    def _search(self, query: str) -> SearchOutcome:
        if query not in self._searches:
            self._searches[query] = search_sources(
                query, self._providers, self._limits.hits_per_provider
            )
        return self._searches[query]

    def _page(self, url: str) -> RetrievedSource | Issue:
        """Devuelve la fuente o la incidencia por el que no puede usarse."""
        if url not in self._pages:
            try:
                page = self._fetcher.fetch(url)
            except FetchError as error:
                self._pages[url] = Issue(
                    kind=_ISSUE_BY_FAILURE[error.kind],
                    origin=url,
                    detail=str(error),
                    attempts=error.attempts,
                )
            else:
                copied = is_copy(self.document, page.content)
                self._pages[url] = Issue(kind=IssueKind.COPY, origin=url) if copied else page
        return self._pages[url]

    def collect(self, claim: Claim) -> tuple[list[RetrievedSource], list[Issue]]:
        """Reúne fuentes para una afirmación y anota las incidencias."""
        outcome = self._search(claim.search_query)
        issues = [
            Issue(
                kind=_ISSUE_BY_FAILURE[failure.kind],
                origin=f"Búsqueda en {_provider_name(failure.provider)}",
                detail=failure.reason,
                attempts=failure.attempts,
            )
            for failure in outcome.failures
        ]
        if not outcome.hits and not outcome.failures:
            issues.append(Issue(kind=IssueKind.NO_RESULTS, origin="Búsqueda"))
        selected: dict[str, RetrievedSource] = {}
        size = 0
        for hit in _interleave(outcome.hits):
            if len(selected) == self._limits.max_sources:
                break
            page = self._page(str(hit.url))
            if isinstance(page, Issue):
                issues.append(page)
            elif page.id in selected:
                continue
            elif size + len(page.content) > self._limits.max_source_chars:
                issues.append(Issue(kind=IssueKind.TOO_LARGE, origin=str(hit.url)))
            else:
                selected[page.id] = page
                size += len(page.content)
        self.sources.update(selected)
        return list(selected.values()), issues


class VeritasPipeline:
    """Ejecuta una auditoría completa."""

    def __init__(
        self,
        extractor: Extractor,
        verifier: Verifier,
        fetcher: Fetcher,
        providers: Sequence[SearchProvider],
        limits: AuditLimits | None = None,
    ) -> None:
        """Recibe los componentes ya construidos."""
        self._extractor = extractor
        self._verifier = verifier
        self._fetcher = fetcher
        self._providers = providers
        self._limits = limits or AuditLimits()

    def run(
        self, document: str, on_progress: Callable[[int, int], None] | None = None
    ) -> AuditResult:
        """Audita ``document``.

        ``on_progress`` recibe las afirmaciones procesadas y el total.

        Raises:
            ValueError: El documento está vacío.
            ModelOutputError: La extracción no produjo una salida utilizable.
            anthropic.APIError: Falló la extracción, o el acceso a la API en general.
        """
        extraction = self._extractor.extract(document)
        run = _Run(document, self._fetcher, self._providers, self._limits)
        assessments: list[ClaimAssessment] = []
        traces: list[ClaimTrace] = []
        total = len(extraction.claims)
        if on_progress:
            on_progress(0, total)
        for index, claim in enumerate(extraction.claims):
            if index >= self._limits.max_claims:
                assessment, trace = self._unchecked(
                    claim, ClaimStatus.SKIPPED, "Se alcanzó el límite de afirmaciones."
                )
            else:
                assessment, trace = self._check(claim, run)
            assessments.append(assessment)
            traces.append(trace)
            if on_progress:
                on_progress(index + 1, total)

        return AuditResult(
            report=AuditReport(
                document=document,
                claims=extraction.claims,
                sources=tuple(run.sources.values()),
                assessments=tuple(assessments),
            ),
            traces=tuple(traces),
            rejected_claims=extraction.rejected,
            limits=self._limits,
        )

    def _check(self, claim: Claim, run: _Run) -> tuple[ClaimAssessment, ClaimTrace]:
        """Busca fuentes y verifica una afirmación."""
        sources, issues = run.collect(claim)
        source_ids = tuple(source.id for source in sources)
        if not sources:
            return self._unchecked(
                claim, ClaimStatus.NO_SOURCES, "No se obtuvo ninguna fuente utilizable.", issues
            )
        try:
            result = self._verifier.verify(claim, sources)
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
        ):
            # Fallos de configuración: afectarían a todas las afirmaciones.
            raise
        except (ModelOutputError, anthropic.APIError) as error:
            detail = str(error) if isinstance(error, ModelOutputError) else type(error).__name__
            return self._unchecked(
                claim,
                ClaimStatus.FAILED,
                "La verificación no pudo completarse.",
                [*issues, Issue(kind=IssueKind.VERIFIER, origin="Verificador", detail=detail)],
                source_ids,
            )
        return result.assessment, ClaimTrace(
            claim_id=claim.id,
            status=ClaimStatus.CHECKED,
            source_ids=source_ids,
            issues=tuple(issues),
            proposed_verdict=result.proposed_verdict,
            rejected_evidence=result.rejected,
            rejected_related=result.rejected_related,
            rejected_rationale=result.rejected_rationale,
        )

    @staticmethod
    def _unchecked(
        claim: Claim,
        status: ClaimStatus,
        rationale: str,
        issues: Sequence[Issue] = (),
        source_ids: tuple[str, ...] = (),
    ) -> tuple[ClaimAssessment, ClaimTrace]:
        """Resultado de una afirmación que no llegó a contrastarse."""
        return (
            ClaimAssessment(
                claim_id=claim.id, verdict=Verdict.INSUFFICIENT, rationale=rationale
            ),
            ClaimTrace(
                claim_id=claim.id, status=status, issues=tuple(issues), source_ids=source_ids
            ),
        )


def build_pipeline(language: str = "es", limits: AuditLimits | None = None) -> VeritasPipeline:
    """Crea el pipeline con los componentes reales."""
    client = anthropic.Anthropic()
    return VeritasPipeline(
        ClaimExtractor(client),
        ClaimVerifier(client),
        PageFetcher(),
        [WikipediaSearch(language), GdeltSearch()],
        limits,
    )
