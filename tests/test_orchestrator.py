"""Pruebas del orquestador con componentes simulados."""

from collections.abc import Sequence

import anthropic
import httpx2
import pytest

from src.agents.extractor import ClaimDraft, ExtractionResult, RejectedClaim
from src.agents.llm import ModelOutputError
from src.agents.verifier import VerificationResult
from src.orchestrator import AuditLimits, AuditResult, ClaimStatus, IssueKind, VeritasPipeline, is_copy
from src.schemas import Claim, ClaimAssessment, Evidence, RetrievedSource, Stance, Verdict
from src.tools.fetcher import FetchError
from src.tools.resilience import FailureKind
from src.tools.search import SearchError, SearchHit

DOCUMENT = "Uno mide 1. Dos mide 2. Tres mide 3."
CLAIMS = tuple(
    Claim(id=f"c{i}", statement=q, quote=q, start=DOCUMENT.index(q), search_query=f"tema {i}")
    for i, q in enumerate(("Uno mide 1", "Dos mide 2", "Tres mide 3"), 1))
REJECTED = RejectedClaim(draft=ClaimDraft(statement="x", quote="y", search_query="z"), reason="No consta.")


def api_error(kind: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    """Error del SDK."""
    request = httpx2.Request("POST", "https://api.test")
    return kind("simulado", response=httpx2.Response(status, request=request), body=None)


class FakeExtractor:
    def __init__(self, claims: tuple[Claim, ...]) -> None:
        self.claims = claims

    def extract(self, document: str) -> ExtractionResult:
        return ExtractionResult(claims=self.claims, rejected=(REJECTED,))


class FakeVerifier:
    """Confirma con la primera fuente, o falla en la afirmación indicada."""

    def __init__(self, errors: dict[str, Exception] | None = None) -> None:
        self.errors = errors or {}
        self.calls: list[tuple[str, list[str]]] = []

    def verify(self, claim: Claim, sources: Sequence[RetrievedSource]) -> VerificationResult:
        self.calls.append((claim.id, [s.id for s in sources]))
        if claim.id in self.errors:
            raise self.errors[claim.id]
        item = Evidence(source_id=sources[0].id, excerpt=sources[0].content, stance=Stance.SUPPORTS)
        return VerificationResult(proposed_verdict=Verdict.SUPPORTED, assessment=ClaimAssessment(
            claim_id=claim.id, verdict=Verdict.SUPPORTED, confidence="media",
            evidence=(item,), rationale="Consta."))


class FakeFetcher:
    """Descarga de memoria; las URL con «roto» fallan."""

    def __init__(self, content: str | None = None) -> None:
        self.content = content
        self.calls: list[str] = []

    def fetch(self, url: str) -> RetrievedSource:
        self.calls.append(url)
        if "roto" in url:
            raise FetchError("Respuesta HTTP 404.")
        return RetrievedSource.model_validate({
            "id": f"s-{url}", "url": url, "retrieved_at": "2026-01-01T00:00:00Z",
            "content": self.content or f"Contenido de {url}"})


class FakeProvider:
    def __init__(self, name: str, paths: list[str] | None) -> None:
        self.name, self.paths = name, paths
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[SearchHit]:
        self.queries.append(query)
        if self.paths is None:
            raise SearchError("Respuesta HTTP 429.", FailureKind.RATE_LIMITED)
        return [SearchHit.model_validate({"url": f"https://{self.name}.test/{p}", "provider": self.name})
                for p in self.paths[:limit]]


def audit(
    providers: list[FakeProvider], *, claims: tuple[Claim, ...] = CLAIMS[:1],
    verifier: FakeVerifier | None = None, fetcher: FakeFetcher | None = None,
    document: str = DOCUMENT, **limits: int,
) -> AuditResult:
    """Ejecuta una auditoría simulada."""
    return VeritasPipeline(FakeExtractor(claims), verifier or FakeVerifier(),
                           fetcher or FakeFetcher(), providers, AuditLimits(**limits)).run(document)


def test_full_run_is_traceable() -> None:
    result = audit([FakeProvider("a", ["1"])], claims=CLAIMS)
    assert {a.verdict for a in result.report.assessments} == {Verdict.SUPPORTED}
    assert [t.status for t in result.traces] == [ClaimStatus.CHECKED] * 3
    assert result.rejected_claims == (REJECTED,)
    assert AuditResult.model_validate_json(result.model_dump_json()) == result


def test_no_claims() -> None:
    provider = FakeProvider("a", ["1"])
    result = audit([provider], claims=())
    assert result.report.assessments == () and provider.queries == []


def test_pages_and_searches_are_cached() -> None:
    fetcher, provider = FakeFetcher(), FakeProvider("a", ["1", "roto"])
    same_query = tuple(c.model_copy(update={"search_query": "tema"}) for c in CLAIMS)
    result = audit([provider], claims=same_query, fetcher=fetcher)
    assert len(fetcher.calls) == 2 and provider.queries == ["tema"]
    assert len(result.report.sources) == 1
    assert all(any("404" in i.detail for i in t.issues) for t in result.traces)


def test_claim_limit_is_reported() -> None:
    verifier = FakeVerifier()
    result = audit([FakeProvider("a", ["1"])], claims=CLAIMS, verifier=verifier, max_claims=1)
    assert [t.status for t in result.traces][1:] == [ClaimStatus.SKIPPED] * 2
    assert len(verifier.calls) == 1
    assert result.report.assessments[2].verdict is Verdict.INSUFFICIENT


def test_search_failure_differs_from_no_results() -> None:
    failed, empty = audit([FakeProvider("a", None)]), audit([FakeProvider("a", [])])
    assert failed.traces[0].status is empty.traces[0].status is ClaimStatus.NO_SOURCES
    assert "429" in failed.traces[0].issues[0].detail
    assert empty.traces[0].issues[0].kind is IssueKind.NO_RESULTS


def test_partial_failures_do_not_stop_the_audit() -> None:
    result = audit([FakeProvider("a", None), FakeProvider("b", ["roto", "1"])])
    assert result.traces[0].status is ClaimStatus.CHECKED
    assert len(result.traces[0].issues) == 2


def test_failed_pages_are_replaced_by_the_next() -> None:
    verifier = FakeVerifier()
    audit([FakeProvider("a", ["roto", "1", "2", "3"])], verifier=verifier, max_sources=2)
    assert verifier.calls[0][1] == ["s-https://a.test/1", "s-https://a.test/2"]


def test_sources_alternate_providers() -> None:
    verifier = FakeVerifier()
    audit([FakeProvider("a", ["1", "2"]), FakeProvider("b", ["1"])], verifier=verifier, max_sources=2)
    assert verifier.calls[0][1] == ["s-https://a.test/1", "s-https://b.test/1"]


def test_oversized_source_is_skipped_not_cut() -> None:
    result = audit([FakeProvider("a", ["1"])], fetcher=FakeFetcher("x" * 11), max_source_chars=10)
    assert result.traces[0].status is ClaimStatus.NO_SOURCES
    assert result.traces[0].issues[0].kind is IssueKind.TOO_LARGE


@pytest.mark.parametrize("error", [
    ModelOutputError("truncada"), api_error(anthropic.RateLimitError, 429),
    api_error(anthropic.BadRequestError, 400),
    anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.test")),
])
def test_verifier_failure_affects_only_its_claim(error: Exception) -> None:
    result = audit([FakeProvider("a", ["1"])], claims=CLAIMS, verifier=FakeVerifier({"c2": error}))
    assert [t.status for t in result.traces] == [
        ClaimStatus.CHECKED, ClaimStatus.FAILED, ClaimStatus.CHECKED]
    assert result.report.assessments[1].verdict is Verdict.INSUFFICIENT


@pytest.mark.parametrize("error", [
    api_error(anthropic.AuthenticationError, 401),
    api_error(anthropic.PermissionDeniedError, 403), api_error(anthropic.NotFoundError, 404),
])
def test_configuration_errors_stop_the_audit(error: Exception) -> None:
    with pytest.raises(type(error)):
        audit([FakeProvider("a", ["1"])], verifier=FakeVerifier({"c1": error}))


def test_copy_of_the_document_is_not_evidence() -> None:
    document = DOCUMENT + " " + " ".join(f"palabra{i}" for i in range(60))
    result = audit([FakeProvider("a", ["1"])], document=document,
                   fetcher=FakeFetcher(f"Portada. {document.upper()} Pie."))
    assert result.traces[0].status is ClaimStatus.NO_SOURCES
    assert result.traces[0].issues[0].kind is IssueKind.COPY


def test_copy_detection_ignores_short_or_different_texts() -> None:
    long = " ".join(f"palabra{i}" for i in range(60))
    assert not is_copy(DOCUMENT, DOCUMENT)
    assert not is_copy(long, " ".join(f"otra{i}" for i in range(60)))
    assert is_copy(long, long)


def test_progress_is_reported() -> None:
    steps: list[tuple[int, int]] = []
    VeritasPipeline(FakeExtractor(CLAIMS), FakeVerifier(), FakeFetcher(), [FakeProvider("a", ["1"])]
                    ).run(DOCUMENT, lambda done, total: steps.append((done, total)))
    assert steps == [(0, 3), (1, 3), (2, 3), (3, 3)]


@pytest.mark.parametrize(("name", "shown"), [
    ("gdelt", "GDELT"), ("wikipedia-es", "Wikipedia (es)"), ("otro", "otro")])
def test_providers_have_readable_names(name: str, shown: str) -> None:
    assert audit([FakeProvider(name, None)]).traces[0].issues[0].origin == f"Búsqueda en {shown}"


def test_outages_are_classified_and_limits_recorded() -> None:
    result = audit([FakeProvider("a", None), FakeProvider("b", ["roto", "1"])], max_claims=2)
    search, page = result.traces[0].issues
    assert search.kind is IssueKind.RATE_LIMITED and search.kind.is_outage
    assert page.kind is IssueKind.UNUSABLE and not page.kind.is_outage
    assert "limitó temporalmente" in search.describe().lower()
    assert result.limits is not None and result.limits.max_claims == 2
