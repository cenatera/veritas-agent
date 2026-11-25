"""Pruebas de los buscadores, sin acceso a la red."""

import httpx2
import pytest

from src.tools import search as module
from src.tools.resilience import FailureKind, Pacer
from src.tools.search import GdeltSearch, SearchError, SearchHit, WikipediaSearch, search_sources


@pytest.fixture(autouse=True)
def no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita esperas reales y estado compartido entre pruebas."""
    monkeypatch.setattr(GdeltSearch.__init__, "__defaults__", (
        None, Pacer(0), module.GDELT_RETRIES, lambda seconds: None))
    monkeypatch.setattr(WikipediaSearch.__init__, "__defaults__", (
        "es", None, module.WIKIPEDIA_RETRIES, lambda seconds: None))


def client(response: httpx2.Response, seen: list[httpx2.Request] | None = None) -> httpx2.Client:
    """Cliente con respuesta fija."""
    def handle(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return response

    return httpx2.Client(transport=httpx2.MockTransport(handle))


class Fake:
    """Buscador controlado."""

    def __init__(self, name: str, urls: list[str] | None) -> None:
        self.name, self._urls = name, urls

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if self._urls is None:
            raise SearchError("caído")
        return [SearchHit.model_validate({"url": u, "provider": self.name}) for u in self._urls]


def test_wikipedia_builds_urls_and_sends_query() -> None:
    seen: list[httpx2.Request] = []
    response = httpx2.Response(200, json={"pages": [
        {"key": "¿Qué?_a/b", "title": "¿Qué? a/b"}, {"key": "B", "title": "B"}]})
    hits = WikipediaSearch("es", client(response, seen)).search("torre & río", 1)
    assert [str(h.url) for h in hits] == ["https://es.wikipedia.org/wiki/%C2%BFQu%C3%A9%3F_a/b"]
    assert seen[0].url.params["q"] == "torre & río"


def test_wikipedia_rejects_bad_language() -> None:
    with pytest.raises(ValueError, match="idioma"):
        WikipediaSearch("evil.test/")


def test_gdelt_parses_articles() -> None:
    seen: list[httpx2.Request] = []
    response = httpx2.Response(200, json={"articles": [{"url": "https://example.org/n", "title": "N"}]})
    hits = GdeltSearch(client(response, seen)).search("torre", 5)
    assert hits[0].title == "N" and hits[0].provider == "gdelt"
    assert seen[0].url.params["maxrecords"] == "5"


def test_no_results_is_not_a_failure() -> None:
    providers = [GdeltSearch(client(httpx2.Response(200, json={}))),
                 WikipediaSearch("es", client(httpx2.Response(200, json={"pages": []})))]
    outcome = search_sources("torre", providers)
    assert outcome.hits == () and outcome.failures == ()


@pytest.mark.parametrize("response", [
    httpx2.Response(429), httpx2.Response(500),
    httpx2.Response(200, text="Please limit requests"),
    httpx2.Response(200, json={"articles": [{"url": "nada"}], "pages": [{}]}),
])
@pytest.mark.parametrize("kind", [GdeltSearch, WikipediaSearch])
def test_service_errors_are_reported(kind: type, response: httpx2.Response) -> None:
    outcome = search_sources("torre", [kind(client=client(response))])
    assert outcome.hits == () and len(outcome.failures) == 1


def test_network_error_is_reported() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("simulado")

    provider = GdeltSearch(httpx2.Client(transport=httpx2.MockTransport(handle)))
    assert "ConnectError" in search_sources("torre", [provider]).failures[0].reason


def test_merges_results_and_keeps_partial_failures() -> None:
    outcome = search_sources("torre", [
        Fake("a", ["https://example.org/1"]), Fake("roto", None),
        Fake("b", ["https://example.org/1", "https://example.org/2"])])
    assert [(str(h.url), h.provider) for h in outcome.hits] == [
        ("https://example.org/1", "a"), ("https://example.org/2", "b")]
    assert [f.provider for f in outcome.failures] == ["roto"]


@pytest.mark.parametrize(("query", "limit", "providers"), [
    (" ", 5, [Fake("a", [])]), ("x", 0, [Fake("a", [])]),
    ("x", 51, [Fake("a", [])]), ("x", 5, []),
])
def test_invalid_arguments(query: str, limit: int, providers: list[Fake]) -> None:
    with pytest.raises(ValueError):
        search_sources(query, providers, limit)


def scripted(*responses: httpx2.Response | Exception) -> tuple[httpx2.Client, list[int]]:
    """Cliente que devuelve respuestas en orden y cuenta las peticiones."""
    queue, count = list(responses), [0]

    def handle(request: httpx2.Request) -> httpx2.Response:
        count[0] += 1
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx2.Client(transport=httpx2.MockTransport(handle)), count


OK = {"articles": [{"url": "https://example.org/n", "title": "N"}]}


@pytest.mark.parametrize("failure", [
    httpx2.Response(429), httpx2.Response(503), httpx2.ReadTimeout("x"), httpx2.ConnectError("x"),
    httpx2.Response(200, text="Please limit requests to one every 5 seconds"),
])
def test_gdelt_recovers_from_transient_failure(failure: httpx2.Response | Exception) -> None:
    http, count = scripted(failure, httpx2.Response(200, json=OK))
    outcome = search_sources("torre", [GdeltSearch(http)])
    assert len(outcome.hits) == 1 and outcome.failures == () and count == [2]


def test_wikipedia_recovers_from_transient_failure() -> None:
    http, count = scripted(httpx2.Response(502), httpx2.Response(200, json={"pages": []}))
    assert WikipediaSearch("es", http).search("torre", 1) == [] and count == [2]


def test_permanent_failure_is_not_retried() -> None:
    http, count = scripted(httpx2.Response(403))
    failure = search_sources("torre", [GdeltSearch(http)]).failures[0]
    assert failure.kind is FailureKind.OTHER and failure.attempts == 1 and count == [1]


def test_gdelt_pauses_after_exhausting_retries() -> None:
    now = [0.0]
    http, count = scripted(*[httpx2.Response(429)] * 3, httpx2.Response(200, json=OK))
    provider = GdeltSearch(http, Pacer(0, clock=lambda: now[0]))
    first = search_sources("a", [provider]).failures[0]
    assert first.kind is FailureKind.RATE_LIMITED and first.attempts == 3
    second = search_sources("b", [provider]).failures[0]
    assert second.kind is FailureKind.RATE_LIMITED and second.attempts == 0 and count == [3]
    now[0] = module.GDELT_COOLDOWN_SECONDS + 1
    assert len(search_sources("c", [provider]).hits) == 1


def test_long_retry_after_is_respected() -> None:
    now = [0.0]
    http, count = scripted(httpx2.Response(429, headers={"retry-after": "600"}))
    pacer = Pacer(0, clock=lambda: now[0])
    search_sources("a", [GdeltSearch(http, pacer)])
    assert count == [1]
    now[0] = 599
    assert pacer.paused() is FailureKind.RATE_LIMITED
    now[0] = 601
    assert pacer.paused() is None


def test_every_gdelt_attempt_waits_its_turn_across_instances() -> None:
    now, waits = [100.0], []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    pacer = Pacer(10, sleep, lambda: now[0])
    GdeltSearch(client(httpx2.Response(200, json={})), pacer).search("a", 1)
    now[0] += 4
    GdeltSearch(client(httpx2.Response(200, json={})), pacer).search("b", 1)
    assert waits == [6.0]
