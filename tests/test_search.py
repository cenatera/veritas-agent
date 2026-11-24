"""Pruebas de los buscadores, sin acceso a la red."""

import httpx2
import pytest

from src.tools.search import GdeltSearch, SearchError, SearchHit, WikipediaSearch, search_sources


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


def test_gdelt_spaces_requests() -> None:
    now, waits = [100.0], []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    provider = GdeltSearch(client(httpx2.Response(200, json={})), sleep, lambda: now[0])
    provider.search("a", 1)
    now[0] += 2
    provider.search("b", 1)
    assert waits == [4.0]
