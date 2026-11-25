"""Pruebas del descargador, sin acceso a la red."""

from collections.abc import Callable
from datetime import date

import httpx2
import pytest

from src.tools import fetcher as module
from src.tools.fetcher import FetchError, PageFetcher

PUBLIC_IP = "93.184.216.34"
TEXT = "El puente mide diez metros y fue construido por ingenieros locales. " * 6


def page(head: str = "", text: str = TEXT) -> str:
    """Documento HTML mínimo."""
    return f"<html><head><title>Puente</title>{head}</head><body><article><p>{text}</p></article></body></html>"


def make(handler: Callable[[httpx2.Request], httpx2.Response], ip: str = PUBLIC_IP) -> PageFetcher:
    """Descargador con transporte y DNS simulados."""
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return PageFetcher(client, lambda host, port: [ip], sleep=lambda seconds: None)


def html(body: str, **headers: str) -> httpx2.Response:
    """Respuesta HTML correcta."""
    return httpx2.Response(200, content=body.encode(),
                           headers={"content-type": "text/html; charset=utf-8", **headers})


def test_builds_source_and_connects_to_checked_ip() -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return html(page('<meta property="article:published_time" content="2024-03-05T10:00:00Z">'))

    source = make(handle).fetch("https://example.org/a?b=1")
    assert str(source.url) == "https://example.org/a?b=1"
    assert source.title == "Puente"
    assert source.published_on == date(2024, 3, 5)
    assert "mide diez metros" in source.content
    assert requests[0].url.host == PUBLIC_IP
    assert requests[0].headers["host"] == "example.org"
    assert requests[0].extensions["sni_hostname"] == "example.org"


@pytest.mark.parametrize(("head", "expected"), [
    ("", None),
    ('<meta property="article:published_time" content="ayer">', None),
    ('<script type="application/ld+json">{"@graph":[{"@type":"NewsArticle","datePublished":"2020-01-02"}]}</script>', date(2020, 1, 2)),
    ('<script type="application/ld+json">[{"@type":"Article","datePublished":"2020-01-02"},{"@type":"Article","datePublished":"2021-01-02"}]</script>', None),
    ('<script type="application/ld+json">{"@type":[{}],"datePublished":"2020-01-02"}</script>', None),
    ('<script type="application/ld+json">{roto</script>', None),
])
def test_date_is_only_the_declared_one(head: str, expected: date | None) -> None:
    assert make(lambda r: html(page(head))).fetch("https://example.org").published_on == expected


def test_wikipedia_creation_date_is_not_used() -> None:
    head = '<script type="application/ld+json">{"@type":"Article","datePublished":"2002-01-02"}</script>'
    assert make(lambda r: html(page(head))).fetch("https://es.wikipedia.org/wiki/X").published_on is None


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1",
                                "::ffff:127.0.0.1", "100.64.0.1", "224.0.0.1", "0.0.0.0"])
def test_non_public_addresses_are_blocked(ip: str) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        pytest.fail("No debe conectarse")

    with pytest.raises(FetchError, match="no pública"):
        make(handle, ip).fetch("https://example.org")


@pytest.mark.parametrize("url", ["ftp://example.org", "file:///etc/passwd", "https://u:p@example.org",
                                 "https://example.org:8080", "nada", ""])
def test_unsafe_urls_are_blocked(url: str) -> None:
    with pytest.raises(FetchError):
        make(lambda r: html(page())).fetch(url)


def test_redirect_target_is_checked_again() -> None:
    client = httpx2.Client(transport=httpx2.MockTransport(
        lambda r: httpx2.Response(302, headers={"location": "http://interno.test/"})))
    ips = {"example.org": PUBLIC_IP, "interno.test": "127.0.0.1"}
    with pytest.raises(FetchError, match="no pública"):
        PageFetcher(client, lambda host, port: [ips[host]]).fetch("https://example.org")


def test_final_url_is_recorded_after_redirect() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/a":
            return httpx2.Response(301, headers={"location": "/b"})
        return html(page())

    assert str(make(handle).fetch("https://example.org/a").url) == "https://example.org/b"


def test_redirect_loop_stops() -> None:
    with pytest.raises(FetchError, match="redirecciones"):
        make(lambda r: httpx2.Response(302, headers={"location": "/"})).fetch("https://example.org")


@pytest.mark.parametrize("response", [
    httpx2.Response(404), httpx2.Response(503),
    httpx2.Response(200, content=b"x", headers={"content-type": "application/pdf"}),
    httpx2.Response(200, content=b"x"),
    httpx2.Response(200, content=b"\xff", headers={"content-type": "text/plain"}),
    httpx2.Response(200, content=b"  ", headers={"content-type": "text/plain"}),
    httpx2.Response(200, content=b"", headers={"content-type": "text/html"}),
])
def test_unusable_responses_fail(response: httpx2.Response) -> None:
    with pytest.raises(FetchError):
        make(lambda r: response).fetch("https://example.org")


def test_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_BYTES", 10)
    with pytest.raises(FetchError, match="tamaño"):
        make(lambda r: html(page())).fetch("https://example.org")


def test_network_failure_is_wrapped() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("simulado")

    with pytest.raises(FetchError, match="ConnectTimeout"):
        make(handle).fetch("https://example.org")


def test_cookies_are_not_shared_between_requests() -> None:
    cookies: list[str | None] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        cookies.append(request.headers.get("cookie"))
        return html(page(), **{"set-cookie": "a=1"})

    agent = make(handle)
    agent.fetch("https://uno.test")
    agent.fetch("https://dos.test")
    assert cookies == [None, None]


@pytest.mark.parametrize(("status", "calls"), [(429, 2), (503, 2), (404, 1), (403, 1)])
def test_only_transient_statuses_are_retried(status: int, calls: int) -> None:
    seen: list[int] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(1)
        return httpx2.Response(status)

    with pytest.raises(FetchError) as caught:
        make(handle).fetch("https://example.org")
    assert len(seen) == calls == caught.value.attempts
    assert caught.value.kind.is_transient == (calls == 2)


def test_recovers_after_one_failure() -> None:
    responses = [httpx2.Response(503), html(page())]
    assert "puente" in make(lambda r: responses.pop(0)).fetch("https://example.org").content


def test_blocked_address_is_never_retried() -> None:
    with pytest.raises(FetchError) as caught:
        make(lambda r: html(page()), "127.0.0.1").fetch("https://example.org")
    assert caught.value.attempts == 1
