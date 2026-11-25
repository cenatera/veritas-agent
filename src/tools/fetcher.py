"""Descarga páginas públicas y crea ``RetrievedSource`` con lo obtenido.

Las URL proceden de terceros: cada salto se limita a direcciones IP públicas y la
conexión usa la misma IP comprobada, para impedir accesos a la red interna.
"""

import hashlib
import ipaddress
import json
import socket
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from typing import Any

import httpx2
import trafilatura
from lxml import etree, html as lxml_html
from pydantic import ValidationError

from src.schemas import RetrievedSource
from src.tools.resilience import (
    FailureKind,
    RetryPolicy,
    Sleep,
    SourceError,
    from_exception,
    from_response,
    with_retries,
)

USER_AGENT = "VeritasAgent/0.1 (fact-checking research tool)"
MAX_BYTES = 3_000_000
MAX_REDIRECTS = 5
TIMEOUT_SECONDS = 15.0
FETCH_RETRIES = RetryPolicy(attempts=2, base_seconds=1.5)
ARTICLE_TYPES = {"Article", "NewsArticle", "ReportageNewsArticle", "BlogPosting", "Report"}

Resolver = Callable[[str, int], list[str]]


class FetchError(SourceError):
    """No se pudo obtener contenido utilizable de una URL."""


def resolve_host(host: str, port: int) -> list[str]:
    """Resuelve un nombre a sus direcciones IP."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _public_address(url: httpx2.URL, resolver: Resolver) -> str:
    """Devuelve una IP pública de la URL o rechaza la conexión."""
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
        raise FetchError("URL no permitida.")
    port = 443 if url.scheme == "https" else 80
    if url.port not in {None, port}:
        raise FetchError("Puerto no permitido.")
    try:
        addresses = resolver(url.host, port)
        parsed = [ipaddress.ip_address(address) for address in addresses]
    except (OSError, ValueError) as error:
        temporary = isinstance(error, socket.gaierror) and error.errno == socket.EAI_AGAIN
        raise FetchError(
            "No se pudo resolver el dominio.",
            FailureKind.NETWORK if temporary else FailureKind.OTHER,
        ) from error
    if not parsed or any(not ip.is_global or ip.is_multicast for ip in parsed):
        raise FetchError("El dominio apunta a una dirección no pública.")
    # IPv4 primero: muchas redes no tienen salida IPv6.
    return str(min(parsed, key=lambda ip: ip.version))


def _objects(value: Any) -> Iterator[dict[str, Any]]:
    """Recorre los objetos de nivel superior de un bloque JSON-LD."""
    if isinstance(value, list):
        for item in value:
            yield from _objects(item)
    elif isinstance(value, dict):
        yield value
        yield from _objects(value.get("@graph"))


def _declared_date(tree: Any) -> date | None:
    """Fecha de publicación declarada; ``None`` si falta o es contradictoria."""
    raw = [str(v) for v in tree.xpath('//meta[@property="article:published_time"]/@content')]
    if not raw:
        for script in tree.xpath('//script[@type="application/ld+json"]/text()'):
            try:
                items = list(_objects(json.loads(script)))
            except (ValueError, RecursionError):
                continue
            for item in items:
                kinds = item.get("@type")
                kinds = kinds if isinstance(kinds, list) else [kinds]
                if any(isinstance(k, str) and k in ARTICLE_TYPES for k in kinds):
                    raw.append(str(item.get("datePublished", "")))
    dates = set()
    for value in filter(None, raw):
        try:
            dates.add(date.fromisoformat(value.strip()[:10]))
        except ValueError:
            continue
    return dates.pop() if len(dates) == 1 else None


def _declared_title(tree: Any) -> str | None:
    """Título declarado en los metadatos o en ``<title>``."""
    found = tree.xpath('//meta[@property="og:title"]/@content') + tree.xpath("//head/title/text()")
    return next((str(title).strip() for title in found if str(title).strip()), None)


class PageFetcher:
    """Descarga páginas con límites de destino, tamaño y tiempo de espera."""

    def __init__(
        self,
        client: httpx2.Client | None = None,
        resolver: Resolver = resolve_host,
        retries: RetryPolicy = FETCH_RETRIES,
        sleep: Sleep = time.sleep,
    ) -> None:
        """Permite inyectar el cliente, la resolución DNS y los reintentos."""
        # Sin conexiones persistentes: dos dominios pueden compartir IP.
        self._client = client or httpx2.Client(
            timeout=TIMEOUT_SECONDS,
            trust_env=False,
            limits=httpx2.Limits(max_keepalive_connections=0),
        )
        self._resolver = resolver
        self._retries = retries
        self._sleep = sleep

    def fetch(self, url: str) -> RetrievedSource:
        """Descarga ``url`` y devuelve su texto principal.

        Raises:
            FetchError: URL no permitida, fallo HTTP o contenido no utilizable.
        """
        final_url, media_type, encoding, body = with_retries(
            lambda: self._attempt(url), self._retries, self._sleep
        )

        title: str | None = None
        published_on: date | None = None
        if media_type == "text/plain":
            try:
                content = body.decode(encoding or "utf-8")
            except (LookupError, UnicodeDecodeError) as error:
                raise FetchError("Codificación no reconocida.") from error
        else:
            try:
                parser = lxml_html.HTMLParser(encoding=encoding) if encoding else None
                tree = lxml_html.fromstring(body, parser=parser)
            except (LookupError, ValueError, etree.LxmlError) as error:
                raise FetchError("HTML no interpretable.") from error
            title = _declared_title(tree)
            # En Wikipedia ``datePublished`` es la creación de la página.
            host = final_url.host
            if host != "wikipedia.org" and not host.endswith(".wikipedia.org"):
                published_on = _declared_date(tree)
            content = trafilatura.extract(tree, include_comments=False) or ""

        content = content.strip()
        if not content:
            raise FetchError("La página no contiene texto utilizable.")
        digest = hashlib.sha256(f"{final_url}\n{content}".encode()).hexdigest()
        try:
            return RetrievedSource.model_validate({
                "id": f"s-{digest[:16]}",
                "url": str(final_url),
                "title": title,
                "published_on": published_on,
                "retrieved_at": datetime.now(UTC),
                "content": content,
            })
        except ValidationError as error:
            raise FetchError("La fuente no cumple el contrato de datos.") from error

    def _attempt(self, url: str) -> tuple[httpx2.URL, str, str | None, bytes]:
        """Un intento de descarga, con los fallos ya clasificados."""
        try:
            return self._download(httpx2.URL(url))
        except (httpx2.HTTPError, httpx2.InvalidURL, UnicodeError) as error:
            raise from_exception(FetchError, error) from error

    def _download(self, url: httpx2.URL) -> tuple[httpx2.URL, str, str | None, bytes]:
        """Sigue redirecciones comprobadas y lee el cuerpo con límite."""
        for _ in range(MAX_REDIRECTS + 1):
            address = _public_address(url, self._resolver)
            self._client.cookies.clear()
            request = self._client.build_request(
                "GET",
                url.copy_with(host=address),
                headers={
                    "Host": url.netloc.decode("ascii"),
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html, text/plain",
                    "Accept-Encoding": "identity",
                },
                extensions={"sni_hostname": url.raw_host.decode("ascii")},
            )
            response = self._client.send(request, stream=True, follow_redirects=False)
            try:
                if response.is_redirect:
                    url = url.join(response.headers["location"])
                    continue
                if response.status_code != 200:
                    raise from_response(FetchError, response)
                media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if media_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                    raise FetchError(f"Tipo no admitido: {media_type or 'desconocido'}.")
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise FetchError("Compresión no admitida.")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        raise FetchError("La página supera el tamaño permitido.")
                return url, media_type, response.charset_encoding, bytes(body)
            finally:
                response.close()
        raise FetchError("Demasiadas redirecciones.")
