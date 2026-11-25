"""Búsqueda de fuentes en servicios abiertos que no requieren clave.

Un resultado solo es una pista: su contenido no cuenta como evidencia hasta que
``PageFetcher`` descarga la página. «Sin resultados» y «el servicio falló» se
distinguen siempre, para no confundir una avería con falta de evidencia. Los fallos
transitorios se reintentan.
"""

import time
from collections.abc import Sequence
from typing import Protocol
from urllib.parse import quote

import httpx2
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from src.schemas import VeritasModel
from src.tools.fetcher import USER_AGENT
from src.tools.resilience import (
    FailureKind,
    Pacer,
    RetryPolicy,
    Sleep,
    SourceError,
    from_exception,
    from_response,
    with_retries,
)

TIMEOUT_SECONDS = 20.0
# GDELT pide al menos 5 segundos entre peticiones; se deja margen.
GDELT_INTERVAL_SECONDS = 10.0
GDELT_COOLDOWN_SECONDS = 120.0
WIKIPEDIA_RETRIES = RetryPolicy(attempts=3, base_seconds=1.0)
GDELT_RETRIES = RetryPolicy(attempts=3, base_seconds=6.0, max_seconds=30.0)
# Compartido: cada auditoría crea su buscador, pero el límite es del equipo.
GDELT_PACER = Pacer(GDELT_INTERVAL_SECONDS)
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class SearchError(SourceError):
    """El servicio no respondió o lo hizo de forma inesperada."""


class SearchHit(VeritasModel):
    """Página candidata devuelta por un servicio."""

    url: HttpUrl
    title: str | None = None
    provider: str = Field(min_length=1)


class SearchFailure(VeritasModel):
    """Servicio que no pudo consultarse."""

    provider: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    kind: FailureKind = FailureKind.OTHER
    attempts: int = Field(default=1, ge=0)


class SearchOutcome(VeritasModel):
    """Candidatas únicas y servicios que fallaron."""

    hits: tuple[SearchHit, ...] = ()
    failures: tuple[SearchFailure, ...] = ()


class SearchProvider(Protocol):
    """Interfaz común de los buscadores."""

    name: str

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Devuelve hasta ``limit`` candidatas o lanza ``SearchError``."""
        ...


class _WikiPage(BaseModel):
    key: str = Field(min_length=1)
    title: str


class _WikiResponse(BaseModel):
    pages: list[_WikiPage]


class _GdeltArticle(BaseModel):
    url: HttpUrl
    title: str | None = None


class _GdeltResponse(BaseModel):
    # GDELT responde ``{}`` cuando no hay resultados.
    articles: list[_GdeltArticle] = []


def _default_client() -> httpx2.Client:
    """Cliente HTTP con identificación genérica."""
    return httpx2.Client(timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})


def _get(client: httpx2.Client, url: str, params: dict[str, str]) -> bytes:
    """Petición GET que traduce los fallos a ``SearchError``."""
    try:
        response = client.get(url, params=params)
    except httpx2.HTTPError as error:
        raise from_exception(SearchError, error) from error
    if response.status_code != 200:
        raise from_response(SearchError, response)
    return response.content


class WikipediaSearch:
    """Búsqueda en Wikipedia."""

    def __init__(
        self,
        language: str = "es",
        client: httpx2.Client | None = None,
        retries: RetryPolicy = WIKIPEDIA_RETRIES,
        sleep: Sleep = time.sleep,
    ) -> None:
        """Selecciona la edición por su código de idioma."""
        if not language.replace("-", "").isascii() or not language.replace("-", "").isalpha():
            raise ValueError("Código de idioma no válido.")
        self.name = f"wikipedia-{language}"
        self._base = f"https://{language}.wikipedia.org"
        self._client = client or _default_client()
        self._retries = retries
        self._sleep = sleep

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Busca artículos por texto."""
        body = with_retries(
            lambda: _get(
                self._client,
                f"{self._base}/w/rest.php/v1/search/page",
                {"q": query, "limit": str(limit)},
            ),
            self._retries,
            self._sleep,
        )
        try:
            pages = _WikiResponse.model_validate_json(body).pages
            return [
                SearchHit(
                    url=HttpUrl(f"{self._base}/wiki/{quote(page.key)}"),
                    title=page.title, provider=self.name,
                )
                for page in pages[:limit]
            ]
        except ValidationError as error:
            raise SearchError("Respuesta no reconocida.") from error


class GdeltSearch:
    """Búsqueda de noticias recientes en GDELT."""

    name = "gdelt"

    def __init__(
        self,
        client: httpx2.Client | None = None,
        pacer: Pacer = GDELT_PACER,
        retries: RetryPolicy = GDELT_RETRIES,
        sleep: Sleep = time.sleep,
    ) -> None:
        """Permite inyectar el cliente HTTP, el ritmo y los reintentos."""
        self._client = client or _default_client()
        self._pacer = pacer
        self._retries = retries
        self._sleep = sleep

    def _request(self, query: str, limit: int) -> _GdeltResponse:
        """Un intento, a su turno."""
        self._pacer.wait()
        body = _get(self._client, GDELT_URL, {
            "query": query, "mode": "artlist", "format": "json", "maxrecords": str(limit),
        })
        try:
            return _GdeltResponse.model_validate_json(body)
        except ValidationError as error:
            # GDELT comunica algunos límites en texto plano con estado 200.
            if b"limit requests" in body[:500].lower():
                raise SearchError(
                    "El servicio pidió reducir las peticiones.", FailureKind.RATE_LIMITED
                ) from error
            raise SearchError("Respuesta no reconocida.") from error

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Busca noticias que coincidan con la consulta.

        Si el servicio sigue fallando tras los reintentos, queda en pausa para no
        agravar el límite ni retrasar las demás afirmaciones.
        """
        paused = self._pacer.paused()
        if paused is not None:
            error = SearchError("Servicio en pausa por un fallo reciente.", paused)
            error.attempts = 0
            raise error
        try:
            response = with_retries(
                lambda: self._request(query, limit), self._retries, self._sleep
            )
        except SearchError as error:
            if error.kind.is_transient:
                self._pacer.pause(
                    max(GDELT_COOLDOWN_SECONDS, error.retry_after or 0.0), error.kind
                )
            raise
        return [
            SearchHit(url=item.url, title=item.title, provider=self.name)
            for item in response.articles[:limit]
        ]


def search_sources(
    query: str, providers: Sequence[SearchProvider], limit: int = 5
) -> SearchOutcome:
    """Consulta todos los servicios; un fallo parcial se registra y no detiene el resto."""
    if not query.strip():
        raise ValueError("La consulta está vacía.")
    if not 1 <= limit <= 50:
        raise ValueError("El límite debe estar entre 1 y 50.")
    if not providers:
        raise ValueError("Se requiere al menos un servicio.")

    hits: dict[HttpUrl, SearchHit] = {}
    failures: list[SearchFailure] = []
    for provider in providers:
        try:
            for hit in provider.search(query.strip(), limit):
                hits.setdefault(hit.url, hit)
        except SearchError as error:
            failures.append(SearchFailure(
                provider=provider.name,
                reason=str(error),
                kind=error.kind,
                attempts=error.attempts,
            ))
    return SearchOutcome(hits=tuple(hits.values()), failures=tuple(failures))
