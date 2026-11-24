"""Búsqueda de fuentes en servicios abiertos que no requieren clave.

Un resultado solo es una pista: su contenido no cuenta como evidencia hasta que
``PageFetcher`` descarga la página. «Sin resultados» y «el servicio falló» se
distinguen siempre, para no confundir una avería con falta de evidencia.
"""

import time
from collections.abc import Callable, Sequence
from typing import Protocol
from urllib.parse import quote

import httpx2
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from src.schemas import VeritasModel
from src.tools.fetcher import USER_AGENT

TIMEOUT_SECONDS = 20.0
GDELT_INTERVAL_SECONDS = 6.0
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class SearchError(RuntimeError):
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
        raise SearchError(f"Fallo de red: {type(error).__name__}.") from error
    if response.status_code != 200:
        raise SearchError(f"Respuesta HTTP {response.status_code}.")
    return response.content


class WikipediaSearch:
    """Búsqueda en Wikipedia."""

    def __init__(self, language: str = "es", client: httpx2.Client | None = None) -> None:
        """Selecciona la edición por su código de idioma."""
        if not language.replace("-", "").isascii() or not language.replace("-", "").isalpha():
            raise ValueError("Código de idioma no válido.")
        self.name = f"wikipedia-{language}"
        self._base = f"https://{language}.wikipedia.org"
        self._client = client or _default_client()

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Busca artículos por texto."""
        body = _get(
            self._client, f"{self._base}/w/rest.php/v1/search/page",
            {"q": query, "limit": str(limit)},
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
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Permite inyectar el cliente HTTP y el reloj."""
        self._client = client or _default_client()
        self._sleep = sleep
        self._clock = clock
        self._next_request = float("-inf")

    def _wait_turn(self) -> None:
        """Espacia las peticiones: GDELT limita su frecuencia."""
        delay = self._next_request - self._clock()
        if delay > 0:
            self._sleep(delay)
        self._next_request = self._clock() + GDELT_INTERVAL_SECONDS

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Busca noticias que coincidan con la consulta."""
        self._wait_turn()
        body = _get(self._client, GDELT_URL, {
            "query": query, "mode": "artlist", "format": "json", "maxrecords": str(limit),
        })
        try:
            # Los errores de GDELT llegan como texto plano con estado 200.
            articles = _GdeltResponse.model_validate_json(body).articles
        except ValidationError as error:
            raise SearchError("Respuesta no reconocida.") from error
        return [
            SearchHit(url=item.url, title=item.title, provider=self.name)
            for item in articles[:limit]
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
            failures.append(SearchFailure(provider=provider.name, reason=str(error)))
    return SearchOutcome(hits=tuple(hits.values()), failures=tuple(failures))
