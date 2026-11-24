"""Comprueba que cada referencia sigue en línea y contiene su extracto. Sin coste.

Uso: ``.venv/bin/python -m evals.check_references``
"""

import sys

from evals.metrics import load_cases
from src.tools.fetcher import FetchError, PageFetcher


def main() -> int:
    """Devuelve 1 si alguna referencia no puede comprobarse."""
    fetcher = PageFetcher()
    pages: dict[str, str] = {}
    failures = checked = 0
    for case in load_cases():
        if case.reference is None:
            continue
        url = str(case.reference.url)
        problem = None
        try:
            if url not in pages:
                pages[url] = fetcher.fetch(url).content
            if case.reference.excerpt not in pages[url]:
                problem = "extracto ausente"
        except FetchError as error:
            problem = str(error)
        checked += 1
        failures += problem is not None
        print(f"{case.id} {'FALLO: ' + problem if problem else 'ok'} {url}")
    print(f"{checked - failures} de {checked} referencias comprobadas.")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
