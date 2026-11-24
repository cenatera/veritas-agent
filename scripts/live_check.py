"""Prueba real del pipeline con resultados conocidos. Consume saldo de la API.

Uso: ``.venv/bin/python -m scripts.live_check``
"""

import json
from pathlib import Path

import anthropic
import httpx2
from dotenv import load_dotenv

from src.agents.extractor import ClaimExtractor
from src.agents.verifier import ClaimVerifier
from src.orchestrator import AuditLimits, VeritasPipeline
from src.report import render_markdown
from src.tools.fetcher import PageFetcher
from src.tools.search import GdeltSearch, WikipediaSearch

ROOT = Path(__file__).resolve().parent.parent
# Una afirmación cierta, una falsa, una no comprobable y una opinión.
DOCUMENT = (
    "La Torre Eiffel se inauguró en 1889. El río Amazonas desemboca en el océano "
    "Pacífico. El panadero Aurelio Quintanilla Bermejo vendió 412 panes el "
    "3 de marzo de 2019. París es la ciudad más bonita del mundo."
)
EXPECTED = {
    "1889": "Confirmada",
    "Amazonas": "Refutada",
    "Quintanilla": "Sin evidencia suficiente",
}


def main() -> None:
    """Ejecuta la auditoría e imprime resultados, aciertos y tokens."""
    load_dotenv(ROOT / ".env")
    usage: list[dict[str, int]] = []

    def record(response: httpx2.Response) -> None:
        response.read()
        if response.status_code == 200:
            usage.append(json.loads(response.content)["usage"])

    client = anthropic.Anthropic(
        http_client=httpx2.Client(timeout=600, event_hooks={"response": [record]})
    )
    result = VeritasPipeline(
        ClaimExtractor(client),
        ClaimVerifier(client),
        PageFetcher(),
        [WikipediaSearch(), GdeltSearch()],
        AuditLimits(max_claims=4, max_sources=2, max_source_chars=150_000),
    ).run(DOCUMENT)

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "live_check.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (output / "live_check.md").write_text(render_markdown(result), encoding="utf-8")

    urls = {source.id: source.url for source in result.report.sources}
    found: dict[str, str] = {}
    for claim, item, trace in zip(
        result.report.claims, result.report.assessments, result.traces, strict=True
    ):
        print(f"\n[{claim.id}] {claim.statement}\n  búsqueda: {claim.search_query}")
        print(f"  {item.verdict.value} / {item.confidence and item.confidence.value} / "
              f"{trace.status.value}\n  {item.rationale}")
        for evidence in item.evidence:
            print(f"  - {evidence.stance.value}: «{evidence.excerpt}» {urls[evidence.source_id]}")
        for note in (*trace.issues, *(r.reason for r in trace.rejected_evidence)):
            print(f"  ! {note}")
        for key in EXPECTED:
            if key in claim.statement:
                found[key] = item.verdict.value
    for rejected in result.rejected_claims:
        print(f"\nDescartada: {rejected.draft.statement} ({rejected.reason})")

    print("\nEsperado / obtenido:")
    for key, verdict in EXPECTED.items():
        print(f"  {key}: {verdict} / {found.get(key, 'no extraída')}")
    print(f"Afirmaciones extraídas: {len(result.report.claims)} (esperadas {len(EXPECTED)})")
    tokens_in = sum(u["input_tokens"] for u in usage)
    tokens_out = sum(u["output_tokens"] for u in usage)
    print(f"Llamadas: {len(usage)}; tokens: {tokens_in} entrada, {tokens_out} salida")


if __name__ == "__main__":
    main()
