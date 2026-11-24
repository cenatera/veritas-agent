"""Ejecuta la evaluación con Claude real. Consume saldo solo con ``--run``.

Uso::

    .venv/bin/python -m evals.run                       # estimación, sin gastar
    .venv/bin/python -m evals.run --run --max-usd 8     # ejecución con tope

El tope es prudente: no se inicia un caso si su estimación alta no cabe en lo que
queda. Los resultados se guardan caso a caso y se reutilizan al repetir.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import anthropic
import httpx2
from dotenv import load_dotenv

from evals.metrics import CaseResult, EvalCase, Outcome, load_cases, score, summarize
from src.agents.extractor import SYSTEM_PROMPT as EXTRACTOR_PROMPT
from src.agents.extractor import ClaimExtractor
from src.agents.llm import MODEL, ModelOutputError
from src.agents.verifier import SYSTEM_PROMPT as VERIFIER_PROMPT
from src.agents.verifier import ClaimVerifier
from src.costs import INPUT_USD, OUTPUT_USD, estimate_high_usd
from src.orchestrator import AuditLimits, AuditResult, VeritasPipeline
from src.tools.fetcher import PageFetcher
from src.tools.search import GdeltSearch, WikipediaSearch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "eval"
LIMITS = AuditLimits(max_claims=1, max_sources=2, max_source_chars=150_000)
# Media de la prueba real del 24-11-2026: 0,57 USD en tres afirmaciones.
TYPICAL_USD = 0.19

Audit = Callable[[str], AuditResult]


class Meter:
    """Acumula el coste según los tokens que declara cada respuesta."""

    def __init__(self) -> None:
        self.usd = 0.0

    def record(self, response: httpx2.Response) -> None:
        """Suma una respuesta correcta de la API."""
        response.read()
        if response.status_code == 200:
            usage = json.loads(response.content)["usage"]
            self.usd += (usage["input_tokens"] * INPUT_USD
                         + usage["output_tokens"] * OUTPUT_USD) / 1_000_000


def fingerprint(case: EvalCase) -> str:
    """Identifica caso y configuración: si cambian, el resultado no se reutiliza."""
    parts = [case.model_dump_json(), LIMITS.model_dump_json(), MODEL,
             EXTRACTOR_PROMPT, VERIFIER_PROMPT]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def load_saved(cases: Sequence[EvalCase], output: Path) -> dict[str, CaseResult]:
    """Resultados guardados y vigentes; los incompletos se repiten."""
    saved: dict[str, CaseResult] = {}
    for case in cases:
        path = output / f"{case.id}.json"
        if path.exists():
            result = CaseResult.model_validate_json(path.read_text("utf-8"))
            if result.fingerprint == fingerprint(case) and result.outcome is not Outcome.INCOMPLETE:
                saved[case.id] = result
    return saved


def run_cases(
    cases: Sequence[EvalCase], audit: Audit, meter: Meter, max_usd: float, output: Path = OUTPUT
) -> list[CaseResult]:
    """Ejecuta los casos pendientes mientras quepan en ``max_usd``."""
    output.mkdir(parents=True, exist_ok=True)
    results = load_saved(cases, output)
    for case in cases:
        if case.id in results:
            continue
        if meter.usd + estimate_high_usd(case.text, LIMITS) > max_usd:
            print(f"Tope alcanzado antes de {case.id}: {meter.usd:.2f} USD en esta ejecución.")
            break
        before = meter.usd
        mark = fingerprint(case)
        try:
            result_audit = audit(case.text)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError):
            raise
        except (ModelOutputError, anthropic.APIError) as error:
            result = CaseResult(case_id=case.id, fingerprint=mark, outcome=Outcome.INCOMPLETE,
                                usd=meter.usd - before, error=type(error).__name__)
        else:
            result = score(case, result_audit, mark, meter.usd - before)
            (output / f"{case.id}.audit.json").write_text(
                result_audit.model_dump_json(indent=2), "utf-8")
        (output / f"{case.id}.json").write_text(result.model_dump_json(indent=2), "utf-8")
        results[case.id] = result
        print(f"{case.id}: {result.outcome.value} ({meter.usd:.2f} USD en esta ejecución)")
    return list(results.values())


def main(arguments: Sequence[str] | None = None) -> int:
    """Muestra la estimación o ejecuta la evaluación."""
    parser = argparse.ArgumentParser(description="Evaluación de Veritas-Agent")
    parser.add_argument("--run", action="store_true", help="llama a la API y consume saldo")
    parser.add_argument("--max-usd", type=float, help="tope prudente de gasto")
    options = parser.parse_args(arguments)

    cases = load_cases()
    pending = [case for case in cases if case.id not in load_saved(cases, OUTPUT)]
    high = sum(estimate_high_usd(case.text, LIMITS) for case in pending)
    print(f"Casos: {len(cases)}; pendientes: {len(pending)}. Coste orientativo: "
          f"{len(pending) * TYPICAL_USD:.2f} USD; estimación alta: {high:.2f} USD.")
    if not options.run:
        print("No se ha llamado a la API. Usa --run --max-usd N para ejecutar.")
        return 0
    if options.max_usd is None or not 0 < options.max_usd < float("inf"):
        parser.error("--run exige --max-usd con un importe positivo.")

    load_dotenv(ROOT / ".env")
    meter = Meter()
    # Sin reintentos: un intento fallido podría facturarse sin quedar registrado.
    client = anthropic.Anthropic(max_retries=0, http_client=httpx2.Client(
        timeout=600, event_hooks={"response": [meter.record]}))
    pipeline = VeritasPipeline(ClaimExtractor(client), ClaimVerifier(client), PageFetcher(),
                               [WikipediaSearch(), GdeltSearch()], LIMITS)
    results = run_cases(cases, pipeline.run, meter, options.max_usd)
    summary = summarize(cases, results)
    (OUTPUT / "summary.md").write_text(summary, "utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
