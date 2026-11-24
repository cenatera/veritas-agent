"""Estimación prudente del coste de una auditoría con Claude.

No es un tope: los tokens reales dependen del texto, y un modelo de respaldo o un
reintento pueden cambiar el importe.
"""

from src.agents.llm import MAX_TOKENS
from src.orchestrator import AuditLimits

# Tarifa de Claude Opus 5.5 en USD por millón de tokens (noviembre de 2026).
INPUT_USD = 4.0
OUTPUT_USD = 20.0
# Prudente: el español suele superar los tres caracteres por token.
CHARS_PER_TOKEN = 2.5
PROMPT_TOKENS = 2_000


def estimate_high_usd(document: str, limits: AuditLimits) -> float:
    """Coste si se agotan los límites y cada llamada usa toda su salida."""
    characters = len(document) + limits.max_claims * limits.max_source_chars
    calls = 1 + limits.max_claims
    input_tokens = characters / CHARS_PER_TOKEN + calls * PROMPT_TOKENS
    return (input_tokens * INPUT_USD + calls * MAX_TOKENS * OUTPUT_USD) / 1_000_000
