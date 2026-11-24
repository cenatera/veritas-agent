"""Llamada a Claude con salida estructurada, común a todos los agentes."""

from collections.abc import Iterable
from typing import TypeVar

from anthropic import Anthropic
from anthropic.types.beta import BetaContentBlockParam
from pydantic import BaseModel, ValidationError

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

OutputT = TypeVar("OutputT", bound=BaseModel)


class ModelOutputError(RuntimeError):
    """El modelo no produjo una respuesta completa y utilizable."""


def request_structured(
    client: Anthropic,
    *,
    model: str,
    system: str,
    content: str | Iterable[BetaContentBlockParam],
    output: type[OutputT],
) -> OutputT:
    """Solicita una respuesta que cumpla el esquema ``output``.

    Raises:
        ModelOutputError: Rechazo, truncamiento o salida inválida.
        anthropic.APIError: La API no pudo atender la petición.
    """
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            output_format=output,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
    except ValidationError as error:
        # El SDK valida el JSON antes de exponer ``stop_reason``.
        raise ModelOutputError("Respuesta truncada o inválida.") from error

    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise ModelOutputError(f"Respuesta incompleta: {response.stop_reason}.")
    return response.parsed_output
