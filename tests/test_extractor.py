"""Pruebas del extractor con el SDK real y transporte HTTP simulado."""

import json
from typing import Any

import anthropic
import httpx2
import pytest

from src.agents.extractor import ClaimDraft, ClaimExtractor, anchor_claims
from src.agents.llm import ModelOutputError

DOCUMENT = "La torre mide 324 metros. Abrió en 1889. Es muy bonita."
VALID = {"statement": "La torre abrió en 1889.", "quote": "Abrió en 1889",
         "search_query": "torre apertura"}


def extractor(
    text: str | None, stop_reason: str = "end_turn", status: int = 200
) -> tuple[ClaimExtractor, list[dict[str, Any]]]:
    """Extractor con respuesta fija y registro de los cuerpos enviados."""
    sent: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append({**json.loads(request.content), "_beta": request.headers["anthropic-beta"]})
        if status != 200:
            return httpx2.Response(status, json={"type": "error", "error": {
                "type": "authentication_error", "message": "simulado"}})
        content = [] if text is None else [{"type": "text", "text": text}]
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": "claude-opus-5-5", "content": content,
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = anthropic.Anthropic(
        api_key="test", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)))
    return ClaimExtractor(client), sent


def test_extracts_and_sends_expected_request() -> None:
    agent, sent = extractor(json.dumps({"claims": [VALID]}))
    result = agent.extract(DOCUMENT)

    assert [c.id for c in result.claims] == ["c1"]
    assert DOCUMENT[result.claims[0].start:].startswith("Abrió en 1889")
    assert not result.rejected
    body = sent[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["messages"] == [{"role": "user", "content": DOCUMENT}]
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in body["_beta"]


def test_no_claims_is_a_valid_result() -> None:
    agent, _ = extractor('{"claims": []}')
    assert agent.extract("Qué bonito día.").claims == ()


@pytest.mark.parametrize(("text", "stop_reason"), [
    (None, "refusal"),
    ('{"claims": [', "max_tokens"),
    ('{"claims": []}', "max_tokens"),
    ('{"otra": 1}', "end_turn"),
    (None, "end_turn"),
])
def test_unusable_response_fails_loudly(text: str | None, stop_reason: str) -> None:
    agent, _ = extractor(text, stop_reason)
    with pytest.raises(ModelOutputError):
        agent.extract(DOCUMENT)


def test_api_error_is_not_hidden() -> None:
    agent, _ = extractor(None, status=401)
    with pytest.raises(anthropic.AuthenticationError):
        agent.extract(DOCUMENT)


def test_empty_document_makes_no_request() -> None:
    agent, sent = extractor('{"claims": []}')
    with pytest.raises(ValueError, match="vacío"):
        agent.extract(" \n")
    assert sent == []


@pytest.mark.parametrize(("statement", "quote", "reason"), [
    ("La torre abrió en 1889.", "Se inauguró en 1889", "literalmente"),
    ("La torre abrió en 1890.", "Abrió en 1889", "Cifras ausentes"),
    ("La torre mide 32 metros.", "mide 324 metros", "Cifras ausentes"),
    ("La torre abrió.", "", "vacía"),
    (" ", "Abrió en 1889", "vacía"),
])
def test_ungrounded_draft_is_rejected(statement: str, quote: str, reason: str) -> None:
    result = anchor_claims(DOCUMENT, [ClaimDraft(statement=statement, quote=quote, search_query="torre")])
    assert result.claims == ()
    assert reason in result.rejected[0].reason


def test_empty_query_is_rejected() -> None:
    assert not anchor_claims(DOCUMENT, [ClaimDraft(**{**VALID, "search_query": " "})]).claims


def test_rejection_keeps_ids_consecutive() -> None:
    bad = ClaimDraft(statement="Falso", quote="inexistente", search_query="x")
    result = anchor_claims(DOCUMENT, [bad, ClaimDraft(**VALID)])
    assert [c.id for c in result.claims] == ["c1"]
    assert len(result.rejected) == 1


def test_repeated_quotes_use_successive_positions() -> None:
    draft = ClaimDraft(statement="Subió.", quote="Subió", search_query="x")
    result = anchor_claims("Subió. Subió.", [draft, draft, draft])
    assert [c.start for c in result.claims] == [0, 7]
    assert "duplicada" in result.rejected[0].reason


def test_one_quote_may_hold_several_claims() -> None:
    drafts = [ClaimDraft(statement=s, quote="Ana y Luis viajaron", search_query="x")
              for s in ("Ana viajó.", "Luis viajó.")]
    result = anchor_claims("Ana y Luis viajaron.", drafts)
    assert [c.start for c in result.claims] == [0, 0]
