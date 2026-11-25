"""Pruebas de la interfaz con el pipeline simulado."""

from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import pytest
from streamlit.testing.v1 import AppTest

from src import orchestrator
from src.agents.llm import ModelOutputError
from src.orchestrator import AuditLimits, AuditResult, VeritasPipeline
from tests.test_orchestrator import CLAIMS, DOCUMENT, FakeExtractor, FakeFetcher, FakeProvider, FakeVerifier

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class Broken:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def run(self, document: str, on_progress: Callable[[int, int], None]) -> AuditResult:
        raise self.error


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, AuditLimits]]:
    """Sustituye el pipeline real y registra sus construcciones."""
    seen: list[tuple[str, AuditLimits]] = []

    def build(language: str, limits: AuditLimits) -> VeritasPipeline:
        seen.append((language, limits))
        return VeritasPipeline(FakeExtractor(CLAIMS), FakeVerifier(), FakeFetcher(),
                               [FakeProvider("a", ["1"])], limits)

    monkeypatch.setattr(orchestrator, "build_pipeline", build)
    return seen


def start() -> AppTest:
    """Interfaz recién abierta."""
    return AppTest.from_file(APP, default_timeout=20).run()


def submit(app: AppTest, text: str = DOCUMENT) -> AppTest:
    """Escribe, acepta y verifica."""
    app.text_area(key="document").set_value(text)
    app.checkbox(key="accepted").check().run()
    return app.button[0].click().run()


def test_nothing_runs_without_consent(calls: list[tuple[str, AuditLimits]]) -> None:
    app = start()
    assert not app.exception and app.button[0].disabled
    app.text_area(key="document").set_value(DOCUMENT).run()
    assert app.button[0].disabled
    app.checkbox(key="accepted").check().run()
    assert not app.button[0].disabled
    app.text_area(key="document").set_value("  ").run()
    assert app.button[0].disabled and calls == []


def test_shows_report_and_does_not_rerun(calls: list[tuple[str, AuditLimits]]) -> None:
    app = submit(start())
    assert not app.exception
    assert "Informe de verificación" in [h.value for h in app.header]
    assert sum(m.value.startswith("#### ") and m.value[5].isdigit() for m in app.markdown) == 3
    assert calls[0][0] == "es" and calls[0][1].max_claims == 3
    app.run()
    assert len(calls) == 1


def test_warns_when_text_changes(calls: list[tuple[str, AuditLimits]]) -> None:
    app = submit(start())
    assert not app.warning
    app.text_area(key="document").set_value("Otro texto").run()
    assert "versión anterior" in app.warning[0].value and len(calls) == 1


REQUEST = httpx2.Request("POST", "https://api.test")


@pytest.mark.parametrize(("error", "message"), [
    (anthropic.AuthenticationError("x", response=httpx2.Response(401, request=REQUEST), body=None), "clave"),
    (anthropic.RateLimitError("x", response=httpx2.Response(429, request=REQUEST), body=None), "límite"),
    (anthropic.InternalServerError("x", response=httpx2.Response(500, request=REQUEST), body=None), "500"),
    (anthropic.APIConnectionError(request=REQUEST), "conectar"),
    (anthropic.AnthropicError("x"), "credenciales"),
    (ModelOutputError("Respuesta incompleta."), "extraer"),
])
def test_errors_are_explained(
    monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
) -> None:
    monkeypatch.setattr(orchestrator, "build_pipeline", lambda language, limits: Broken(error))
    app = submit(start())
    assert not app.exception and message in app.error[0].value
    assert "Informe de verificación" not in [h.value for h in app.header]


def test_privacy_cost_and_consent_are_always_visible() -> None:
    app = start()
    assert "se envía a Anthropic" in app.info[0].value and "GDELT" in app.info[0].value
    assert app.metric[0].label == "Estimación alta del coste" and "USD" in app.metric[0].value
    assert app.checkbox(key="accepted").value is False


def use_pipeline(
    monkeypatch: pytest.MonkeyPatch, claims: tuple, providers: list[FakeProvider]
) -> None:
    """Sustituye el pipeline por uno con las afirmaciones y buscadores dados."""
    monkeypatch.setattr(orchestrator, "build_pipeline", lambda language, limits: VeritasPipeline(
        FakeExtractor(claims), FakeVerifier(), FakeFetcher(), providers, limits))


def test_question_shows_reformulation_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    use_pipeline(monkeypatch, (), [FakeProvider("a", ["1"])])
    app = submit(start(), "¿Cuánto mide el puente?")
    assert any("es una pregunta" in m.value for m in app.markdown)
    assert not app.exception


def test_source_outage_shows_a_clear_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    use_pipeline(monkeypatch, CLAIMS, [FakeProvider("gdelt", None)])
    app = submit(start())
    assert "limitaron temporalmente" in app.warning[0].value
    assert "no equivale a que la afirmación sea falsa" in app.warning[0].value


def test_claim_limit_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    use_pipeline(monkeypatch, CLAIMS, [FakeProvider("a", ["1"])])
    app = start()
    app.slider[0].set_value(1)
    app = submit(app)
    assert "a 3 en «Alcance»" in app.warning[0].value
