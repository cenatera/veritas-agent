"""Interfaz web de Veritas-Agent.

Uso: ``.venv/bin/streamlit run app.py``
"""

from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

from src import orchestrator
from src.agents.llm import ModelOutputError
from src.costs import estimate_high_usd
from src.orchestrator import AuditLimits, AuditResult
from src.report import render_markdown

MAX_DOCUMENT_CHARS = 30_000
LANGUAGES = {"Español": "es", "English": "en", "Français": "fr", "Português": "pt"}


def run_audit(document: str, language: str, limits: AuditLimits) -> AuditResult | None:
    """Ejecuta la auditoría y muestra el progreso o un error comprensible."""
    bar = st.progress(0.0, "Extrayendo afirmaciones…")

    def update(done: int, total: int) -> None:
        bar.progress(done / total if total else 1.0, f"Afirmaciones procesadas: {done} de {total}")

    try:
        return orchestrator.build_pipeline(language, limits).run(document, update)
    except anthropic.AuthenticationError:
        st.error("La API rechazó la clave. Revisa `ANTHROPIC_API_KEY`.")
    except anthropic.PermissionDeniedError:
        st.error("La clave no tiene permiso para usar el modelo.")
    except anthropic.RateLimitError:
        st.error("Se alcanzó el límite de uso de la API. Inténtalo más tarde.")
    except anthropic.APIConnectionError:
        st.error("No se pudo conectar con la API.")
    except anthropic.APIStatusError as error:
        st.error(f"La API respondió con el error {error.status_code}.")
    except anthropic.AnthropicError:
        st.error("No hay credenciales. Define `ANTHROPIC_API_KEY` en el archivo `.env`.")
    except ModelOutputError as error:
        st.error(f"El modelo no pudo extraer afirmaciones. {error}")
    finally:
        bar.empty()
    return None


def show_result(result: AuditResult, markdown: str) -> None:
    """Muestra el informe y las descargas."""
    st.divider()
    left, right = st.columns(2)
    left.download_button("Descargar informe", markdown, "informe.md", "text/markdown",
                         width="stretch")
    right.download_button("Descargar registro completo", result.model_dump_json(indent=2),
                          "auditoria.json", "application/json", width="stretch",
                          help="Incluye propuestas descartadas sin verificar: no publicar.")
    st.markdown(markdown)


def main() -> None:
    """Compone la página."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    st.set_page_config(page_title="Veritas-Agent", page_icon="🔎")
    st.title("Veritas-Agent")
    st.caption("Verificación periodística con citas comprobables. Apoya la revisión "
               "editorial; no la sustituye.")

    with st.sidebar:
        st.header("Alcance")
        language = LANGUAGES[st.selectbox("Idioma de Wikipedia", list(LANGUAGES))]
        limits = AuditLimits(
            max_claims=st.slider("Máximo de afirmaciones", 1, 10, 3),
            max_sources=st.slider("Fuentes por afirmación", 1, 4, 2),
            max_source_chars=st.slider(
                "Caracteres de fuentes por afirmación", 20_000, 200_000, 150_000, 10_000),
        )

    document = st.text_area("Texto que quieres verificar", height=220,
                            max_chars=MAX_DOCUMENT_CHARS, key="document")
    st.info("El texto se envía a Anthropic, y las palabras clave a Wikipedia y GDELT. "
            "No uses material confidencial o embargado.")
    st.metric("Estimación alta del coste", f"{estimate_high_usd(document, limits):.2f} USD",
              help="Si se agotan los límites. Lo normal es que sea bastante menor, pero no "
                   "es un tope garantizado.")
    accepted = st.checkbox("Acepto el envío del texto y el coste de la API", key="accepted")

    if st.button("Verificar", type="primary", disabled=not (accepted and document.strip())):
        st.session_state.pop("audit", None)
        result = run_audit(document, language, limits)
        if result:
            st.session_state["audit"] = (result, render_markdown(result))

    if "audit" in st.session_state:
        result, markdown = st.session_state["audit"]
        if result.report.document != document:
            st.warning("El informe corresponde a una versión anterior del texto.")
        show_result(result, markdown)


main()
