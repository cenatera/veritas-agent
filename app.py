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
from src.guidance import MAX_SOURCE_CHARS, MAX_SOURCES
from src.orchestrator import MAX_CLAIMS, AuditLimits, AuditResult
from src.report import ClaimView, ReportView, build_view, view_to_markdown
from src.schemas import Verdict

MAX_DOCUMENT_CHARS = 30_000
LANGUAGES = {"Español": "es", "English": "en", "Français": "fr", "Português": "pt"}
BADGES = {
    Verdict.SUPPORTED: (":material/check_circle:", "green"),
    Verdict.REFUTED: (":material/cancel:", "red"),
    Verdict.DISPUTED: (":material/balance:", "orange"),
    Verdict.INSUFFICIENT: (":material/help:", "gray"),
}
METRIC_LABELS = {
    Verdict.SUPPORTED: "Confirmadas",
    Verdict.REFUTED: "Refutadas",
    Verdict.DISPUTED: "En disputa",
    Verdict.INSUFFICIENT: "Sin evidencia",
}
# Hoja de estilo fija: nunca incluye texto del usuario ni de las fuentes.
STYLE = """
<style>
.stMainBlockContainer { max-width: 56rem; padding-top: 3.5rem; padding-bottom: 5rem; }
.stMainBlockContainer [data-testid="stMarkdownContainer"] :is(p, li) { line-height: 1.65; }
.stMainBlockContainer [data-testid="stMarkdownContainer"] li { margin-bottom: 0.4rem; }
.stMainBlockContainer [data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
h1, h2, h3, h4 { letter-spacing: -0.01em; text-wrap: balance; }
[data-testid="stCaptionContainer"] { opacity: 1; }
[data-testid="stCaptionContainer"] p { color: inherit; opacity: 0.78; font-size: 0.9rem; }
[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p { color: CanvasText; }
.stTextArea textarea { line-height: 1.6; }
.st-key-summary-table { overflow-x: auto; }
.st-key-summary-table table { width: 100%; min-width: 36rem; font-size: 0.925rem; }
.st-key-summary-table :is(th, td) { padding: 0.6rem 0.75rem; vertical-align: top;
  border-left: 0; border-right: 0; overflow-wrap: normal; }
.st-key-summary-table th { font-weight: 600; white-space: nowrap; }
.st-key-summary-table td:nth-child(3) { white-space: nowrap; }
</style>
"""


def run_audit(document: str, language: str, limits: AuditLimits) -> AuditResult | None:
    """Ejecuta la auditoría y muestra el progreso o un error comprensible."""
    bar = st.progress(0.0, "Extrayendo afirmaciones…")

    def update(done: int, total: int) -> None:
        bar.progress(
            done / total if total else 1.0,
            f"Afirmaciones procesadas: {done} de {total}. Si una fuente limita las peticiones, "
            "se esperará antes de reintentar.",
        )

    try:
        return orchestrator.build_pipeline(language, limits).run(document, update)
    except anthropic.AuthenticationError:
        st.error("La API rechazó la clave. Revise `ANTHROPIC_API_KEY` en el archivo `.env`.")
    except anthropic.PermissionDeniedError:
        st.error("La clave no tiene permiso para usar el modelo.")
    except anthropic.RateLimitError:
        st.error("Se alcanzó el límite de uso de la API de Anthropic. Espere unos minutos y "
                 "vuelva a verificar.")
    except anthropic.APIConnectionError:
        st.error("No se pudo conectar con la API de Anthropic. Compruebe la conexión a internet "
                 "y vuelva a verificar.")
    except anthropic.APIStatusError as error:
        st.error(f"La API de Anthropic respondió con el error {error.status_code}. Vuelva a "
                 "verificar más tarde.")
    except anthropic.AnthropicError:
        st.error("No hay credenciales. Defina `ANTHROPIC_API_KEY` en el archivo `.env`.")
    except ModelOutputError as error:
        st.error(f"El modelo no pudo extraer afirmaciones. {error} Pruebe con un texto más breve.")
    finally:
        bar.empty()
    return None


def show_claim(claim: ClaimView) -> None:
    """Ficha de una afirmación."""
    icon, color = BADGES[claim.verdict]
    with st.container(border=True, key=f"claim-{claim.number}"):
        st.badge(claim.verdict.value, icon=icon, color=color)
        st.markdown(f"#### {claim.number}. {claim.title}")
        st.caption(f"Confianza: {claim.confidence} · Estado: {claim.status}")
        st.markdown(claim.body)


def show_report(result: AuditResult, view: ReportView) -> None:
    """Informe completo y descargas."""
    st.divider()
    st.header(view.title, anchor=False)
    st.caption(f"Generado: {view.generated}")
    st.markdown(view.lead)

    if view.claims:
        for column, (verdict, count) in zip(st.columns(4), view.counts.items(), strict=True):
            column.metric(METRIC_LABELS[verdict], count, border=True)
    if view.guidance:
        with st.container(border=True, key="no-claims"):
            st.markdown(view.guidance)
    for notice in view.notices:
        st.warning(notice, icon=":material/warning:")

    if view.claims:
        st.subheader("De un vistazo", anchor=False)
        with st.container(key="summary-table"):
            st.markdown(view.table)
        st.subheader("Detalle por afirmación", anchor=False)
        for claim in view.claims:
            show_claim(claim)

    with st.expander("Fuentes no citadas, método y límites"):
        st.markdown(view.appendix)

    left, right = st.columns(2)
    left.download_button("Descargar informe", view_to_markdown(view), "informe.md",
                         "text/markdown", icon=":material/download:", width="stretch")
    right.download_button("Descargar registro completo", result.model_dump_json(indent=2),
                          "auditoria.json", "application/json", icon=":material/data_object:",
                          width="stretch",
                          help="Incluye propuestas descartadas sin verificar: no publicar.")


def choose_scope() -> tuple[str, AuditLimits]:
    """Panel lateral con idioma y límites."""
    with st.sidebar:
        st.header("Alcance", anchor=False)
        st.caption("Más alcance implica mayor cobertura y mayor coste.")
        language = LANGUAGES[st.selectbox("Idioma de Wikipedia", list(LANGUAGES))]
        limits = AuditLimits(
            max_claims=st.slider("Máximo de afirmaciones", 1, MAX_CLAIMS, 3),
            max_sources=st.slider("Fuentes por afirmación", 1, MAX_SOURCES, 2),
            max_source_chars=st.slider("Caracteres de fuentes por afirmación", 20_000,
                                       MAX_SOURCE_CHARS, 150_000, 10_000),
        )
    return language, limits


def ask_consent(document: str, limits: AuditLimits) -> bool:
    """Privacidad, coste y aceptación explícita."""
    with st.container(border=True, key="consent"):
        st.markdown("#### Antes de verificar")
        st.info("El texto se envía a Anthropic, y las palabras clave a Wikipedia y GDELT. "
                "No use material confidencial o embargado.", icon=":material/lock:")
        st.metric("Estimación alta del coste", f"{estimate_high_usd(document, limits):.2f} USD",
                  help="Si se agotan los límites. Lo normal es que sea bastante menor, pero no "
                       "es un tope garantizado.")
        return st.checkbox("Acepto el envío del texto y el coste de la API", key="accepted")


def main() -> None:
    """Compone la página."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    st.set_page_config(page_title="Veritas-Agent", page_icon=":material/fact_check:")
    st.html(STYLE)
    st.title("Veritas-Agent", anchor=False)
    st.caption("Verificación periodística con citas comprobables. Apoya la revisión "
               "editorial; no la sustituye.")

    language, limits = choose_scope()
    document = st.text_area(
        "Texto que desea verificar", height=220, max_chars=MAX_DOCUMENT_CHARS, key="document",
        placeholder="Pegue una noticia o escriba afirmaciones con datos concretos.",
        help="Escriba afirmaciones, no preguntas: quién, qué, dónde y cuándo.",
    )
    accepted = ask_consent(document, limits)

    if st.button("Verificar", type="primary", icon=":material/fact_check:", width="stretch",
                 disabled=not (accepted and document.strip())):
        st.session_state.pop("audit", None)
        result = run_audit(document, language, limits)
        if result:
            st.session_state["audit"] = (result, build_view(result))

    if "audit" in st.session_state:
        result, view = st.session_state["audit"]
        if result.report.document != document:
            st.warning("El informe corresponde a una versión anterior del texto.")
        show_report(result, view)


main()
