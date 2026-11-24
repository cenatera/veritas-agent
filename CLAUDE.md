# Proyecto: Veritas-Agent (Sistema Multi-Agente de Verificación Periodística)

## Objetivo
Construir un pipeline en Python que audite textos, noticias o documentos, extraiga afirmaciones verificables, las contraste en fuentes abiertas reales y genere un informe ejecutivo con citas verificables y nivel de confianza.

## Estándar Técnico y Editorial (Reglas Obligatorias)
1. **Cero alucinaciones:** Ningún agente puede inventar URLs, fechas, cifras ni fuentes. Si no hay respaldo verificable, debe etiquetarlo como `Sin evidencia suficiente`.
2. **Salidas estructuradas:** Usar `Pydantic` (`BaseModel`) para validar los datos que pasan de un agente a otro.
3. **Arquitectura modular:** Mantener separados los agentes (`src/agents/`), las herramientas externas (`src/tools/`), la orquestación (`src/orchestrator.py`) y la interfaz (`app.py`).
4. **Código limpio y documentado:** Todo el código, comentarios y docstrings deben estar claros, tipados y listos para ser revisados en un portafolio profesional.
5. **Trabajo incremental:** Construir y probar un módulo a la vez. Nunca avanzar al siguiente agente sin haber ejecutado una prueba unitaria exitosa del actual.
