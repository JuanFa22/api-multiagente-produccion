"""Agente de Investigación: busca en la base de conocimiento de mercado (búsqueda simulada local)."""
import asyncio
import json
import re
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app import config
from app.agents.common import herramientas_usadas, respuesta_final
from app.state import Contribution, OrchestratorState

PROMPT = (
    "Sos el Agente de Investigación. Tu única tarea es buscar información de MERCADO con tus "
    "herramientas y resumirla con cifras y el ID de cada fuente (por ejemplo [M-02]). "
    "No calcules ventas internas ni inventes datos: si no encontrás algo, decilo. "
    "Respondé en español, en un máximo de 6 líneas."
)


@lru_cache(maxsize=1)
def _documentos() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(config.MERCADO_JSON.read_text(encoding="utf-8")))


def _tokens(texto: str) -> set[str]:
    return {t for t in re.findall(r"[a-záéíóúñ]+", texto.lower()) if len(t) > 2}


@tool
async def buscar_mercado(consulta: str, max_resultados: int = 3) -> str:
    """Busca documentos de investigación de mercado (consumo masivo, helados, golosinas, galletitas).

    Args:
        consulta: palabras clave de lo que se quiere investigar (ej. 'helados NOA verano').
        max_resultados: cantidad máxima de documentos a devolver (1 a 5).

    Devuelve los documentos más relevantes con su ID, título y texto. Si no hay
    coincidencias devuelve un mensaje que empieza con 'SIN_RESULTADOS'.
    """

    def _buscar() -> str:
        q = _tokens(consulta)
        puntuados = []
        for doc in _documentos():
            base = _tokens(doc["titulo"] + " " + doc["texto"]) | set(doc["tags"])
            puntaje = len(q & base) + 2 * len(q & set(doc["tags"]))
            if puntaje:
                puntuados.append((puntaje, doc))
        if not puntuados:
            return "SIN_RESULTADOS: probá con otras palabras clave (producto, región, categoría)."
        puntuados.sort(key=lambda x: -x[0])
        tope = max(1, min(max_resultados, 5))
        return "\n".join(f"[{d['id']}] {d['titulo']}: {d['texto']}" for _, d in puntuados[:tope])

    return await asyncio.to_thread(_buscar)


TOOLS = [buscar_mercado]


def crear_nodo_investigador(model: BaseChatModel):
    """Devuelve el nodo del grafo. El especialista recibe SOLO su instrucción (sin todo el historial)."""
    agente = create_react_agent(model, TOOLS, prompt=PROMPT, name="investigador")

    async def investigador_node(state: OrchestratorState) -> dict[str, Any]:
        paso = state.get("step_count", 0)
        instruccion = state["instruction"]
        resultado = await agente.ainvoke({"messages": [HumanMessage(content=instruccion)]})
        salida = respuesta_final(resultado["messages"]) or "Sin resultados."
        aporte: Contribution = {
            "agent": "Investigador", "step": paso, "instruction": instruccion,
            "output": salida, "tools_used": herramientas_usadas(resultado["messages"]),
        }
        return {
            "messages": [HumanMessage(content=salida, name="Investigador")],
            "research_output": salida,
            "contributions": [aporte],
        }

    return investigador_node
