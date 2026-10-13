"""Grafo jerárquico del Módulo 6 + control de riesgo y Human-in-the-loop del Módulo 7.

START -> supervisor <-> (Investigador | Analista) -> sintesis -> evaluar_riesgo
      -> [si es crítica] aprobacion_humana (PAUSA) -> ejecutar_accion -> END
"""
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app import config
from app.agents.analysis_agent import crear_nodo_analista
from app.agents.research_agent import crear_nodo_investigador
from app.agents.supervisor import crear_nodo_sintesis, crear_nodo_supervisor
from app.hitl import (
    ActionExecutor, aprobacion_humana_node, crear_nodo_ejecutar_accion, ejecutor_en_memoria,
    evaluar_riesgo_node, ruta_tras_aprobacion, ruta_tras_riesgo,
)
from app.state import OrchestratorState


def crear_modelo() -> BaseChatModel:
    """Gemini (import diferido para que los tests no necesiten clave)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GOOGLE_API_KEY:
        raise RuntimeError("Falta GOOGLE_API_KEY (ver .env.example).")
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL, google_api_key=config.GOOGLE_API_KEY,
        temperature=0, max_retries=config.LLM_MAX_RETRIES,
    )


def construir_grafo(
    supervisor_model: BaseChatModel,
    worker_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    action_executor: ActionExecutor | None = None,
) -> CompiledStateGraph:
    worker_model = worker_model or supervisor_model
    builder = StateGraph(OrchestratorState)

    builder.add_node("supervisor", crear_nodo_supervisor(supervisor_model))
    builder.add_node("Investigador", crear_nodo_investigador(worker_model))
    builder.add_node("Analista", crear_nodo_analista(worker_model))
    builder.add_node("sintesis", crear_nodo_sintesis(supervisor_model))
    builder.add_node("evaluar_riesgo", evaluar_riesgo_node)
    builder.add_node("aprobacion_humana", aprobacion_humana_node)
    builder.add_node("ejecutar_accion", crear_nodo_ejecutar_accion(action_executor or ejecutor_en_memoria))

    builder.add_edge(START, "supervisor")
    builder.add_edge("Investigador", "supervisor")
    builder.add_edge("Analista", "supervisor")
    builder.add_conditional_edges(
        "supervisor", lambda s: s["next_agent"],
        {"Investigador": "Investigador", "Analista": "Analista", "FINALIZAR": "sintesis"},
    )
    builder.add_edge("sintesis", "evaluar_riesgo")
    builder.add_conditional_edges("evaluar_riesgo", ruta_tras_riesgo, {"aprobacion_humana": "aprobacion_humana", "__end__": END})
    builder.add_conditional_edges("aprobacion_humana", ruta_tras_aprobacion, {"ejecutar_accion": "ejecutar_accion", "__end__": END})
    builder.add_edge("ejecutar_accion", END)
    return builder.compile(checkpointer=checkpointer)
