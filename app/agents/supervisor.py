"""Nodo Supervisor (router inteligente) y nodo de síntesis final."""
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app import config
from app.agents.common import texto
from app.state import Decision, NextAgent, OrchestratorState

SUPERVISOR_PROMPT = (
    "Sos el Supervisor de un equipo de dos especialistas:\n"
    "- Investigador: busca información de MERCADO (tendencias, estacionalidad, benchmarks).\n"
    "- Analista: calcula sobre las VENTAS INTERNAS y compara con el mercado.\n"
    "Dada la conversación y el estado actual, decidí quién debe intervenir ahora o si es momento de FINALIZAR.\n"
    "Reglas:\n"
    "1. Normalmente primero el Investigador y después el Analista (el Analista necesita los datos de mercado).\n"
    "2. Podés pedir UN refinamiento a un especialista si su salida es incompleta o no cita cifras; "
    "en 'instruction' explicá qué falta.\n"
    "3. Elegí FINALIZAR solo cuando ambos aportaron y la información es suficiente.\n"
    "4. Vos coordinás: nunca hagas el trabajo de los especialistas.\n"
    "En 'instruction' escribí una orden concreta y autocontenida para el especialista elegido."
)


class RouteDecision(BaseModel):
    """Salida estructurada del supervisor: el LLM solo puede elegir nodos válidos."""

    next_agent: Literal["Investigador", "Analista", "FINALIZAR"] = Field(description="Siguiente nodo o FINALIZAR.")
    reason: str = Field(description="Justificación breve de la decisión.")
    instruction: str = Field(default="", description="Instrucción concreta para el especialista elegido.")


def _visitas(state: OrchestratorState, agente: str) -> int:
    return sum(1 for c in state.get("contributions", []) if c["agent"] == agente)


def aplicar_guardas(decision: RouteDecision, state: OrchestratorState) -> tuple[RouteDecision, str | None]:
    """Reglas deterministas que acotan al LLM: condición de parada clara + criterio de suficiencia."""
    paso = state.get("step_count", 0) + 1
    # 1) Tope de pasos: corta el "Supervisor Infinito".
    if paso > config.MAX_SUPERVISOR_STEPS:
        return RouteDecision(next_agent="FINALIZAR", reason="Tope de pasos alcanzado.", instruction=""), "max_steps"
    # 2) Tope de visitas por especialista (solo un refinamiento).
    if decision.next_agent in ("Investigador", "Analista") and _visitas(state, decision.next_agent) >= config.MAX_VISITS_PER_AGENT:
        return RouteDecision(next_agent="FINALIZAR", reason=f"{decision.next_agent} ya alcanzó el máximo de intervenciones.", instruction=""), "max_visits"
    # 3) Suficiencia: no se puede finalizar sin los dos dominios.
    if decision.next_agent == "FINALIZAR":
        for faltante in ("Investigador", "Analista"):
            if _visitas(state, faltante) == 0:
                return RouteDecision(next_agent=faltante, reason=f"Falta el aporte del {faltante}.",
                                     instruction=decision.instruction or state["messages"][0].content), "missing_domain"
    return decision, None


def _regla_de_respaldo(state: OrchestratorState) -> RouteDecision:
    """Decisión determinista si el LLM del supervisor falla (cuota, red, formato)."""
    consulta = texto(state["messages"][0].content)
    if _visitas(state, "Investigador") == 0:
        return RouteDecision(next_agent="Investigador", reason="Respaldo: falta investigación de mercado.", instruction=consulta)
    if _visitas(state, "Analista") == 0:
        return RouteDecision(next_agent="Analista", reason="Respaldo: falta el análisis de ventas.", instruction=consulta)
    return RouteDecision(next_agent="FINALIZAR", reason="Respaldo: ambos especialistas ya aportaron.", instruction="")


def _resumen_estado(state: OrchestratorState) -> str:
    partes = [f"Consulta original: {texto(state['messages'][0].content)}", f"Pasos del supervisor hasta ahora: {state.get('step_count', 0)}"]
    for c in state.get("contributions", []):
        partes.append(f"- Aporte de {c['agent']} (paso {c['step']}): {c['output']}")
    if not state.get("contributions"):
        partes.append("- Todavía no intervino ningún especialista.")
    return "\n".join(partes)


def crear_nodo_supervisor(model: BaseChatModel):
    router = model.with_structured_output(RouteDecision)

    async def supervisor_node(state: OrchestratorState) -> dict[str, Any]:
        paso = state.get("step_count", 0) + 1
        try:
            decision = await router.ainvoke([("system", SUPERVISOR_PROMPT), HumanMessage(content=_resumen_estado(state))])
            if decision is None:
                raise ValueError("el modelo no devolvió una decisión estructurada")
        except Exception as exc:  # recuperación de errores: el flujo no se cae
            print(f"[Supervisor] fallo del LLM ({type(exc).__name__}); uso regla de respaldo.")
            decision = _regla_de_respaldo(state)
        final, guarda = aplicar_guardas(decision, state)
        registro: Decision = {"step": paso, "next_agent": final.next_agent, "reason": final.reason,
                              "instruction": final.instruction, "guard": guarda}
        return {
            "next_agent": final.next_agent,
            "instruction": final.instruction,
            "task_completed": final.next_agent == "FINALIZAR",
            "step_count": paso,
            "decisions": [registro],
        }

    return supervisor_node


SINTESIS_PROMPT = (
    "Sos el redactor de la respuesta final. Con los aportes de los especialistas, respondé la consulta "
    "original en español, en un máximo de 10 líneas: conclusión primero, luego cifras clave y fuentes [M-xx]. "
    "No agregues datos que no estén en los aportes."
)


def crear_nodo_sintesis(model: BaseChatModel):
    async def sintesis_node(state: OrchestratorState) -> dict[str, Any]:
        resp = await model.ainvoke([("system", SINTESIS_PROMPT), HumanMessage(content=_resumen_estado(state))])
        final = texto(resp.content)
        return {"final_answer": final, "messages": [AIMessage(content=final, name="Sintesis")]}

    return sintesis_node
