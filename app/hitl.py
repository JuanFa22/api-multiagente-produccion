"""Human-in-the-loop: detección de tareas críticas, nodo de aprobación y ejecución de la acción."""
import re
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app import config
from app.agents.common import texto
from app.state import OrchestratorState

# Ejecutor de la acción con efectos secundarios (se inyecta; en producción escribe en Redis).
ActionExecutor = Callable[[dict[str, Any]], Awaitable[str]]


def evaluar_riesgo(consulta: str) -> tuple[bool, str]:
    """Regla determinista: ¿la consulta pide una acción con efectos secundarios?"""
    q = consulta.lower()
    hallada = [m.group(0) for p in config.CRITICAL_PATTERNS if (m := re.search(p, q))]
    if hallada:
        return True, f"La consulta pide una acción con efectos externos (detectado: {', '.join(hallada)})."
    return False, "Solo lectura y análisis: no requiere aprobación."


async def evaluar_riesgo_node(state: OrchestratorState) -> dict[str, Any]:
    critica, motivo = evaluar_riesgo(texto(state["messages"][0].content))
    return {"critical": critica, "risk_reason": motivo}


def ruta_tras_riesgo(state: OrchestratorState) -> str:
    return "aprobacion_humana" if state.get("critical") else "__end__"


async def aprobacion_humana_node(state: OrchestratorState) -> dict[str, Any]:
    """Pausa la ejecución hasta recibir una aprobación externa.

    IMPORTANTE: al reanudar, este nodo se re-ejecuta desde el principio, por eso NO hay
    código con efectos antes del interrupt(); la acción real ocurre en el nodo siguiente.
    """
    accion = {
        "tipo": "enviar_email",
        "destinatario": config.REPORT_RECIPIENT,
        "asunto": "Informe de ventas y mercado",
        "cuerpo": state.get("final_answer", ""),
    }
    decision = interrupt({
        "pregunta": "¿Aprobás ejecutar esta acción?",
        "motivo": state.get("risk_reason", ""),
        "accion_propuesta": accion,
    })
    if isinstance(decision, dict):
        aprobado, comentario = bool(decision.get("aprobado")), str(decision.get("comentario", ""))
    else:
        aprobado, comentario = str(decision).strip().lower() in {"si", "sí", "yes", "true", "ok"}, ""
    return {"approval": {"aprobado": aprobado, "comentario": comentario, "accion": accion}}


def ruta_tras_aprobacion(state: OrchestratorState) -> str:
    return "ejecutar_accion" if state.get("approval", {}).get("aprobado") else "__end__"


def crear_nodo_ejecutar_accion(executor: ActionExecutor):
    async def ejecutar_accion_node(state: OrchestratorState) -> dict[str, Any]:
        resultado = await executor(state["approval"]["accion"])
        return {"action_result": resultado, "messages": [AIMessage(content=resultado, name="Ejecutor")]}

    return ejecutar_accion_node


async def ejecutor_en_memoria(accion: dict[str, Any]) -> str:
    """Ejecutor por defecto (tests / uso local): simula el envío sin efectos reales."""
    return f"[SIMULADO] Email enviado a {accion['destinatario']}"
