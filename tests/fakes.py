"""Modelos simulados SIN estado (seguros para jobs concurrentes): no necesitan API key."""
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agents.supervisor import RouteDecision

TOOL_ARGS = {
    "buscar_mercado": {"consulta": "helados NOA verano"},
    "ventas_por_mes": {"producto": "Helado Familiar", "region": "NOA"},
}


class StatelessWorker(BaseChatModel):
    """Si el último mensaje es un ToolMessage responde en texto; si no, llama a su herramienta."""

    tool_name: str = "buscar_mercado"
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "stateless-worker"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StatelessWorker":
        return StatelessWorker(tool_name=tools[0].name, fail=self.fail)

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if self.fail:
            raise RuntimeError("fallo simulado del LLM")
        if isinstance(messages[-1], ToolMessage):
            msg = AIMessage(content=f"Resumen basado en {self.tool_name} con cifras [M-02].")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": self.tool_name, "args": TOOL_ARGS[self.tool_name], "id": "t1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _Router:
    async def ainvoke(self, messages: Any, *a: Any, **k: Any) -> RouteDecision:
        resumen = messages[-1].content
        if "Aporte de Investigador" not in resumen:
            return RouteDecision(next_agent="Investigador", reason="falta mercado", instruction="Buscá helados NOA")
        if "Aporte de Analista" not in resumen:
            return RouteDecision(next_agent="Analista", reason="falta análisis", instruction="Analizá Helado Familiar NOA")
        return RouteDecision(next_agent="FINALIZAR", reason="suficiente", instruction="")


class StatelessSupervisor:
    """Fake del modelo del supervisor: ruteo por estado + síntesis fija."""

    def with_structured_output(self, schema: Any) -> _Router:
        return _Router()

    async def ainvoke(self, messages: Any, *a: Any, **k: Any) -> AIMessage:
        return AIMessage(content="RESPUESTA FINAL: helados NOA alineados con el mercado.")


def fake_graph_factory(worker: BaseChatModel | None = None):
    from app.graph import construir_grafo

    def factory(checkpointer: Any, ejecutor: Any):
        w = worker or StatelessWorker()
        return construir_grafo(StatelessSupervisor(), w, checkpointer=checkpointer, action_executor=ejecutor)

    return factory
