"""Estado compartido y estructurado del orquestador."""
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

AgentName = Literal["Investigador", "Analista"]
NextAgent = Literal["Investigador", "Analista", "FINALIZAR"]


class Contribution(TypedDict):
    """Trazabilidad: qué agente aportó qué, en qué paso y con qué herramientas."""

    agent: str
    step: int
    instruction: str
    output: str
    tools_used: list[str]


class Decision(TypedDict):
    """Registro de cada decisión del supervisor."""

    step: int
    next_agent: str
    reason: str
    instruction: str
    guard: str | None  # motivo si una regla de seguridad cambió la decisión del LLM


class OrchestratorState(MessagesState):
    """Hereda `messages` (con reducer add_messages) y agrega campos de coordinación.

    - Cada especialista escribe SOLO en su parcela (research_output / analysis_output).
    - `contributions` y `decisions` usan el reducer `operator.add`: se acumulan, nunca se pisan.
    """

    next_agent: NextAgent
    instruction: str
    task_completed: bool
    step_count: int
    research_output: str
    analysis_output: str
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    final_answer: str
    # --- Human-in-the-loop ---
    critical: bool
    risk_reason: str
    approval: dict
    action_result: str
