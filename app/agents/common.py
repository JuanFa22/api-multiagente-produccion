"""Utilidades compartidas por los agentes especialistas."""
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


def texto(content: Any) -> str:
    """Normaliza el contenido de un mensaje (str o lista de bloques) a texto plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
    return str(content)


def respuesta_final(messages: list[BaseMessage]) -> str:
    """Último mensaje del modelo sin tool_calls = respuesta del especialista."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and texto(msg.content):
            return texto(msg.content)
    return ""


def herramientas_usadas(messages: list[BaseMessage]) -> list[str]:
    """Nombres de las herramientas que el especialista llamó (para la traza)."""
    return [call["name"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
