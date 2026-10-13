"""Observabilidad con LangSmith o Arize Phoenix (a elección con la variable OBSERVABILITY).

Para Phoenix: OBSERVABILITY=phoenix (y Phoenix corriendo en http://localhost:6006).

LangSmith:

LangChain/LangGraph envían trazas automáticamente cuando están definidas estas variables:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=<tu clave>
    LANGSMITH_PROJECT=api-multiagente-produccion
Cada job se ejecuta dentro de un span raíz `procesar_job` (decorador @traceable), de modo que en
LangSmith se ve UNA traza por job con la jerarquía: job -> grafo -> nodo/agente -> herramienta / modelo.
"""
import os
from typing import Any, Callable

from app import config

try:  # langsmith es opcional en tests
    from langsmith import traceable as _traceable
except ImportError:  # pragma: no cover
    _traceable = None


def phoenix_activo() -> bool:
    return (os.getenv("OBSERVABILITY") or "").lower() == "phoenix"


_phoenix_iniciado = False


def iniciar_observabilidad() -> None:
    """Si OBSERVABILITY=phoenix, registra el colector de Arize Phoenix (OpenInference) una sola vez."""
    global _phoenix_iniciado
    if not phoenix_activo() or _phoenix_iniciado:
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    url = os.getenv("PHOENIX_URL", "http://localhost:6006")
    proyecto = os.getenv("PHOENIX_PROJECT", "api-multiagente-produccion")
    tracer_provider = register(
        project_name=proyecto, endpoint=f"{url}/v1/traces", batch=False, set_global_tracer_provider=False
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _phoenix_iniciado = True


def tracing_activo() -> bool:
    flag = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower() == "true"
    return flag and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def estado_observabilidad() -> str:
    if phoenix_activo():
        return f"Phoenix ACTIVO (proyecto: {os.getenv('PHOENIX_PROJECT', 'api-multiagente-produccion')}, {os.getenv('PHOENIX_URL', 'http://localhost:6006')})"
    if tracing_activo():
        return f"LangSmith ACTIVO (proyecto: {os.getenv('LANGSMITH_PROJECT', 'default')})"
    return "LangSmith DESACTIVADO (definí LANGSMITH_TRACING=true y LANGSMITH_API_KEY en .env)"


def traceable(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorador que crea el span raíz; si langsmith no está instalado no hace nada."""
    if _traceable is None or phoenix_activo():  # con Phoenix, cada ainvoke ya genera su traza (run_name=job-...)
        return lambda fn: fn
    # process_inputs: evita serializar `self` (el pool con el grafo) en cada traza.
    return _traceable(
        name=name, run_type="chain", tags=[f"prompt:{config.PROMPT_TEMPLATE_VERSION}"],
        process_inputs=lambda inputs: {"job_id": inputs.get("job_id"), "action": (inputs.get("item") or {}).get("action", "start")},
    )


def run_config(job_id: str, accion: str) -> dict[str, Any]:
    """Config de ejecución del grafo con tags y metadata visibles (y filtrables) en LangSmith."""
    return {
        "configurable": {"thread_id": job_id},          # thread_id = job_id -> checkpoints en Redis
        "recursion_limit": config.RECURSION_LIMIT,
        "run_name": f"job-{accion}",
        "tags": ["api-multiagente", f"accion:{accion}", f"prompt:{config.PROMPT_TEMPLATE_VERSION}"],
        "metadata": {"job_id": job_id, "prompt_template_version": config.PROMPT_TEMPLATE_VERSION},
    }
