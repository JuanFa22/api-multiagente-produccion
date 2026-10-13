"""Agente de Análisis: cómputo sobre las ventas internas (CSV) y comparación con lo investigado."""
import asyncio
import csv
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
    "Sos el Agente de Análisis. Calculás con tus herramientas, nunca de memoria. "
    "Usá las ventas internas para obtener cifras y calcular variaciones, y compará con los datos "
    "de mercado que te pasen en la instrucción. Si una herramienta devuelve ERROR, corregí los "
    "parámetros con las opciones que indica y reintentá. Respondé en español, máximo 8 líneas, "
    "indicando si el desempeño está alineado, por encima o por debajo del mercado."
)


@lru_cache(maxsize=1)
def _ventas() -> tuple[dict[str, Any], ...]:
    with config.VENTAS_CSV.open(encoding="utf-8", newline="") as fh:
        return tuple({**r, "unidades": int(r["unidades"]), "monto": float(r["monto"])} for r in csv.DictReader(fh))


@tool
async def ventas_por_mes(producto: str, region: str | None = None) -> dict[str, Any] | str:
    """Devuelve la serie mensual (unidades y monto en ARS) de un producto, opcionalmente de una región.

    Args:
        producto: nombre EXACTO del producto ('Alfajor Triple', 'Galletitas Dulces' o 'Helado Familiar').
        region: región opcional ('Centro' o 'NOA'). Si se omite suma todas las regiones.

    Si el producto o la región no existen devuelve 'ERROR: ...' con los valores válidos.
    """

    def _calcular() -> dict[str, Any] | str:
        filas = list(_ventas())
        productos = sorted({r["producto"] for r in filas})
        regiones = sorted({r["region"] for r in filas})
        if producto.strip().lower() not in {p.lower() for p in productos}:
            return f"ERROR: producto '{producto}' no existe. Valores válidos: {', '.join(productos)}."
        if region and region.strip().lower() not in {r.lower() for r in regiones}:
            return f"ERROR: region '{region}' no existe. Valores válidos: {', '.join(regiones)}."
        sel = [r for r in filas if r["producto"].lower() == producto.strip().lower()
               and (not region or r["region"].lower() == region.strip().lower())]
        meses: dict[str, dict[str, float]] = {}
        for r in sel:
            m = meses.setdefault(r["mes"], {"unidades": 0, "monto_ars": 0.0})
            m["unidades"] += r["unidades"]
            m["monto_ars"] += r["monto"]
        return {"producto": sel[0]["producto"], "region": region or "todas", "meses": dict(sorted(meses.items()))}

    return await asyncio.to_thread(_calcular)


@tool
async def calcular_variacion(valor_actual: float, valor_anterior: float) -> dict[str, float] | str:
    """Calcula la variación porcentual entre dos valores (actual vs. anterior).

    Usala para medir el crecimiento de un período contra otro. Devuelve la
    diferencia absoluta y el porcentaje de variación, o 'ERROR: ...' si el valor anterior es 0.
    """
    if valor_anterior == 0:
        return "ERROR: el valor anterior es 0; no se puede calcular una variación porcentual."
    return {"diferencia": round(valor_actual - valor_anterior, 2),
            "variacion_pct": round((valor_actual - valor_anterior) / valor_anterior * 100, 2)}


TOOLS = [ventas_por_mes, calcular_variacion]


def crear_nodo_analista(model: BaseChatModel):
    agente = create_react_agent(model, TOOLS, prompt=PROMPT, name="analista")

    async def analista_node(state: OrchestratorState) -> dict[str, Any]:
        paso = state.get("step_count", 0)
        # Contexto mínimo: instrucción del supervisor + el hallazgo del investigador (no todo el historial).
        contexto = state.get("research_output") or "(todavía no hay datos de mercado)"
        instruccion = f"{state['instruction']}\n\nDatos de mercado aportados por el Investigador:\n{contexto}"
        resultado = await agente.ainvoke({"messages": [HumanMessage(content=instruccion)]})
        salida = respuesta_final(resultado["messages"]) or "Sin análisis."
        aporte: Contribution = {
            "agent": "Analista", "step": paso, "instruction": state["instruction"],
            "output": salida, "tools_used": herramientas_usadas(resultado["messages"]),
        }
        return {
            "messages": [HumanMessage(content=salida, name="Analista")],
            "analysis_output": salida,
            "contributions": [aporte],
        }

    return analista_node
