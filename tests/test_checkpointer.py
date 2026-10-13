"""El checkpointer de Redis conserva el estado entre instancias (equivale a un reinicio del proceso)."""
import fakeredis
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from app.redis_checkpointer import AsyncRedisCheckpointer


def _grafo(saver):
    def pedir(state: MessagesState):
        resp = interrupt({"pregunta": "¿ok?"})
        return {"messages": [("ai", f"respuesta humana: {resp}")]}

    b = StateGraph(MessagesState)
    b.add_node("pedir", pedir)
    b.add_edge(START, "pedir")
    b.add_edge("pedir", END)
    return b.compile(checkpointer=saver)


async def test_pausa_y_reanuda_con_un_checkpointer_nuevo() -> None:
    server = fakeredis.FakeServer()
    cfg = {"configurable": {"thread_id": "t1"}}
    g1 = _grafo(AsyncRedisCheckpointer(fakeredis.FakeAsyncRedis(server=server)))
    await g1.ainvoke({"messages": [HumanMessage("hola")]}, cfg)
    assert (await g1.aget_state(cfg)).next == ("pedir",)

    # "Reinicio": objetos nuevos, mismo Redis
    g2 = _grafo(AsyncRedisCheckpointer(fakeredis.FakeAsyncRedis(server=server)))
    assert (await g2.aget_state(cfg)).next == ("pedir",)
    out = await g2.ainvoke(Command(resume="sí"), cfg)
    assert out["messages"][-1].content == "respuesta humana: sí"
    assert len([c async for c in g2.checkpointer.alist(cfg)]) >= 3     # historial persistido


async def test_hilos_aislados_y_borrado() -> None:
    saver = AsyncRedisCheckpointer(fakeredis.FakeAsyncRedis())
    g = _grafo(saver)
    for t in ("a", "b"):
        await g.ainvoke({"messages": [HumanMessage(t)]}, {"configurable": {"thread_id": t}})
    assert (await g.aget_state({"configurable": {"thread_id": "a"}})).values["messages"][0].content == "a"
    await saver.adelete_thread("a")
    assert (await saver.aget_tuple({"configurable": {"thread_id": "a"}})) is None
    assert (await saver.aget_tuple({"configurable": {"thread_id": "b"}})) is not None
