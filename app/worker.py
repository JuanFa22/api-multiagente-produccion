"""Workers asíncronos: sacan jobs de la cola Redis, ejecutan el grafo y actualizan el estado."""
import asyncio
import json
import time
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from app import config, jobs
from app.agents.common import texto
from app.jobs import JobStore
from app.observability import run_config, traceable


class WorkerPool:
    def __init__(self, redis: Redis, graph: CompiledStateGraph, store: JobStore, concurrency: int | None = None) -> None:
        self.redis, self.graph, self.store = redis, graph, store
        self.concurrency = concurrency or config.WORKER_CONCURRENCY
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._loop(i), name=f"worker-{i}") for i in range(self.concurrency)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _loop(self, n: int) -> None:
        while True:
            try:
                popped = await self.redis.blpop(config.QUEUE_NAME, timeout=1)
                if popped is None:
                    continue
                await self.process(json.loads(popped[1]))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # un item corrupto o un fallo de Redis no debe matar al worker
                print(f"[worker-{n}] error inesperado: {type(exc).__name__}: {exc}")
                await asyncio.sleep(0.5)

    async def process(self, item: dict[str, Any]) -> None:
        """Procesa un item de la cola. NUNCA propaga: cualquier excepción deja el job en FAILED."""
        job_id = item["job_id"]
        try:
            await self.store.update(job_id, status=jobs.RUNNING, started_at=time.time())
            await asyncio.wait_for(self._run(job_id, item), timeout=config.JOB_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.store.update(job_id, status=jobs.FAILED, error=f"Timeout: el job superó {config.JOB_TIMEOUT_S}s", finished_at=time.time())
        except Exception as exc:
            await self.store.update(job_id, status=jobs.FAILED, error=f"{type(exc).__name__}: {exc}", finished_at=time.time())

    @traceable("procesar_job")
    async def _run(self, job_id: str, item: dict[str, Any]) -> None:
        accion = item.get("action", "start")
        cfg = run_config(job_id, accion)
        if accion == "resume":
            entrada: Any = Command(resume=item["resume"])
        else:
            job = await self.store.get(job_id)
            entrada = {"messages": [("user", job["query"])], "step_count": 0, "contributions": [], "decisions": []}

        await self.graph.ainvoke(entrada, cfg)
        snapshot = await self.graph.aget_state(cfg)
        pendientes = [i for t in snapshot.tasks for i in t.interrupts]
        if snapshot.next and pendientes:
            # El grafo quedó PAUSADO en el nodo humano: el estado vive en el checkpointer de Redis.
            await self.store.update(job_id, status=jobs.WAITING_APPROVAL, interrupt=pendientes[0].value)
            return

        v = snapshot.values
        aprobacion = v.get("approval") or {}
        rechazado = bool(v.get("critical")) and not aprobacion.get("aprobado", False)
        resultado = {
            "final_answer": v.get("final_answer", ""),
            "critical": bool(v.get("critical")),
            "approval": aprobacion.get("aprobado"),
            "action_result": v.get("action_result", ""),
            "steps": v.get("step_count", 0),
            "decisions": [{"step": d["step"], "next": d["next_agent"], "reason": d["reason"]} for d in v.get("decisions", [])],
            "contributions": [{"agent": c["agent"], "tools": c["tools_used"]} for c in v.get("contributions", [])],
        }
        await self.store.update(job_id, status=jobs.REJECTED if rechazado else jobs.COMPLETED, result=resultado, finished_at=time.time())
