"""Checkpointer de LangGraph sobre Redis (async), solo con comandos básicos de Redis.

No requiere módulos extra (RedisJSON / RediSearch): funciona con `redis:7`, Redis Cloud o Memurai.
Guarda el estado de cada hilo (`thread_id` = job_id) para que sobreviva a reinicios de la API
y permita pausar / reanudar en el nodo Human-in-the-loop.

Claves (todas con el thread_id en la 3ª posición):
    lg:cp:{thread}:{ns}:{checkpoint_id}   hash   checkpoint + metadata + padre
    lg:idx:{thread}:{ns}                  zset   índice de checkpoint_ids (orden lexicográfico = cronológico)
    lg:blob:{thread}:{ns}:{canal}:{ver}   hash   valor serializado de cada canal
    lg:wr:{thread}:{ns}:{checkpoint_id}   hash   escrituras pendientes (incluye los interrupts)
"""
import base64
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP, BaseCheckpointSaver, ChannelVersions, Checkpoint, CheckpointMetadata,
    CheckpointTuple, get_checkpoint_id, get_checkpoint_metadata,
)
from redis.asyncio import Redis


def _s(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class AsyncRedisCheckpointer(BaseCheckpointSaver[str]):
    def __init__(self, client: Redis, *, prefix: str = "lg") -> None:
        """`client` debe crearse SIN decode_responses (los datos son binarios)."""
        super().__init__()
        self.client = client
        self.prefix = prefix

    # ------------------------------------------------------------------ claves
    def _k(self, kind: str, thread_id: str, ns: str, *rest: Any) -> str:
        return ":".join([self.prefix, kind, thread_id, ns, *map(str, rest)])

    # ------------------------------------------------------------------ lectura
    async def _tuple(self, thread_id: str, ns: str, checkpoint_id: str, config: RunnableConfig | None = None) -> CheckpointTuple | None:
        data = await self.client.hgetall(self._k("cp", thread_id, ns, checkpoint_id))
        if not data:
            return None
        d = {_s(k): v for k, v in data.items()}
        checkpoint: Checkpoint = self.serde.loads_typed((_s(d["cp_t"]), d["cp"]))
        metadata = self.serde.loads_typed((_s(d["md_t"]), d["md"]))
        parent_id = _s(d["parent"]) or None

        values: dict[str, Any] = {}
        for channel, version in checkpoint["channel_versions"].items():
            blob = await self.client.hgetall(self._k("blob", thread_id, ns, channel, version))
            if not blob:
                continue
            b = {_s(k): v for k, v in blob.items()}
            if _s(b["t"]) == "empty":
                continue
            values[channel] = self.serde.loads_typed((_s(b["t"]), b["d"]))

        raw_writes = await self.client.hgetall(self._k("wr", thread_id, ns, checkpoint_id))
        parsed = []
        for field, raw in raw_writes.items():
            w = json.loads(_s(raw))
            task_id, idx = _s(field).rsplit("|", 1)
            parsed.append((task_id, int(idx), w["c"], self.serde.loads_typed((w["t"], base64.b64decode(w["d"])))))
        parsed.sort(key=lambda x: (x[0], x[1]))
        pending = [(task_id, channel, value) for task_id, _, channel, value in parsed]

        base_cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}}
        return CheckpointTuple(
            config=config or base_cfg,
            checkpoint={**checkpoint, "channel_values": values},
            metadata=metadata,
            pending_writes=pending,
            parent_config=(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": parent_id}} if parent_id else None
            ),
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            return await self._tuple(thread_id, ns, checkpoint_id, config)
        latest = await self.client.zrevrangebylex(self._k("idx", thread_id, ns), "+", "-", start=0, num=1)
        if not latest:
            return None
        return await self._tuple(thread_id, ns, _s(latest[0]))

    async def alist(
        self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None, limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config:
            index_keys = [self._k("idx", config["configurable"]["thread_id"], config["configurable"].get("checkpoint_ns", ""))]
        else:
            index_keys = [_s(k) async for k in self.client.scan_iter(match=f"{self.prefix}:idx:*")]
        wanted_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        remaining = limit
        for key in index_keys:
            _, _, thread_id, ns = key.split(":", 3)
            for raw_id in await self.client.zrevrangebylex(key, "+", "-"):
                cid = _s(raw_id)
                if wanted_id and cid != wanted_id:
                    continue
                if before_id and cid >= before_id:
                    continue
                tup = await self._tuple(thread_id, ns, cid)
                if tup is None:
                    continue
                if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                    continue
                if remaining is not None and remaining <= 0:
                    return
                if remaining is not None:
                    remaining -= 1
                yield tup

    # ------------------------------------------------------------------ escritura
    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        pipe = self.client.pipeline()
        for channel, version in new_versions.items():
            t, d = self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
            pipe.hset(self._k("blob", thread_id, ns, channel, version), mapping={"t": t, "d": d})
        cp_t, cp_d = self.serde.dumps_typed(c)
        md_t, md_d = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        pipe.hset(self._k("cp", thread_id, ns, checkpoint["id"]), mapping={
            "cp_t": cp_t, "cp": cp_d, "md_t": md_t, "md": md_d,
            "parent": config["configurable"].get("checkpoint_id") or "",
        })
        pipe.zadd(self._k("idx", thread_id, ns), {checkpoint["id"]: 0})
        await pipe.execute()
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint["id"]}}

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        key = self._k("wr", thread_id, ns, config["configurable"]["checkpoint_id"])
        pipe = self.client.pipeline()
        for idx, (channel, value) in enumerate(writes):
            real_idx = WRITES_IDX_MAP.get(channel, idx)
            t, d = self.serde.dumps_typed(value)
            payload = json.dumps({"c": channel, "t": t, "d": base64.b64encode(d).decode(), "p": task_path})
            field = f"{task_id}|{real_idx}"
            if real_idx >= 0:
                pipe.hsetnx(key, field, payload)   # no pisar escrituras ya registradas
            else:
                pipe.hset(key, field, payload)     # canales especiales (interrupt, error...) sí se actualizan
        await pipe.execute()

    async def adelete_thread(self, thread_id: str) -> None:
        keys = [k async for k in self.client.scan_iter(match=f"{self.prefix}:*:{thread_id}:*")]
        if keys:
            await self.client.delete(*keys)
