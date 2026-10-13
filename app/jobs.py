"""Estado de los jobs en Redis (hash `job:{id}`)."""
import json
import time
from typing import Any

from redis.asyncio import Redis

from app import config

PENDING = "pending"
RUNNING = "running"
WAITING_APPROVAL = "waiting_approval"
COMPLETED = "completed"
REJECTED = "rejected"
FAILED = "FAILED"
TERMINAL = {COMPLETED, REJECTED, FAILED}


class JobStore:
    """Cliente Redis con decode_responses=True."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def _key(job_id: str) -> str:
        return f"{config.JOB_PREFIX}{job_id}"

    async def create(self, job_id: str, query: str) -> None:
        # Se registra ANTES de encolar: evita el 404 si el worker o el cliente son más rápidos.
        now = time.time()
        await self.redis.hset(self._key(job_id), mapping={
            "job_id": job_id, "status": PENDING, "query": query,
            "created_at": now, "updated_at": now,
        })

    async def update(self, job_id: str, **fields: Any) -> None:
        mapping = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)) for k, v in fields.items()}
        mapping["updated_at"] = str(time.time())
        await self.redis.hset(self._key(job_id), mapping=mapping)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self.redis.hgetall(self._key(job_id))
        if not raw:
            return None
        job: dict[str, Any] = dict(raw)
        for field in ("result", "interrupt"):
            if field in job:
                job[field] = json.loads(job[field])
        for field in ("created_at", "updated_at", "started_at", "finished_at"):
            if field in job:
                job[field] = float(job[field])
        if "finished_at" in job:
            job["duration_s"] = round(job["finished_at"] - job["created_at"], 2)
        return job

    async def enqueue(self, item: dict[str, Any]) -> None:
        await self.redis.rpush(config.QUEUE_NAME, json.dumps(item))
