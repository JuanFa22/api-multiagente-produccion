"""Tests de la API completa con fakeredis y modelos simulados (offline, sin Docker ni API keys)."""
import asyncio
import os

import fakeredis
import httpx
import pytest
from fakeredis import FakeServer

from app import config
from app.api import create_app
from tests.fakes import StatelessWorker, fake_graph_factory


@pytest.fixture
def patch_redis(monkeypatch):
    """Por defecto usa fakeredis (offline). Con TEST_REDIS_URL=redis://localhost:6379/15 usa un Redis real."""
    real = os.getenv("TEST_REDIS_URL")
    if real:
        from redis import Redis as SyncRedis

        SyncRedis.from_url(real).flushdb()
        monkeypatch.setattr("app.api.config.REDIS_URL", real)
        return None
    server = FakeServer()
    monkeypatch.setattr(
        "app.api.Redis.from_url",
        lambda url, decode_responses=False: fakeredis.FakeAsyncRedis(server=server, decode_responses=decode_responses),
    )
    return server


async def _esperar(client: httpx.AsyncClient, job_id: str, estados: set[str], timeout: float = 10.0) -> dict:
    fin = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < fin:
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in estados:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"timeout esperando {estados}; último estado: {job['status']}")


async def _con_app(factory, prueba, **kw):
    app = create_app(graph_factory=factory, **kw)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await prueba(client, app)


async def test_endpoint_encola_y_devuelve_job_id_al_instante(patch_redis) -> None:
    async def prueba(client, app):
        r = await client.post("/tasks", json={"query": "Investigá helados NOA y analizá Helado Familiar"})
        assert r.status_code == 202 and r.json()["status"] == "pending"
        job_id = r.json()["job_id"]
        job = await _esperar(client, job_id, {"completed"})
        assert "RESPUESTA FINAL" in job["result"]["final_answer"]
        assert [c["agent"] for c in job["result"]["contributions"]] == ["Investigador", "Analista"]
        assert job["result"]["critical"] is False

    await _con_app(fake_graph_factory(), prueba)


async def test_estado_inicial_existe_antes_de_encolar(patch_redis) -> None:
    async def prueba(client, app):
        r = await client.post("/tasks", json={"query": "Investigá helados NOA y analizá ventas"})
        assert (await client.get(f"/tasks/{r.json()['job_id']}")).status_code == 200   # nunca 404
        assert (await client.get("/tasks/inexistente")).status_code == 404

    await _con_app(fake_graph_factory(), prueba, start_workers=False)


async def test_excepcion_en_background_deja_el_job_en_FAILED(patch_redis) -> None:
    async def prueba(client, app):
        job_id = (await client.post("/tasks", json={"query": "Investigá helados NOA y analizá ventas"})).json()["job_id"]
        job = await _esperar(client, job_id, {"FAILED"})
        assert "fallo simulado del LLM" in job["error"]

    await _con_app(fake_graph_factory(StatelessWorker(fail=True)), prueba)


async def test_hitl_pausa_y_se_ejecuta_solo_tras_aprobar(patch_redis) -> None:
    async def prueba(client, app):
        q = "Investigá helados NOA, analizá Helado Familiar y enviá el resumen por email a gerencia"
        job_id = (await client.post("/tasks", json={"query": q})).json()["job_id"]
        job = await _esperar(client, job_id, {"waiting_approval"})
        assert job["interrupt"]["accion_propuesta"]["tipo"] == "enviar_email"
        assert await app.state.redis.llen(config.OUTBOX_KEY) == 0           # NADA se ejecutó todavía

        assert (await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 202
        job = await _esperar(client, job_id, {"completed"})
        assert job["result"]["approval"] is True and "Email enviado" in job["result"]["action_result"]
        assert await app.state.redis.llen(config.OUTBOX_KEY) == 1           # el efecto ocurrió UNA vez
        # no se puede aprobar dos veces
        assert (await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409

    await _con_app(fake_graph_factory(), prueba)


async def test_hitl_rechazo_no_ejecuta_la_accion(patch_redis) -> None:
    async def prueba(client, app):
        q = "Investigá helados NOA, analizá Helado Familiar y enviá el resumen por email a gerencia"
        job_id = (await client.post("/tasks", json={"query": q})).json()["job_id"]
        await _esperar(client, job_id, {"waiting_approval"})
        await client.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "no es el momento"})
        job = await _esperar(client, job_id, {"rejected"})
        assert job["result"]["approval"] is False
        assert await app.state.redis.llen(config.OUTBOX_KEY) == 0

    await _con_app(fake_graph_factory(), prueba)


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="requiere un Redis real: TEST_REDIS_URL=redis://localhost:6379/15")
async def test_el_estado_sobrevive_al_reinicio_de_la_api(patch_redis) -> None:
    """Pausa en la API #1, 'reinicio' (app nueva, mismo Redis) y aprobación en la API #2."""
    q = "Investigá helados NOA, analizá Helado Familiar y enviá el resumen por email a gerencia"
    guardado: dict = {}

    async def antes(client, app):
        guardado["job_id"] = (await client.post("/tasks", json={"query": q})).json()["job_id"]
        await _esperar(client, guardado["job_id"], {"waiting_approval"})

    async def despues(client, app):
        job_id = guardado["job_id"]
        assert (await client.get(f"/tasks/{job_id}")).json()["status"] == "waiting_approval"
        await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
        job = await _esperar(client, job_id, {"completed"})
        assert "Email enviado" in job["result"]["action_result"]

    await _con_app(fake_graph_factory(), antes)      # API #1 (se apaga al salir)
    await _con_app(fake_graph_factory(), despues)    # API #2 (grafo y checkpointer nuevos, mismo Redis)


async def test_cinco_peticiones_concurrentes(patch_redis) -> None:
    async def prueba(client, app):
        qs = ["Investigá helados NOA y analizá Helado Familiar"] * 5
        respuestas = await asyncio.gather(*[client.post("/tasks", json={"query": q}) for q in qs])
        ids = [r.json()["job_id"] for r in respuestas]
        assert len(set(ids)) == 5
        jobs_ = await asyncio.gather(*[_esperar(client, i, {"completed"}) for i in ids])
        assert all(j["status"] == "completed" for j in jobs_)

    await _con_app(fake_graph_factory(), prueba)


def test_clasificacion_de_riesgo() -> None:
    from app.hitl import evaluar_riesgo

    assert evaluar_riesgo("Analizá la demanda de alfajores y comparala con ventas")[0] is False   # 'demanda' != 'mandá'
    assert evaluar_riesgo("Investigá el mercado y enviá el resumen por email")[0] is True
    assert evaluar_riesgo("Mandá un correo a gerencia con las ventas")[0] is True
    assert evaluar_riesgo("Estudiá la aplicación de precios en el mercado")[0] is False
