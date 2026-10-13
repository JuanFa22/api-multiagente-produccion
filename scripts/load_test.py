"""Prueba de carga: 5 peticiones CONCURRENTES contra la API.

Uso (con la API y Redis ya levantados):
    python scripts/load_test.py                 # 5 peticiones
    python scripts/load_test.py --auto-approve  # además aprueba solo los jobs que piden aprobación humana

Al terminar, abrí Phoenix (http://localhost:6006) -> tu proyecto -> pestaña "Traces": ahí están el costo por
ejecución y la latencia p95 de esta corrida (sacá las capturas para /screenshots).
"""
import argparse
import asyncio
import statistics
import time

import httpx

QUERIES: tuple[str, ...] = (
    "Investigá cómo viene el mercado de helados en el NOA entre enero y marzo de 2026 y analizá si Helado Familiar en NOA está alineado.",
    "Investigá la demanda de alfajores y comparala con las ventas de Alfajor Triple en la región Centro.",
    "Investigá la situación de las galletitas dulces en el mercado y analizá las ventas de Galletitas Dulces en Centro.",
    "Investigá la estacionalidad de los helados y analizá la variación de Helado Familiar en Centro entre enero y marzo.",
    "Investigá helados en el NOA, analizá Helado Familiar y enviá el resumen por email a gerencia.",  # CRÍTICA -> pide aprobación humana
)
TERMINALES = {"completed", "rejected", "FAILED"}


async def seguir(client: httpx.AsyncClient, job_id: str, auto_approve: bool, t0: float) -> dict:
    aprobado = False
    while True:
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] == "waiting_approval" and not aprobado:
            if auto_approve:
                await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "aprobado por load_test"})
                aprobado = True
            else:
                print(f"  [!] {job_id} espera aprobación humana: POST /tasks/{job_id}/approve")
                aprobado = True  # avisar una sola vez
        if job["status"] in TERMINALES:
            job["latencia_cliente_s"] = round(time.perf_counter() - t0, 2)
            return job
        await asyncio.sleep(0.5)


async def main(base_url: str, auto_approve: bool) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        t0 = time.perf_counter()
        resp = await asyncio.gather(*[client.post("/tasks", json={"query": q}) for q in QUERIES])   # concurrentes
        ids = [r.json()["job_id"] for r in resp]
        print(f"{len(ids)} jobs encolados en {time.perf_counter() - t0:.2f}s (la API no se bloquea): {ids}")
        jobs = await asyncio.gather(*[seguir(client, i, auto_approve, t0) for i in ids])

    lat = sorted(j["latencia_cliente_s"] for j in jobs)
    p95 = lat[min(len(lat) - 1, max(0, round(0.95 * len(lat)) - 1))]
    print("\n" + "=" * 70)
    for j in jobs:
        print(f"{j['job_id']}  {j['status']:<10} {j['latencia_cliente_s']:>6.1f}s  crítica={j.get('result', {}).get('critical')}  {j.get('error', '')}")
    print("=" * 70)
    print(f"Latencia (vista del cliente): media {statistics.mean(lat):.1f}s | p95 {p95:.1f}s | máx {lat[-1]:.1f}s")
    print("Contrastá estos números con Phoenix (costo por ejecución y p95 oficiales de la plataforma).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--auto-approve", action="store_true")
    a = p.parse_args()
    asyncio.run(main(a.base_url, a.auto_approve))
