"""Configuración central de la API de producción."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent.parent
VENTAS_CSV: Path = BASE_DIR / "data" / "ventas.csv"
MERCADO_JSON: Path = BASE_DIR / "data" / "mercado.json"

# --- LLM ---
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "6"))  # reintentos ante 429 / cuota

# --- Redis ---
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME: str = "queue:tasks"
JOB_PREFIX: str = "job:"
OUTBOX_KEY: str = "outbox:emails"

# --- Worker ---
WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "5"))
JOB_TIMEOUT_S: int = int(os.getenv("JOB_TIMEOUT_S", "300"))

# --- Observabilidad (LangSmith) ---
PROMPT_TEMPLATE_VERSION: str = os.getenv("PROMPT_TEMPLATE_VERSION", "v1")

# --- Condiciones de parada del orquestador (heredadas de la Pre-entrega 6) ---
MAX_SUPERVISOR_STEPS: int = 8
MAX_VISITS_PER_AGENT: int = 2
RECURSION_LIMIT: int = 40

# --- Human-in-the-loop ---
REPORT_RECIPIENT: str = os.getenv("REPORT_RECIPIENT", "gerencia@empresa-demo.com")
# Una consulta es CRÍTICA si pide una acción con efectos en el mundo real (patrones con límites
# de palabra: "demanda" NO debe activar "mandá").
CRITICAL_PATTERNS: tuple[str, ...] = (
    r"\benvi[aáe](r|lo|la)?\b", r"\bmand[aá](r|lo|la)?\b", r"\b(e-?mail|correo|mail)s?\b",
    r"\bpublic[aá](r|lo|la)?\b", r"\baplic[aá](r|lo|la)?\b", r"\bejecut[aá](r|lo|la)?\b",
    r"\bborr[aá](r|lo|la)?\b", r"\belimin[aá](r|lo|la)?\b", r"\breembols\w*",
)
