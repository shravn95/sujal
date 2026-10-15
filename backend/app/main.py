import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# Force UTF-8 encoding for Windows standard streams
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend root directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import inngest.fast_api

from app.config import settings
from app.api import upload, query, forecast, preprocess, report, routes_upload, routes_chat, routes_datasets
from app.inngest_app.client import inngest_client
from app.inngest_app.functions import analyze_dataset
from app.services.workspace import init_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Shared app state — populated during lifespan startup
# ---------------------------------------------------------------------------
_app_state: Dict[str, Any] = {
    "llm_configured": False,
    "llm_reachable": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: check LLM config, attempt a live ping, store results in app state, initialize DB."""
    _app_state["llm_configured"] = settings.is_llm_configured
    _app_state["llm_reachable"] = False

    if not settings.is_llm_configured:
        logger.warning(
            "NVIDIA_API_KEY is not set — the app will run in fallback/rule-based mode only."
        )
    else:
        logger.info(
            "NVIDIA_API_KEY detected — attempting connectivity ping to %s (model: %s)…",
            settings.NVIDIA_BASE_URL,
            settings.NVIDIA_MODEL,
        )
        try:
            import openai

            ping_client = openai.OpenAI(
                base_url=settings.NVIDIA_BASE_URL,
                api_key=settings.NVIDIA_API_KEY,
                timeout=15.0,
                max_retries=1,
            )
            ping_client.chat.completions.create(
                model=settings.NVIDIA_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            _app_state["llm_reachable"] = True
            logger.info(
                "✓ LLM connectivity ping succeeded — Nemotron is reachable and responding."
            )
        except openai.AuthenticationError:
            logger.error(
                "✗ LLM ping failed: Authentication error — check that NVIDIA_API_KEY is valid "
                "and not expired. App will run in fallback mode."
            )
        except openai.APITimeoutError:
            logger.error(
                "✗ LLM ping failed: Request timed out — NVIDIA API may be slow or unreachable. "
                "App will run in fallback mode."
            )
        except openai.RateLimitError:
            logger.warning(
                "✗ LLM ping failed: Rate limit hit on startup — key is valid but quota may be "
                "exhausted. App will attempt LLM calls per query but may fall back."
            )
            _app_state["llm_reachable"] = False
        except Exception as exc:
            logger.error(
                "✗ LLM ping failed: %s — App will run in fallback mode.", exc
            )

    # Initialize workspace SQLite database for session tracking
    try:
        init_db()
        logger.info("✓ Workspace database initialized successfully.")
    except Exception as db_exc:
        logger.error("✗ Failed to initialize workspace database: %s", db_exc)

    yield  # Application runs here

    logger.info("AutoInsights API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AutoInsights AI Analytics Platform Engine",
    description="Backend API powered by NVIDIA Nemotron-3 Ultra 550B & DuckDB Data Analyst Engine",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend — configurable via ALLOWED_ORIGINS env
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api", tags=["AI Query & Reasoning"])
app.include_router(forecast.router, prefix="/api", tags=["Forecasting"])
app.include_router(preprocess.router, prefix="/api", tags=["Preprocessing"])
app.include_router(report.router, prefix="/api", tags=["Report Export"])

# KEEP OUR WORKING PHASE 1 & PHASE 2 CHATBOT ROUTERS:
app.include_router(routes_upload.router, prefix="/api", tags=["Phase 1: Upload & Auto-Dashboard"])
app.include_router(routes_chat.router, prefix="/api/v1", tags=["Phase 2: Dataset Q&A Chatbot"])
app.include_router(routes_datasets.router, prefix="/api", tags=["Workspace Datasets"])
# Register Inngest endpoint for background pipeline jobs
inngest.fast_api.serve(app, inngest_client, [analyze_dataset])


# ---------------------------------------------------------------------------
# Core Endpoints & Health Checks
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "engine": "AutoInsights AI Analytics Engine",
        "model": settings.NVIDIA_MODEL,
        "docs": "/docs",
    }


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "AutoInsight API",
    }


@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models():
    """OpenAI-compatible model listing endpoint for SDK compatibility."""
    return {
        "object": "list",
        "data": [
            {"id": "Nemotron-3-Ultra-550B", "object": "model", "owned_by": "nvidia"}
        ]
    }


@app.get("/api/health/llm")
@app.get("/health/llm")
def llm_health_check():
    """Return LLM connectivity status — called by the frontend status badge."""
    return {
        "llm_configured": _app_state["llm_configured"],
        "llm_reachable": _app_state["llm_reachable"],
        "model": settings.NVIDIA_MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)