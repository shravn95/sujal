import asyncio
import json
import os
import inngest
import pandas as pd
from app.config import settings
from app.models.schemas import DatasetUnderstanding, LLMAnalysisResponse
from app.services.nemotron_service import (
    understand_dataset,
    plan_dashboard,
    generate_fallback_analysis,
)
from app.services.chart_validator import measure_set
from app.services.pipeline_status import write_status
from .client import inngest_client, ANALYZE_EVENT

MAX_RETRIES = 2  # 3 attempts total


def _path(dataset_id: str, suffix: str) -> str:
    return os.path.join(settings.STORAGE_DIR, f"{dataset_id}{suffix}")


def _load_meta(dataset_id: str) -> dict:
    with open(_path(dataset_id, ".meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _save_analysis(dataset_id: str, analysis: dict, plan_source: str) -> None:
    with open(_path(dataset_id, ".analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"plan_source": plan_source, "analysis": analysis}, f)


async def _on_failure(ctx: inngest.Context) -> None:
    """Runs once all retries are exhausted: build the local fallback so the user still gets a dashboard."""
    original = ctx.event.data.get("event", {}).get("data", {})
    dataset_id = original.get("dataset_id")
    if not dataset_id:
        return
    error = ctx.event.data.get("error", {}).get("message", "AI analysis failed")
    meta = _load_meta(dataset_id)
    fallback = generate_fallback_analysis(meta["llm_profile"], meta["filename"])
    _save_analysis(dataset_id, fallback.model_dump(), "fallback")
    write_status(dataset_id, state="done", fallback_reason=str(error)[:200])


def _llm_configured() -> bool:
    return bool(settings.NVIDIA_API_KEY and settings.NVIDIA_API_KEY != "nvapi-your-actual-key-here")


@inngest_client.create_function(
    fn_id="analyze-dataset",
    trigger=inngest.TriggerEvent(event=ANALYZE_EVENT),
    retries=MAX_RETRIES,
    on_failure=_on_failure,
)
async def analyze_dataset(ctx: inngest.Context) -> dict:
    dataset_id = ctx.event.data["dataset_id"]
    attempt = getattr(ctx, "attempt", 0) + 1

    def _status(stage: str) -> None:
        write_status(
            dataset_id,
            state="running",
            stage=stage,
            attempt=attempt,
            max_attempts=MAX_RETRIES + 1,
            message=None,
        )

    async def understand() -> dict:
        _status("understanding")
        meta = _load_meta(dataset_id)
        if not _llm_configured():
            fb = generate_fallback_analysis(meta["llm_profile"], meta["filename"])
            _save_analysis(dataset_id, fb.model_dump(), "fallback")
            write_status(dataset_id, state="done", fallback_reason="LLM not configured")
            return {"skipped": True}
        try:
            u = await asyncio.to_thread(understand_dataset, meta["llm_profile"], meta["filename"])
        except Exception as e:
            write_status(dataset_id, state="retrying", message=str(e)[:200])
            raise
        return u.model_dump()

    understanding = await ctx.step.run("understand-dataset", understand)
    if understanding.get("skipped"):
        return {"plan_source": "fallback"}

    async def plan() -> dict:
        _status("planning")
        meta = _load_meta(dataset_id)
        u = DatasetUnderstanding.model_validate(understanding)
        try:
            df = pd.read_parquet(_path(dataset_id, ".parquet"))
            profile = meta["llm_profile"]

            blocked = (
                set(profile.get("identifier_columns", []))
                | set(profile.get("junk_columns", []))
                | set(profile.get("constant_columns", []))
                | set(profile.get("time_columns", []))
            )

            print("\n========== MEASURE DEBUG ==========")
            print("DataFrame shape:", df.shape)
            print("Profile numeric columns:", profile.get("numeric_columns", []))
            print("Profile time columns:", profile.get("time_columns", []))
            print("Blocked columns:", sorted(blocked))
            print("DataFrame columns:", df.columns.tolist())
            print("DataFrame dtypes:\n", df.dtypes)

            for c in df.columns:
                s = df[c]
                cleaned = s.dropna().astype(str).str.replace(",", "", regex=False).str.strip()
                convertible_ratio = (
                    pd.to_numeric(cleaned, errors="coerce").notna().mean()
                    if len(cleaned) else 0
                )
                print(
                    f"{c!r}: dtype={s.dtype}, "
                    f"blocked={c in blocked}, "
                    f"numeric_dtype={pd.api.types.is_numeric_dtype(s)}, "
                    f"numeric_conversion_ratio={convertible_ratio:.1%}, "
                    f"sample={s.dropna().head(3).tolist()}"
                )

            print("Detected measures:", measure_set(df, profile))
            print("===================================\n")

            result = await asyncio.to_thread(
                plan_dashboard, profile, u, df
            )
        except Exception as e:
            write_status(dataset_id, state="retrying", message=str(e)[:200])
            raise
        analysis = LLMAnalysisResponse(
            executive_summary=u.executive_summary,
            key_observations=u.key_observations,
            kpis=result.kpis,
            charts=result.charts,
        )
        _save_analysis(dataset_id, analysis.model_dump(), "llm")
        write_status(dataset_id, state="done", fallback_reason=None)
        return {"plan_source": "llm", "chart_count": len(result.charts)}

    return await ctx.step.run("plan-charts", plan)


@inngest_client.create_function(
    fn_id="audit-chat-query",
    trigger=inngest.TriggerEvent(event="chat/query.executed"),
)

@inngest_client.create_function(
    fn_id="audit-chat-query",
    trigger=inngest.TriggerEvent(event="chat/query.executed"),
)
async def audit_chat_query(ctx: inngest.Context, step: inngest.Step):
    data = ctx.event.data
    dataset_id = data.get("dataset_id")
    user_message = data.get("message")
    sql_query = data.get("sql_query")

    @step.run("verify-dataset-audit")
    def verify_audit():
        return {
            "status": "audited",
            "dataset_id": dataset_id,
            "query_length": len(str(user_message)),
            "sql_executed": sql_query[:50] if sql_query else "none"
        }

    return verify_audit()