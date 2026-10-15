import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI

from app.config import settings
from app.models.schemas import DatasetChatRequest, ChartSpec
from app.services.chat_service import execute_dataset_sql, get_parquet_path
from app.services.chart_hydrator import hydrate_charts
from app.services.nemotron_service import clean_json_string

router = APIRouter()

client = OpenAI(
    base_url=settings.NVIDIA_BASE_URL,
    api_key=settings.NVIDIA_API_KEY,
)

def sse_event(event_type: str, content=None, data=None) -> str:
    payload = {"type": event_type}
    if content is not None:
        payload["content"] = content
    if data is not None:
        payload["data"] = data
    return f"data: {json.dumps(payload)}\n\n"

@router.post("/chat")
async def chat_with_dataset_stream(req: DatasetChatRequest):
    try:
        parquet_path = get_parquet_path(req.dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    async def event_generator():
        try:
            df = pd.read_parquet(parquet_path)

            col_summaries = []
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                samples = df[col].dropna().astype(str).unique()[:4].tolist()
                col_summaries.append({
                    "name": col,
                    "dtype": dtype_str,
                    "sample_values": samples
                })

            schema_context = {
                "table_name": "dataset",
                "row_count": len(df),
                "columns": col_summaries,
            }

            yield sse_event("thinking", "Inspecting dataset schema and formulating DuckDB SQL query...\n")

            sql_system_prompt = (
                "You are an expert DuckDB SQL & Data Visualization Analyst.\n"
                "The user is asking a question about a table named `dataset`.\n"
                "CRITICAL RULES:\n"
                "1. Always wrap column names with spaces or special characters in double quotes, e.g., \"Total Spent\", \"Transaction Date\".\n"
                "2. Write a valid, read-only DuckDB SELECT query against `FROM dataset`.\n"
                "3. Return ONLY a raw JSON object with these keys:\n"
                "{\n"
                '  "sql_query": "SELECT ...",\n'
                '  "explanation": "Brief explanation",\n'
                '  "needs_chart": true or false,\n'
                '  "chart_type": "bar" | "line" | "pie" | "scatter" | "area",\n'
                '  "chart_title": "Chart Title",\n'
                '  "x_column": "Exact column name from dataset",\n'
                '  "y_column": "Exact column name from dataset",\n'
                '  "aggregation": "sum" | "avg" | "count" | "min" | "max"\n'
                "}"
            )

            sql_user_prompt = (
                f"Dataset Schema:\n{json.dumps(schema_context, indent=2)}\n\n"
                f"User Question: {req.message}\n\n"
                "Return ONLY the JSON object."
            )

            completion = client.chat.completions.create(
                model=settings.NEMOTRON_MODEL,
                messages=[
                    {"role": "system", "content": sql_system_prompt},
                    {"role": "user", "content": sql_user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                stream=True,
            )

            raw_sql_json = ""
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield sse_event("thinking", reasoning)
                if delta.content:
                    raw_sql_json += delta.content

            parsed_plan = json.loads(clean_json_string(raw_sql_json))
            sql_query = parsed_plan.get("sql_query", "SELECT * FROM dataset LIMIT 10")

            yield sse_event("thinking", f"\n\nExecuting DuckDB SQL:\n{sql_query}\n")

            records, sql_err = execute_dataset_sql(req.dataset_id, sql_query)
            if sql_err:
                yield sse_event("thinking", f"\nSQL warning ({sql_err}), falling back to summary stats...\n")
                records, _ = execute_dataset_sql(req.dataset_id, "SELECT * FROM dataset LIMIT 15")

            yield sse_event("thinking", "\nSynthesizing analytical response from query results...\n")

            synth_prompt = (
                f"User Question: {req.message}\n"
                f"SQL Query Executed: {sql_query}\n"
                f"SQL Query Results (top rows):\n{json.dumps(records[:25], indent=2)}\n\n"
                "Write a clear, concise, data-backed answer directly addressing the user's question. "
                "Use bullet points and exact numbers from the SQL results. Do not include <think> tags in the final answer."
            )

            ans_completion = client.chat.completions.create(
                model=settings.NEMOTRON_MODEL,
                messages=[{"role": "user", "content": synth_prompt}],
                temperature=0.4,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=False,
            )
            answer_text = ans_completion.choices[0].message.content or ""
            answer_text = clean_json_string(answer_text) if answer_text.startswith("{") else answer_text.strip()

            suggested_charts = []
            x_col = parsed_plan.get("x_column")
            y_col = parsed_plan.get("y_column")
            if parsed_plan.get("needs_chart") and x_col in df.columns and (not y_col or y_col in df.columns):
                c_type = parsed_plan.get("chart_type") or "bar"
                if c_type not in ("bar", "line", "scatter", "pie", "area"):
                    c_type = "bar"
                agg = parsed_plan.get("aggregation") or "sum"
                if agg not in ("sum", "avg", "count", "min", "max", "none"):
                    agg = "sum"

                spec = ChartSpec(
                    id=f"chat-{ os.urandom(3).hex() }",
                    title=parsed_plan.get("chart_title") or f"{y_col or 'Count'} by {x_col}",
                    description=parsed_plan.get("explanation") or f"Custom visualization for: {req.message}",
                    chart_type=c_type,
                    x_column=x_col,
                    y_column=y_col or x_col,
                    aggregation=agg,
                )
                hydrated = hydrate_charts(df, [spec])
                for h in hydrated:
                    type_label_map = {
                        "bar": "Bar Chart",
                        "line": "Line Graph",
                        "area": "Area Graph",
                        "pie": "Pie Chart",
                        "scatter": "Scatterplot",
                    }
                    suggested_charts.append({
                        "id": h.id,
                        "title": h.title,
                        "type": type_label_map.get(h.chart_type, "Bar Chart"),
                        "x_axis": h.x_column,
                        "y_axis": h.y_column,
                        "aggregation": h.aggregation,
                        "data": h.data,
                        "insight_tooltip": h.description,
                    })

            final_payload = {
                "answer": answer_text,
                "sql_query": sql_query,
                "query_results": records[:20],
                "detailed_report": f"# Analysis for: {req.message}\n\n{answer_text}\n\n### Executed SQL\n```sql\n{sql_query}\n```",
                "plan_source": "llm",
            }
            if suggested_charts:
                final_payload["suggested_charts"] = suggested_charts

            yield sse_event("payload", data=final_payload)

        except Exception as exc:
            yield sse_event("error", content=str(exc))

    return StreamingResponse(event_generator(), media_type="text/event-stream")