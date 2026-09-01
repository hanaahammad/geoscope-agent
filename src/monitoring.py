from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DB = PROJECT_ROOT / "logs" / "runs.duckdb"
DLT_DB = PROJECT_ROOT / "logs" / "geoscope_monitoring.duckdb"


RUN_COLUMNS: dict[str, str] = {
    # Domain / geographic context
    "application": "VARCHAR",
    "crop": "VARCHAR",
    "season": "VARCHAR",
    "aoi_summary": "VARCHAR",
    "aoi_geojson": "VARCHAR",
    "stac_scene_count": "INTEGER",
    "start_date": "VARCHAR",
    "end_date": "VARCHAR",
    "max_cloud_cover": "DOUBLE",

    # AI execution metadata
    "framework": "VARCHAR",
    "execution_mode": "VARCHAR",
    "model": "VARCHAR",
    "prompt_id": "VARCHAR",
    "prompt_version": "VARCHAR",

    # Retrieval metadata
    "retrieval_approach": "VARCHAR",
    "original_query": "VARCHAR",
    "rewritten_query": "VARCHAR",
    "top_k": "INTEGER",
    "candidate_k": "INTEGER",

    # Context metadata
    "chunk_count": "INTEGER",
    "context_characters": "INTEGER",
    "estimated_context_tokens": "INTEGER",

    # Framework trace / agent metadata
    "trace_json": "VARCHAR",
    "tool_calls_json": "VARCHAR",
    "step_count": "INTEGER",
}


def _json_or_none(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def init_db() -> None:
    """
    Create the local run store and evolve the schema safely.

    Existing installations are preserved. New observability columns are added
    only when they do not already exist.
    """
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(LOG_DB)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR,
                created_at TIMESTAMP,
                question VARCHAR,
                answer VARCHAR,
                sources_json VARCHAR,
                latency_seconds DOUBLE,
                status VARCHAR,
                error_message VARCHAR
            )
            """
        )

        existing_columns = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info('runs')"
            ).fetchall()
        }

        for column_name, column_type in RUN_COLUMNS.items():
            if column_name not in existing_columns:
                con.execute(
                    f'ALTER TABLE runs ADD COLUMN "{column_name}" {column_type}'
                )


def log_run(
    run_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    latency_seconds: float,
    status: str,
    error_message: str = "",

    # Domain / geographic context
    application: str = "",
    crop: str = "",
    season: str = "",
    aoi_summary: str = "",
    aoi_geojson: dict[str, Any] | None = None,
    stac_scene_count: int = 0,
    start_date: str = "",
    end_date: str = "",
    max_cloud_cover: float | None = None,

    # AI execution
    framework: str = "",
    execution_mode: str = "",
    model: str = "",
    prompt_id: str = "",
    prompt_version: str = "",

    # Retrieval
    retrieval_approach: str = "",
    original_query: str = "",
    rewritten_query: str = "",
    top_k: int | None = None,
    candidate_k: int | None = None,

    # Context
    chunk_count: int | None = None,
    context_characters: int | None = None,
    estimated_context_tokens: int | None = None,

    # Trace / agent metadata
    trace: list[dict[str, Any]] | dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    step_count: int | None = None,
) -> None:
    """Store one GeoScope execution with AI-engineering metadata."""
    init_db()

    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "question": question,
        "answer": answer,
        "sources_json": json.dumps(
            sources or [],
            ensure_ascii=False,
            default=str,
        ),
        "latency_seconds": latency_seconds,
        "status": status,
        "error_message": error_message,

        "application": application,
        "crop": crop,
        "season": season,
        "aoi_summary": aoi_summary,
        "aoi_geojson": _json_or_none(aoi_geojson),
        "stac_scene_count": int(stac_scene_count or 0),
        "start_date": start_date,
        "end_date": end_date,
        "max_cloud_cover": max_cloud_cover,

        "framework": framework,
        "execution_mode": execution_mode,
        "model": model,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,

        "retrieval_approach": retrieval_approach,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "top_k": top_k,
        "candidate_k": candidate_k,

        "chunk_count": chunk_count,
        "context_characters": context_characters,
        "estimated_context_tokens": estimated_context_tokens,

        "trace_json": _json_or_none(trace),
        "tool_calls_json": _json_or_none(tool_calls),
        "step_count": step_count,
    }

    columns = list(record.keys())
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(f'"{c}"' for c in columns)

    with duckdb.connect(str(LOG_DB)) as con:
        con.execute(
            f"INSERT INTO runs ({column_sql}) VALUES ({placeholders})",
            [record[c] for c in columns],
        )


def _read_dlt_table(table_suffix: str) -> pd.DataFrame:
    """
    Read a dlt-created table without assuming its schema name.

    dlt normally creates tables inside the configured `monitoring` dataset.
    This discovery approach keeps the monitoring UI resilient if the exact
    schema name changes.
    """
    if not DLT_DB.exists():
        return pd.DataFrame()

    try:
        with duckdb.connect(str(DLT_DB), read_only=True) as con:
            tables = con.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE lower(table_name) = lower(?)
                   OR lower(table_name) LIKE lower(?)
                ORDER BY
                    CASE WHEN lower(table_name) = lower(?) THEN 0 ELSE 1 END,
                    table_schema,
                    table_name
                """,
                [
                    table_suffix,
                    f"%{table_suffix}%",
                    table_suffix,
                ],
            ).fetchall()

            if not tables:
                return pd.DataFrame()

            schema, table = tables[0]
            return con.execute(
                f'SELECT * FROM "{schema}"."{table}"'
            ).df()
    except Exception:
        # Monitoring must remain usable even if the auxiliary dlt DB is
        # temporarily unavailable or has an old schema.
        return pd.DataFrame()


def _latest_by_run_id(
    df: pd.DataFrame,
    timestamp_candidates: list[str],
) -> pd.DataFrame:
    if df.empty or "run_id" not in df.columns:
        return pd.DataFrame()

    timestamp_col = next(
        (c for c in timestamp_candidates if c in df.columns),
        None,
    )

    result = df.copy()

    if timestamp_col:
        result[timestamp_col] = pd.to_datetime(
            result[timestamp_col],
            errors="coerce",
            utc=True,
        )
        result = result.sort_values(timestamp_col)

    return result.drop_duplicates(subset=["run_id"], keep="last")


def recent_runs(limit: int = 100) -> pd.DataFrame:
    """
    Return recent runs enriched with judge evaluation and human feedback.

    Base execution metadata is read from logs/runs.duckdb.
    Generation evaluation and user feedback are read from the dlt-managed
    logs/geoscope_monitoring.duckdb and joined by run_id when available.
    """
    init_db()

    with duckdb.connect(str(LOG_DB)) as con:
        runs = con.execute(
            """
            SELECT *
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).df()

    if runs.empty:
        return runs

    # User-friendly aliases used by Page 5.
    runs["sources"] = runs.get("sources_json")
    runs["retrieved_chunks"] = runs.get("sources_json")
    runs["context_tokens"] = runs.get("estimated_context_tokens")
    runs["trace"] = runs.get("trace_json")
    runs["trajectory"] = runs.get("trace_json")

    # ------------------------------------------------------------------
    # Enrich with generation evaluation written through dlt.
    # ------------------------------------------------------------------
    evaluations = _latest_by_run_id(
        _read_dlt_table("generation_evaluations"),
        ["evaluation_timestamp", "_dlt_load_id"],
    )

    if not evaluations.empty:
        wanted = [
            "run_id",
            "judge_model",
            "judge_provider",
            "relevance",
            "groundedness",
            "completeness",
            "technical_correctness",
            "citation_quality",
            "geographic_relevance",
            "overall",
            "judge_verdict",
            "failure_category",
            "comment",
        ]
        eval_cols = [c for c in wanted if c in evaluations.columns]
        evaluations = evaluations[eval_cols]

        runs = runs.merge(
            evaluations,
            on="run_id",
            how="left",
            suffixes=("", "_judge"),
        )

    # ------------------------------------------------------------------
    # Enrich with latest human feedback written through dlt.
    # ------------------------------------------------------------------
    feedback = _latest_by_run_id(
        _read_dlt_table("user_feedback"),
        ["feedback_timestamp", "_dlt_load_id"],
    )

    if not feedback.empty:
        wanted = [
            "run_id",
            "rating",
            "comment",
            "feedback_timestamp",
        ]
        fb_cols = [c for c in wanted if c in feedback.columns]
        feedback = feedback[fb_cols].rename(
            columns={
                "rating": "human_feedback",
                "comment": "human_feedback_comment",
            }
        )

        runs = runs.merge(
            feedback,
            on="run_id",
            how="left",
        )

    return runs
