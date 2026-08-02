from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DB = PROJECT_ROOT / "logs" / "runs.duckdb"


def init_db() -> None:
    """
    Create the monitoring database and migrate the runs table when newer
    context columns are missing.
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

        required_columns = {
            "application": "VARCHAR",
            "crop": "VARCHAR",
            "season": "VARCHAR",
            "aoi_summary": "VARCHAR",
            "aoi_geojson": "VARCHAR",
            "stac_scene_count": "INTEGER",
            "start_date": "VARCHAR",
            "end_date": "VARCHAR",
            "max_cloud_cover": "DOUBLE",
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                con.execute(
                    f"ALTER TABLE runs "
                    f"ADD COLUMN {column_name} {column_type}"
                )


def log_run(
    run_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    latency_seconds: float,
    status: str,
    error_message: str = "",
    application: str = "",
    crop: str = "",
    season: str = "",
    aoi_summary: str = "",
    aoi_geojson: dict[str, Any] | None = None,
    stac_scene_count: int = 0,
    start_date: str = "",
    end_date: str = "",
    max_cloud_cover: float | None = None,
) -> None:
    """
    Store one GeoScope run together with its AOI, STAC, and user-filter context.
    """
    init_db()

    with duckdb.connect(str(LOG_DB)) as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id,
                created_at,
                question,
                answer,
                sources_json,
                latency_seconds,
                status,
                error_message,
                application,
                crop,
                season,
                aoi_summary,
                aoi_geojson,
                stac_scene_count,
                start_date,
                end_date,
                max_cloud_cover
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                datetime.now(timezone.utc).replace(tzinfo=None),
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                float(latency_seconds),
                status,
                error_message,
                application,
                crop,
                season,
                aoi_summary,
                (
                    json.dumps(aoi_geojson, ensure_ascii=False)
                    if aoi_geojson
                    else None
                ),
                int(stac_scene_count),
                start_date,
                end_date,
                max_cloud_cover,
            ],
        )


def recent_runs(limit: int = 100):
    """
    Return recent runs with geographic and analysis context for Streamlit.
    """
    init_db()

    with duckdb.connect(str(LOG_DB)) as con:
        return con.execute(
            """
            SELECT
                created_at,
                question,
                application,
                crop,
                season,
                aoi_summary,
                stac_scene_count,
                start_date,
                end_date,
                max_cloud_cover,
                latency_seconds,
                status
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).df()
