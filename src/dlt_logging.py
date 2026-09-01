from __future__ import annotations

from pathlib import Path
from typing import Any

import dlt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "logs" / "geoscope_monitoring.duckdb"


def _pipeline():
    """
    dlt pipeline used for evaluation and human-feedback event logging.

    The main execution/run record is stored in logs/runs.duckdb by
    src.monitoring.log_run(). dlt is used here for append-oriented structured
    events whose schemas can evolve independently.
    """
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    return dlt.pipeline(
        pipeline_name="geoscope_monitoring",
        destination=dlt.destinations.duckdb(
            credentials=str(DUCKDB_PATH)
        ),
        dataset_name="monitoring",
    )


def log_retrieval_evaluation(
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    _pipeline().run(
        records,
        table_name="retrieval_evaluations",
        write_disposition="append",
    )


def log_generation_evaluation(
    record: dict[str, Any],
) -> None:
    if not record:
        return

    _pipeline().run(
        [record],
        table_name="generation_evaluations",
        write_disposition="append",
    )


def log_user_feedback(
    record: dict[str, Any],
) -> None:
    if not record:
        return

    _pipeline().run(
        [record],
        table_name="user_feedback",
        write_disposition="append",
    )
