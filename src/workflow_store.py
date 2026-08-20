from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "geoscope_workflows.duckdb"

WORKFLOW_STEPS = [
    "AOI defined",
    "STAC search",
    "Knowledge retrieval",
    "GeoAI recommendation",
    "GeoTIFF processing",
    "Evaluation",
    "Completion",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    _ensure_schema(con)
    return con


def _ensure_schema(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_projects (
            project_id VARCHAR PRIMARY KEY,
            project_name VARCHAR NOT NULL,
            description VARCHAR,
            status VARCHAR NOT NULL,
            current_step INTEGER NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            completed_at VARCHAR,
            archived_at VARCHAR,
            snapshot_json VARCHAR
        )
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_steps (
            project_id VARCHAR NOT NULL,
            step_number INTEGER NOT NULL,
            step_name VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            started_at VARCHAR,
            completed_at VARCHAR,
            notes VARCHAR,
            PRIMARY KEY (project_id, step_number)
        )
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_events (
            event_id VARCHAR PRIMARY KEY,
            project_id VARCHAR NOT NULL,
            event_type VARCHAR NOT NULL,
            event_message VARCHAR,
            created_at VARCHAR NOT NULL
        )
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_artifacts (
            artifact_id VARCHAR PRIMARY KEY,
            project_id VARCHAR NOT NULL,
            artifact_type VARCHAR NOT NULL,
            artifact_name VARCHAR,
            metadata_json VARCHAR,
            created_at VARCHAR NOT NULL
        )
        """
    )


def create_project(
    project_name: str,
    description: str = "",
) -> str:
    project_name = project_name.strip()

    if not project_name:
        raise ValueError("Project name is required.")

    project_id = f"GEO-{uuid.uuid4().hex[:8].upper()}"
    now = _now()

    with _connect() as con:
        con.execute(
            """
            INSERT INTO analysis_projects
            (
                project_id,
                project_name,
                description,
                status,
                current_step,
                created_at,
                updated_at,
                snapshot_json
            )
            VALUES (?, ?, ?, 'ACTIVE', 1, ?, ?, ?)
            """,
            [
                project_id,
                project_name,
                description.strip(),
                now,
                now,
                json.dumps({}),
            ],
        )

        for index, step_name in enumerate(
            WORKFLOW_STEPS,
            start=1,
        ):
            con.execute(
                """
                INSERT INTO analysis_steps
                (
                    project_id,
                    step_number,
                    step_name,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    project_id,
                    index,
                    step_name,
                    "IN_PROGRESS"
                    if index == 1
                    else "PENDING",
                ],
            )

        _add_event(
            con,
            project_id,
            "PROJECT_CREATED",
            f"Project created: {project_name}",
        )

    return project_id


def _add_event(
    con,
    project_id: str,
    event_type: str,
    message: str,
) -> None:
    con.execute(
        """
        INSERT INTO analysis_events
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            project_id,
            event_type,
            message,
            _now(),
        ],
    )


def list_projects(
    include_archived: bool = True,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            project_id,
            project_name,
            description,
            status,
            current_step,
            created_at,
            updated_at,
            completed_at,
            archived_at
        FROM analysis_projects
    """

    if not include_archived:
        query += " WHERE status <> 'ARCHIVED' "

    query += " ORDER BY updated_at DESC "

    with _connect() as con:
        rows = con.execute(query).fetchall()
        columns = [
            item[0]
            for item in con.description
        ]

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def get_project(
    project_id: str,
) -> dict[str, Any] | None:
    with _connect() as con:
        row = con.execute(
            """
            SELECT *
            FROM analysis_projects
            WHERE project_id = ?
            """,
            [project_id],
        ).fetchone()

        if not row:
            return None

        columns = [
            item[0]
            for item in con.description
        ]
        project = dict(zip(columns, row))

        steps = con.execute(
            """
            SELECT
                step_number,
                step_name,
                status,
                started_at,
                completed_at,
                notes
            FROM analysis_steps
            WHERE project_id = ?
            ORDER BY step_number
            """,
            [project_id],
        ).fetchall()

        project["steps"] = [
            {
                "step_number": row[0],
                "step_name": row[1],
                "status": row[2],
                "started_at": row[3],
                "completed_at": row[4],
                "notes": row[5],
            }
            for row in steps
        ]

        events = con.execute(
            """
            SELECT
                event_type,
                event_message,
                created_at
            FROM analysis_events
            WHERE project_id = ?
            ORDER BY created_at DESC
            """,
            [project_id],
        ).fetchall()

        project["events"] = [
            {
                "event_type": row[0],
                "message": row[1],
                "created_at": row[2],
            }
            for row in events
        ]

        artifacts = con.execute(
            """
            SELECT
                artifact_id,
                artifact_type,
                artifact_name,
                metadata_json,
                created_at
            FROM analysis_artifacts
            WHERE project_id = ?
            ORDER BY created_at DESC
            """,
            [project_id],
        ).fetchall()

        project["artifacts"] = [
            {
                "artifact_id": row[0],
                "artifact_type": row[1],
                "artifact_name": row[2],
                "metadata": (
                    json.loads(row[3])
                    if row[3]
                    else {}
                ),
                "created_at": row[4],
            }
            for row in artifacts
        ]

        snapshot = project.get("snapshot_json")

        project["snapshot"] = (
            json.loads(snapshot)
            if snapshot
            else {}
        )

    return project


def save_snapshot(
    project_id: str,
    snapshot: dict[str, Any],
) -> None:
    now = _now()

    with _connect() as con:
        con.execute(
            """
            UPDATE analysis_projects
            SET snapshot_json = ?, updated_at = ?
            WHERE project_id = ?
            """,
            [
                json.dumps(
                    snapshot,
                    default=str,
                ),
                now,
                project_id,
            ],
        )

        _add_event(
            con,
            project_id,
            "SNAPSHOT_SAVED",
            "Current GeoScope analysis state saved.",
        )


def update_step(
    project_id: str,
    step_number: int,
    status: str,
    notes: str = "",
) -> None:
    status = status.upper().strip()

    if status not in {
        "PENDING",
        "IN_PROGRESS",
        "COMPLETED",
    }:
        raise ValueError(
            "Invalid step status."
        )

    now = _now()

    with _connect() as con:
        current = con.execute(
            """
            SELECT status
            FROM analysis_steps
            WHERE project_id = ?
              AND step_number = ?
            """,
            [project_id, step_number],
        ).fetchone()

        if not current:
            raise ValueError(
                "Workflow step not found."
            )

        started_at = (
            now
            if status == "IN_PROGRESS"
            and current[0] == "PENDING"
            else None
        )

        completed_at = (
            now
            if status == "COMPLETED"
            else None
        )

        con.execute(
            """
            UPDATE analysis_steps
            SET
                status = ?,
                started_at = COALESCE(started_at, ?),
                completed_at = COALESCE(?, completed_at),
                notes = CASE
                    WHEN ? <> '' THEN ?
                    ELSE notes
                END
            WHERE project_id = ?
              AND step_number = ?
            """,
            [
                status,
                started_at,
                completed_at,
                notes,
                notes,
                project_id,
                step_number,
            ],
        )

        current_step = con.execute(
            """
            SELECT COALESCE(
                MIN(step_number),
                ?
            )
            FROM analysis_steps
            WHERE project_id = ?
              AND status <> 'COMPLETED'
            """,
            [
                len(WORKFLOW_STEPS),
                project_id,
            ],
        ).fetchone()[0]

        con.execute(
            """
            UPDATE analysis_projects
            SET current_step = ?, updated_at = ?
            WHERE project_id = ?
            """,
            [
                current_step,
                now,
                project_id,
            ],
        )

        _add_event(
            con,
            project_id,
            "STEP_UPDATED",
            (
                f"Step {step_number} updated "
                f"to {status}."
            ),
        )


def add_artifact(
    project_id: str,
    artifact_type: str,
    artifact_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    artifact_id = str(uuid.uuid4())

    with _connect() as con:
        con.execute(
            """
            INSERT INTO analysis_artifacts
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                artifact_id,
                project_id,
                artifact_type,
                artifact_name,
                json.dumps(
                    metadata or {},
                    default=str,
                ),
                _now(),
            ],
        )

        _add_event(
            con,
            project_id,
            "ARTIFACT_ADDED",
            (
                f"{artifact_type}: "
                f"{artifact_name}"
            ),
        )

    return artifact_id


def set_project_status(
    project_id: str,
    status: str,
) -> None:
    status = status.upper().strip()

    if status not in {
        "ACTIVE",
        "COMPLETED",
        "ARCHIVED",
    }:
        raise ValueError(
            "Invalid project status."
        )

    now = _now()
    completed_at = (
        now if status == "COMPLETED" else None
    )
    archived_at = (
        now if status == "ARCHIVED" else None
    )

    with _connect() as con:
        con.execute(
            """
            UPDATE analysis_projects
            SET
                status = ?,
                updated_at = ?,
                completed_at = COALESCE(?, completed_at),
                archived_at = COALESCE(?, archived_at)
            WHERE project_id = ?
            """,
            [
                status,
                now,
                completed_at,
                archived_at,
                project_id,
            ],
        )

        _add_event(
            con,
            project_id,
            "STATUS_CHANGED",
            f"Project status changed to {status}.",
        )


def project_metrics() -> dict[str, int]:
    with _connect() as con:
        rows = con.execute(
            """
            SELECT status, COUNT(*)
            FROM analysis_projects
            GROUP BY status
            """
        ).fetchall()

    values = {
        "ACTIVE": 0,
        "COMPLETED": 0,
        "ARCHIVED": 0,
    }

    for status, count in rows:
        values[status] = count

    return values
