from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow_store import (
    WORKFLOW_STEPS,
    add_artifact,
    create_project,
    get_project,
    list_projects,
    project_metrics,
    save_snapshot,
    set_project_status,
    update_step,
)


st.set_page_config(
    page_title="Projects and Workflows",
    page_icon="📂",
    layout="wide",
)

apply_global_style()

st.title("📂 GeoScope Projects and Workflows")

st.markdown(
    """
GeoScope analyses can be saved as persistent projects and resumed later.

A project keeps the workflow state independently from the temporary
Streamlit session:

```text
Create project
→ define AOI
→ search STAC
→ retrieve knowledge
→ generate recommendation
→ process GeoTIFF
→ evaluate
→ complete / archive
```
"""
)


# ---------------------------------------------------------------------------
# Session-state snapshot helpers
# ---------------------------------------------------------------------------

SESSION_KEYS = [
    "aoi_geojson",
    "aoi_summary",
    "aoi_label",
    "stac_scenes",
    "start_date",
    "end_date",
    "max_cloud_cover",
    "last_question",
    "last_augmented_question",
    "last_answer",
    "last_sources",
    "last_retrieval_approach",
    "last_evaluation",
    "generated_geotiff_name",
    "generated_geotiff_summary",
]


def collect_geoscope_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}

    for key in SESSION_KEYS:
        if key in st.session_state:
            value = st.session_state[key]

            # Do not persist binary GeoTIFF bytes in DuckDB JSON.
            if key == "generated_geotiff":
                continue

            snapshot[key] = value

    return snapshot


def restore_snapshot(
    snapshot: dict[str, Any],
) -> None:
    for key, value in snapshot.items():
        st.session_state[key] = value


def infer_completed_steps(
    snapshot: dict[str, Any],
) -> list[int]:
    completed: list[int] = []

    if snapshot.get("aoi_geojson"):
        completed.append(1)

    if snapshot.get("stac_scenes"):
        completed.append(2)

    if snapshot.get("last_sources"):
        completed.append(3)

    if snapshot.get("last_answer"):
        completed.append(4)

    if snapshot.get(
        "generated_geotiff_summary"
    ):
        completed.append(5)

    if snapshot.get("last_evaluation"):
        completed.append(6)

    return completed


# ---------------------------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------------------------

metrics = project_metrics()

m1, m2, m3 = st.columns(3)

m1.metric(
    "Active projects",
    metrics.get("ACTIVE", 0),
)
m2.metric(
    "Completed projects",
    metrics.get("COMPLETED", 0),
)
m3.metric(
    "Archived projects",
    metrics.get("ARCHIVED", 0),
)


# ---------------------------------------------------------------------------
# Create project
# ---------------------------------------------------------------------------

st.divider()

create_tab, manage_tab = st.tabs(
    [
        "Create new project",
        "My projects",
    ]
)

with create_tab:
    st.subheader("Create a GeoScope analysis project")

    project_name = st.text_input(
        "Project name",
        placeholder=(
            "Kom Ombo Wheat Monitoring 2026"
        ),
    )

    description = st.text_area(
        "Objective / description",
        placeholder=(
            "Monitor wheat vegetation condition using "
            "Sentinel-2 and GeoScope RAG guidance."
        ),
        height=100,
    )

    if st.button(
        "Create project",
        type="primary",
        use_container_width=True,
    ):
        try:
            project_id = create_project(
                project_name,
                description,
            )

            st.session_state[
                "active_project_id"
            ] = project_id

            st.success(
                f"Project created: {project_id}"
            )
            st.rerun()

        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------------
# Project list
# ---------------------------------------------------------------------------

with manage_tab:
    projects = list_projects(
        include_archived=True
    )

    if not projects:
        st.info(
            "No persistent GeoScope projects exist yet."
        )

    else:
        rows = []

        for project in projects:
            detail = get_project(
                project["project_id"]
            )

            completed_steps = sum(
                1
                for step in detail["steps"]
                if step["status"] == "COMPLETED"
            )

            rows.append(
                {
                    "project_id": project["project_id"],
                    "project_name": project["project_name"],
                    "status": project["status"],
                    "progress": (
                        f"{completed_steps}/"
                        f"{len(WORKFLOW_STEPS)}"
                    ),
                    "current_step": (
                        WORKFLOW_STEPS[
                            min(
                                max(
                                    int(
                                        project[
                                            "current_step"
                                        ]
                                    )
                                    - 1,
                                    0,
                                ),
                                len(WORKFLOW_STEPS)
                                - 1,
                            )
                        ]
                    ),
                    "updated_at": project["updated_at"],
                }
            )

        project_df = pd.DataFrame(rows)

        st.dataframe(
            project_df,
            use_container_width=True,
            hide_index=True,
        )

        project_ids = [
            item["project_id"]
            for item in projects
        ]

        selected_id = st.selectbox(
            "Open project",
            project_ids,
            format_func=lambda project_id: (
                next(
                    (
                        f"{item['project_name']} "
                        f"({project_id})"
                        for item in projects
                        if item["project_id"]
                        == project_id
                    ),
                    project_id,
                )
            ),
        )

        if st.button(
            "Open selected project",
            use_container_width=True,
        ):
            st.session_state[
                "active_project_id"
            ] = selected_id
            st.rerun()


# ---------------------------------------------------------------------------
# Active project
# ---------------------------------------------------------------------------

active_project_id = st.session_state.get(
    "active_project_id"
)

if active_project_id:
    project = get_project(
        active_project_id
    )

    if not project:
        st.session_state.pop(
            "active_project_id",
            None,
        )
        st.warning(
            "The active project could not be found."
        )

    else:
        st.divider()
        st.header(
            f"Project: {project['project_name']}"
        )

        st.caption(
            f"Project ID: {project['project_id']}"
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Status",
            project["status"],
        )

        completed_count = sum(
            1
            for step in project["steps"]
            if step["status"] == "COMPLETED"
        )

        p2.metric(
            "Progress",
            (
                f"{completed_count}/"
                f"{len(WORKFLOW_STEPS)}"
            ),
        )

        current_step_number = int(
            project["current_step"]
        )

        current_step_name = WORKFLOW_STEPS[
            min(
                max(
                    current_step_number - 1,
                    0,
                ),
                len(WORKFLOW_STEPS) - 1,
            )
        ]

        p3.metric(
            "Current step",
            current_step_name,
        )

        if project.get("description"):
            st.write(
                f"**Objective:** "
                f"{project['description']}"
            )

        # ---------------------------------------------------------------
        # Resume / save controls
        # ---------------------------------------------------------------

        action1, action2, action3 = st.columns(3)

        with action1:
            if st.button(
                "▶ Resume project",
                type="primary",
                use_container_width=True,
            ):
                restore_snapshot(
                    project.get(
                        "snapshot",
                        {},
                    )
                )

                st.session_state[
                    "active_project_id"
                ] = project["project_id"]

                st.success(
                    "Project state restored into the "
                    "current GeoScope session."
                )

        with action2:
            if st.button(
                "💾 Save current GeoScope state",
                use_container_width=True,
            ):
                snapshot = (
                    collect_geoscope_snapshot()
                )

                save_snapshot(
                    project["project_id"],
                    snapshot,
                )

                inferred = infer_completed_steps(
                    snapshot
                )

                for step_number in inferred:
                    update_step(
                        project["project_id"],
                        step_number,
                        "COMPLETED",
                    )

                next_pending = (
                    max(inferred) + 1
                    if inferred
                    else 1
                )

                if next_pending <= len(
                    WORKFLOW_STEPS
                ):
                    update_step(
                        project["project_id"],
                        next_pending,
                        "IN_PROGRESS",
                    )

                if snapshot.get(
                    "generated_geotiff_summary"
                ):
                    metadata = snapshot[
                        "generated_geotiff_summary"
                    ]

                    add_artifact(
                        project["project_id"],
                        "GeoTIFF",
                        snapshot.get(
                            "generated_geotiff_name",
                            "GeoTIFF output",
                        ),
                        metadata,
                    )

                st.success(
                    "Current GeoScope state persisted."
                )
                st.rerun()

        with action3:
            if st.button(
                "Close project view",
                use_container_width=True,
            ):
                st.session_state.pop(
                    "active_project_id",
                    None,
                )
                st.rerun()

        # ---------------------------------------------------------------
        # Workflow
        # ---------------------------------------------------------------

        st.subheader("Workflow")

        step_rows = []

        for step in project["steps"]:
            icon = {
                "COMPLETED": "✅",
                "IN_PROGRESS": "▶️",
                "PENDING": "○",
            }.get(
                step["status"],
                "○",
            )

            step_rows.append(
                {
                    "Step": (
                        f"{icon} "
                        f"{step['step_number']}. "
                        f"{step['step_name']}"
                    ),
                    "Status": step["status"],
                    "Notes": step.get("notes") or "",
                }
            )

        st.dataframe(
            pd.DataFrame(step_rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Update a workflow step manually"
        ):
            selected_step = st.selectbox(
                "Step",
                list(
                    range(
                        1,
                        len(WORKFLOW_STEPS)
                        + 1,
                    )
                ),
                format_func=lambda number: (
                    f"{number}. "
                    f"{WORKFLOW_STEPS[number - 1]}"
                ),
            )

            step_status = st.selectbox(
                "Status",
                [
                    "PENDING",
                    "IN_PROGRESS",
                    "COMPLETED",
                ],
            )

            step_notes = st.text_area(
                "Notes",
                height=80,
            )

            if st.button(
                "Update step"
            ):
                update_step(
                    project["project_id"],
                    selected_step,
                    step_status,
                    step_notes,
                )
                st.success(
                    "Workflow step updated."
                )
                st.rerun()

        # ---------------------------------------------------------------
        # Saved state
        # ---------------------------------------------------------------

        st.subheader("Persisted analysis state")

        snapshot = project.get(
            "snapshot",
            {},
        )

        if snapshot:
            summary_items = {
                "AOI": snapshot.get(
                    "aoi_summary"
                ),
                "STAC scene items": len(
                    snapshot.get(
                        "stac_scenes",
                        [],
                    )
                ),
                "Question": snapshot.get(
                    "last_question"
                ),
                "Retrieval approach": (
                    snapshot.get(
                        "last_retrieval_approach"
                    )
                ),
                "Answer saved": bool(
                    snapshot.get(
                        "last_answer"
                    )
                ),
                "GeoTIFF": snapshot.get(
                    "generated_geotiff_name"
                ),
                "Evaluation saved": bool(
                    snapshot.get(
                        "last_evaluation"
                    )
                ),
            }

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Item": key,
                            "Saved value": (
                                value
                                if value
                                not in {
                                    None,
                                    "",
                                }
                                else "—"
                            ),
                        }
                        for key, value
                        in summary_items.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander(
                "View saved JSON snapshot"
            ):
                st.json(snapshot)

        else:
            st.info(
                "No GeoScope session state has been "
                "saved for this project yet."
            )

        # ---------------------------------------------------------------
        # Artifacts
        # ---------------------------------------------------------------

        st.subheader("Artifacts")

        artifacts = project.get(
            "artifacts",
            [],
        )

        if artifacts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Type": item[
                                "artifact_type"
                            ],
                            "Name": item[
                                "artifact_name"
                            ],
                            "Created": item[
                                "created_at"
                            ],
                        }
                        for item in artifacts
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "No persisted artifacts yet."
            )

        # ---------------------------------------------------------------
        # Event history
        # ---------------------------------------------------------------

        st.subheader("Project history")

        events = project.get(
            "events",
            [],
        )

        if events:
            st.dataframe(
                pd.DataFrame(events),
                use_container_width=True,
                hide_index=True,
            )

        # ---------------------------------------------------------------
        # Completion / archive controls
        # ---------------------------------------------------------------

        st.divider()

        status1, status2, status3 = st.columns(3)

        with status1:
            if st.button(
                "Mark project active",
                use_container_width=True,
                disabled=(
                    project["status"]
                    == "ACTIVE"
                ),
            ):
                set_project_status(
                    project["project_id"],
                    "ACTIVE",
                )
                st.rerun()

        with status2:
            if st.button(
                "✅ Complete project",
                use_container_width=True,
                disabled=(
                    project["status"]
                    == "COMPLETED"
                ),
            ):
                update_step(
                    project["project_id"],
                    len(WORKFLOW_STEPS),
                    "COMPLETED",
                    "Project marked as completed.",
                )

                set_project_status(
                    project["project_id"],
                    "COMPLETED",
                )

                st.rerun()

        with status3:
            if st.button(
                "🗄️ Archive project",
                use_container_width=True,
                disabled=(
                    project["status"]
                    == "ARCHIVED"
                ),
            ):
                set_project_status(
                    project["project_id"],
                    "ARCHIVED",
                )
                st.rerun()
