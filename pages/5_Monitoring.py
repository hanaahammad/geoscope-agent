from __future__ import annotations

import sys
from pathlib import Path

import duckdb

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import recent_runs
from src.ui import apply_global_style, page_hero


st.set_page_config(
    page_title="Monitoring",
    page_icon="📊",
    layout="wide",
)

apply_global_style()


GOVERNANCE_DB = (
    PROJECT_ROOT
    / "logs"
    / "geoscope_monitoring.duckdb"
)


def _read_governance_table(
    table_name: str,
) -> pd.DataFrame:
    """
    Read a dlt-created monitoring table when it exists.

    dlt normally stores these tables under the `monitoring` schema.
    The function intentionally returns an empty DataFrame when the
    project has not generated that type of record yet.
    """
    if not GOVERNANCE_DB.exists():
        return pd.DataFrame()

    candidates = [
        f'monitoring.{table_name}',
        table_name,
    ]

    try:
        with duckdb.connect(
            str(GOVERNANCE_DB),
            read_only=True,
        ) as con:
            for candidate in candidates:
                try:
                    return con.execute(
                        f"SELECT * FROM {candidate}"
                    ).df()
                except Exception:
                    continue
    except Exception:
        pass

    return pd.DataFrame()


def _numeric_average(
    frame: pd.DataFrame,
    column: str,
) -> float | None:
    if frame.empty or column not in frame.columns:
        return None

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def _metric_value(
    value: float | None,
) -> str:
    if value is None:
        return "Not evaluated"

    return f"{value:.2f} / 5"


def _positive_feedback_rate(
    feedback: pd.DataFrame,
) -> tuple[str, int]:
    if feedback.empty or "rating" not in feedback.columns:
        return "Not evaluated", 0

    ratings = (
        feedback["rating"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    rated = ratings[ratings.ne("")]

    if rated.empty:
        return "Not evaluated", 0

    positives = rated.str.contains(
        "Yes",
        case=False,
        regex=False,
    ).sum()

    return (
        f"{100.0 * positives / len(rated):.1f}%",
        len(rated),
    )

page_hero(
    "📊 Monitoring",
    (
        "Explore GeoScope usage, geographic context, response performance, "
        "STAC availability, and run status through interactive filters and charts."
    ),
    eyebrow="Step 5",
    chips=["DuckDB logs", "AOI context", "Interactive charts", "Run inspection"],
)

st.markdown(
    """
<div class="gs-section-note">
The dashboard is backward-compatible with old records. Fields that were not
logged previously are displayed as <strong>Not recorded</strong>.
</div>
""",
    unsafe_allow_html=True,
)

try:
    runs = recent_runs(limit=1000)
except ModuleNotFoundError as exc:
    st.error(
        "DuckDB is missing from the active environment. Run "
        "`python -m pip install duckdb`."
    )
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not load monitoring data: {exc}")
    st.stop()

expected_columns = {
    "created_at": pd.NaT,
    "question": "",
    "application": "",
    "crop": "",
    "season": "",
    "aoi_summary": "",
    "stac_scene_count": 0,
    "start_date": "",
    "end_date": "",
    "max_cloud_cover": None,
    "latency_seconds": 0.0,
    "status": "",
}

for column_name, default_value in expected_columns.items():
    if column_name not in runs.columns:
        runs[column_name] = default_value

runs = runs[list(expected_columns.keys())].copy()

if runs.empty:
    st.info(
        "No runs are available yet. Ask a question from Step 3 and return here."
    )
    st.stop()

text_columns = [
    "question",
    "application",
    "crop",
    "season",
    "aoi_summary",
    "status",
    "start_date",
    "end_date",
]

for column in text_columns:
    runs[column] = (
        runs[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Not recorded")
    )

runs["created_at"] = pd.to_datetime(
    runs["created_at"],
    errors="coerce",
)

runs["latency_seconds"] = pd.to_numeric(
    runs["latency_seconds"],
    errors="coerce",
)

runs["stac_scene_count"] = pd.to_numeric(
    runs["stac_scene_count"],
    errors="coerce",
).fillna(0).astype(int)

runs["max_cloud_cover"] = pd.to_numeric(
    runs["max_cloud_cover"],
    errors="coerce",
)

st.subheader("Explore the logs")

f1, f2, f3, f4 = st.columns(4)

with f1:
    application_options = sorted(
        runs["application"].dropna().unique().tolist()
    )
    selected_applications = st.multiselect(
        "Application",
        application_options,
        default=application_options,
    )

with f2:
    status_options = sorted(
        runs["status"].dropna().unique().tolist()
    )
    selected_statuses = st.multiselect(
        "Status",
        status_options,
        default=status_options,
    )

with f3:
    crop_options = sorted(
        runs["crop"].dropna().unique().tolist()
    )
    selected_crops = st.multiselect(
        "Crop",
        crop_options,
        default=crop_options,
    )

with f4:
    search_text = st.text_input(
        "Search question or AOI",
        placeholder="wheat, Cairo, flood...",
    )

filtered = runs[
    runs["application"].isin(selected_applications)
    & runs["status"].isin(selected_statuses)
    & runs["crop"].isin(selected_crops)
].copy()

if search_text.strip():
    needle = search_text.strip().lower()
    filtered = filtered[
        filtered["question"].str.lower().str.contains(
            needle,
            regex=False,
        )
        | filtered["aoi_summary"].str.lower().str.contains(
            needle,
            regex=False,
        )
    ]

st.caption(
    f"Showing {len(filtered)} of {len(runs)} logged runs."
)

successful = int(
    filtered["status"].str.lower().eq("success").sum()
)
with_aoi = int(
    filtered["aoi_summary"].ne("Not recorded").sum()
)
average_latency = filtered["latency_seconds"].mean()
average_scenes = filtered["stac_scene_count"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Visible runs", len(filtered))
m2.metric("Successful", successful)
m3.metric(
    "Average latency",
    f"{average_latency:.2f} s"
    if pd.notna(average_latency)
    else "N/A",
)
m4.metric(
    "Average STAC scenes",
    f"{average_scenes:.1f}"
    if pd.notna(average_scenes)
    else "N/A",
)

overview_tab, geography_tab, governance_tab, table_tab, inspect_tab = st.tabs(
    [
        "Overview",
        "AOI and STAC",
        "AI Governance",
        "Interactive table",
        "Inspect a run",
    ]
)

with overview_tab:
    left, right = st.columns(2)

    with left:
        st.markdown("### Response latency")

        latency_data = filtered.dropna(
            subset=["created_at", "latency_seconds"]
        )

        if latency_data.empty:
            st.info("No latency data is available for the selected filters.")
        else:
            latency_chart = (
                alt.Chart(latency_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "created_at:T",
                        title="Run time",
                    ),
                    y=alt.Y(
                        "latency_seconds:Q",
                        title="Latency (seconds)",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "created_at:T",
                            title="Run time",
                        ),
                        alt.Tooltip(
                            "latency_seconds:Q",
                            title="Latency",
                            format=".2f",
                        ),
                        alt.Tooltip(
                            "application:N",
                            title="Application",
                        ),
                        alt.Tooltip(
                            "question:N",
                            title="Question",
                        ),
                    ],
                )
                .interactive()
            )

            st.altair_chart(
                latency_chart,
                use_container_width=True,
            )

    with right:
        st.markdown("### Runs by application")

        app_counts = (
            filtered.groupby("application", dropna=False)
            .size()
            .reset_index(name="runs")
            .sort_values("runs", ascending=False)
        )

        app_chart = (
            alt.Chart(app_counts)
            .mark_bar()
            .encode(
                x=alt.X(
                    "runs:Q",
                    title="Runs",
                ),
                y=alt.Y(
                    "application:N",
                    sort="-x",
                    title=None,
                ),
                tooltip=["application:N", "runs:Q"],
            )
            .interactive()
        )

        st.altair_chart(
            app_chart,
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:
        st.markdown("### Status distribution")

        status_counts = (
            filtered.groupby("status", dropna=False)
            .size()
            .reset_index(name="runs")
        )

        status_chart = (
            alt.Chart(status_counts)
            .mark_arc(innerRadius=55)
            .encode(
                theta="runs:Q",
                color=alt.Color(
                    "status:N",
                    legend=alt.Legend(title="Status"),
                ),
                tooltip=["status:N", "runs:Q"],
            )
        )

        st.altair_chart(
            status_chart,
            use_container_width=True,
        )

    with right:
        st.markdown("### Crop distribution")

        crop_counts = (
            filtered.groupby("crop", dropna=False)
            .size()
            .reset_index(name="runs")
            .sort_values("runs", ascending=False)
        )

        crop_chart = (
            alt.Chart(crop_counts)
            .mark_bar()
            .encode(
                x=alt.X(
                    "crop:N",
                    sort="-y",
                    title=None,
                ),
                y=alt.Y(
                    "runs:Q",
                    title="Runs",
                ),
                tooltip=["crop:N", "runs:Q"],
            )
            .interactive()
        )

        st.altair_chart(
            crop_chart,
            use_container_width=True,
        )

with geography_tab:
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("### STAC scenes by run")

        scene_data = filtered.dropna(
            subset=["created_at"]
        )

        if scene_data.empty:
            st.info("No STAC scene information is available.")
        else:
            scene_chart = (
                alt.Chart(scene_data)
                .mark_circle(size=110)
                .encode(
                    x=alt.X(
                        "created_at:T",
                        title="Run time",
                    ),
                    y=alt.Y(
                        "stac_scene_count:Q",
                        title="Matching scenes",
                    ),
                    tooltip=[
                        "question:N",
                        "aoi_summary:N",
                        "stac_scene_count:Q",
                        "max_cloud_cover:Q",
                    ],
                )
                .interactive()
            )

            st.altair_chart(
                scene_chart,
                use_container_width=True,
            )

    with g2:
        st.markdown("### AOI coverage")

        st.metric("Runs with AOI", with_aoi)

        coverage_df = pd.DataFrame(
            {
                "AOI availability": [
                    "AOI recorded",
                    "Not recorded",
                ],
                "runs": [
                    with_aoi,
                    max(len(filtered) - with_aoi, 0),
                ],
            }
        )

        coverage_chart = (
            alt.Chart(coverage_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "runs:Q",
                    title="Runs",
                ),
                y=alt.Y(
                    "AOI availability:N",
                    title=None,
                ),
                tooltip=["AOI availability:N", "runs:Q"],
            )
        )

        st.altair_chart(
            coverage_chart,
            use_container_width=True,
        )

    st.markdown("### AOI context")

    aoi_table = filtered[
        [
            "created_at",
            "question",
            "aoi_summary",
            "start_date",
            "end_date",
            "max_cloud_cover",
            "stac_scene_count",
        ]
    ]

    st.dataframe(
        aoi_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "Run time",
                format="YYYY-MM-DD HH:mm",
            ),
            "question": st.column_config.TextColumn(
                "Question",
                width="large",
            ),
            "aoi_summary": st.column_config.TextColumn(
                "AOI",
                width="large",
            ),
            "max_cloud_cover": st.column_config.NumberColumn(
                "Cloud limit",
                format="%.0f%%",
            ),
            "stac_scene_count": st.column_config.NumberColumn(
                "Scenes",
                format="%d",
            ),
        },
    )


with governance_tab:
    st.markdown("### AI Governance")

    st.markdown(
        """
GeoScope treats governance as part of the operational AI lifecycle rather
than as a separate policy statement. The controls below focus on the risks
that are relevant to a public Earth-observation assistant: unsupported
conclusions, insufficient evidence, misleading temporal/geographic
interpretation, lack of traceability, and over-reliance on generated answers.
"""
    )

    evaluations = _read_governance_table(
        "generation_evaluations"
    )
    feedback = _read_governance_table(
        "user_feedback"
    )

    groundedness = _numeric_average(
        evaluations,
        "groundedness",
    )
    relevance = _numeric_average(
        evaluations,
        "relevance",
    )
    technical = _numeric_average(
        evaluations,
        "technical_correctness",
    )
    citation_quality = _numeric_average(
        evaluations,
        "citation_quality",
    )
    geographic = _numeric_average(
        evaluations,
        "geographic_relevance",
    )
    positive_rate, feedback_count = (
        _positive_feedback_rate(feedback)
    )

    g1, g2, g3 = st.columns(3)
    g1.metric(
        "Groundedness",
        _metric_value(groundedness),
        help=(
            "LLM-as-a-judge score measuring whether the answer "
            "is supported by retrieved context."
        ),
    )
    g2.metric(
        "Relevance",
        _metric_value(relevance),
    )
    g3.metric(
        "Technical correctness",
        _metric_value(technical),
    )

    g4, g5, g6 = st.columns(3)
    g4.metric(
        "Citation quality",
        _metric_value(citation_quality),
    )
    g5.metric(
        "Geographic relevance",
        _metric_value(geographic),
    )
    g6.metric(
        "Positive human feedback",
        positive_rate,
        help=(
            f"Based on {feedback_count} explicit feedback record(s)."
        ),
    )

    st.caption(
        f"Structured generation evaluations available: "
        f"{len(evaluations)} · "
        f"Human feedback records available: {len(feedback)}"
    )

    st.markdown("### Governance control mapping")

    governance_rows = [
        {
            "Dimension": "Groundedness",
            "Applicability": "High",
            "GeoScope control": (
                "RAG retrieves evidence before generation; "
                "answers are evaluated against retrieved context."
            ),
            "Evidence / metric": (
                "Retrieved sources + groundedness score"
            ),
        },
        {
            "Dimension": "Explainability",
            "Applicability": "High",
            "GeoScope control": (
                "The user can inspect the rewritten query, vector rank, "
                "final rank, vector distance, FlashRank score, and source text."
            ),
            "Evidence / metric": (
                "Visible retrieval pipeline and source provenance"
            ),
        },
        {
            "Dimension": "Transparency",
            "Applicability": "High",
            "GeoScope control": (
                "Provider/model configuration, retrieval strategy, "
                "AOI/STAC context, and system limitations are made visible."
            ),
            "Evidence / metric": (
                "Configuration shown in UI + documented limitations"
            ),
        },
        {
            "Dimension": "Human oversight",
            "Applicability": "High",
            "GeoScope control": (
                "The user can explicitly accept/reject an answer "
                "and provide a comment directly in Ask GeoAI."
            ),
            "Evidence / metric": (
                "Positive/negative feedback rate and comments"
            ),
        },
        {
            "Dimension": "Reliability / quality",
            "Applicability": "High",
            "GeoScope control": (
                "Retrieval metrics and LLM-as-a-judge evaluate "
                "answer relevance, completeness, correctness, citations, "
                "and geographic relevance."
            ),
            "Evidence / metric": (
                "Hit Rate, MRR, generation evaluation scores"
            ),
        },
        {
            "Dimension": "Traceability / auditability",
            "Applicability": "High",
            "GeoScope control": (
                "Runs are logged in DuckDB; persistent project workflows "
                "record analysis state, progress, artifacts, and events."
            ),
            "Evidence / metric": (
                "Run logs + project history + timestamps"
            ),
        },
        {
            "Dimension": "Ethical / responsible use",
            "Applicability": "Context-specific",
            "GeoScope control": (
                "GeoScope uses public Earth-observation and technical data "
                "and does not perform personal profiling. Responsible-use "
                "controls therefore focus on avoiding unsupported or "
                "misleading geospatial conclusions."
            ),
            "Evidence / metric": (
                "Public-data scope + domain-specific guardrails"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(governance_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Responsible-use controls")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
**Data and human oversight**
- ✅ Public Earth-observation / technical sources
- ✅ No personal profiling in the GeoScope use case
- ✅ Human feedback at the point of use
- ✅ Human remains responsible for accepting recommendations
"""
        )

    with c2:
        st.markdown(
            """
**Analytical safeguards**
- ✅ Evidence-backed RAG answers
- ✅ Retrieved source inspection
- ✅ AOI and geographic context are explicit
- ✅ Distinct acquisition dates checked before time-series recommendations
"""
        )

    st.info(
        "Fairness and demographic-bias metrics are not primary governance "
        "metrics for this use case because GeoScope does not make "
        "individual-level decisions or process demographic profiles. "
        "The governance focus is instead reliability, groundedness, "
        "transparency, explainability, traceability, human oversight, "
        "and domain-specific safeguards."
    )


with table_tab:
    st.markdown("### Filtered monitoring records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "Run time",
                format="YYYY-MM-DD HH:mm:ss",
            ),
            "question": st.column_config.TextColumn(
                "Question",
                width="large",
            ),
            "aoi_summary": st.column_config.TextColumn(
                "AOI description",
                width="large",
            ),
            "latency_seconds": st.column_config.NumberColumn(
                "Latency",
                format="%.2f s",
            ),
            "max_cloud_cover": st.column_config.NumberColumn(
                "Cloud limit",
                format="%.0f%%",
            ),
        },
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered logs as CSV",
        data=csv_data,
        file_name="geoscope_monitoring_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with inspect_tab:
    if filtered.empty:
        st.info("No run matches the selected filters.")
    else:
        labels = []

        for index, row in filtered.reset_index(drop=True).iterrows():
            timestamp = (
                row["created_at"].strftime("%Y-%m-%d %H:%M")
                if pd.notna(row["created_at"])
                else "Unknown time"
            )
            labels.append(
                f"{index + 1}. {timestamp} — {row['question'][:85]}"
            )

        selected_label = st.selectbox(
            "Choose a logged run",
            labels,
        )

        selected_index = labels.index(selected_label)
        selected = filtered.reset_index(drop=True).iloc[
            selected_index
        ]

        q1, q2 = st.columns(2)

        with q1:
            st.markdown("### Question context")
            st.write(f"**Question:** {selected['question']}")
            st.write(
                f"**Application:** {selected['application']}"
            )
            st.write(f"**Crop:** {selected['crop']}")
            st.write(f"**Season:** {selected['season']}")
            st.write(f"**Status:** {selected['status']}")
            st.write(
                f"**Latency:** {selected['latency_seconds']:.2f} seconds"
                if pd.notna(selected["latency_seconds"])
                else "**Latency:** Not recorded"
            )

        with q2:
            st.markdown("### Geographic context")
            st.write(f"**AOI:** {selected['aoi_summary']}")
            st.write(
                f"**STAC scenes:** {selected['stac_scene_count']}"
            )
            st.write(
                f"**Search dates:** {selected['start_date']} "
                f"→ {selected['end_date']}"
            )
            st.write(
                f"**Maximum cloud cover:** "
                f"{selected['max_cloud_cover']}"
            )
