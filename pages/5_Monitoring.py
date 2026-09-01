from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import recent_runs


st.set_page_config(
    page_title="Monitoring & AI Engineering",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# VISUAL STYLE
# =============================================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.geoscope-hero {
    padding: 1.25rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #103c4a 0%, #176b68 52%, #2a9d8f 100%);
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(16,60,74,.14);
}
.geoscope-hero h2 {
    margin: 0 0 .35rem 0;
    color: white;
}
.geoscope-hero p {
    margin: 0;
    color: #e9fffb;
    font-size: 1.02rem;
}

.metric-card {
    padding: .9rem 1rem;
    border-radius: 15px;
    border: 1px solid rgba(80,100,110,.14);
    background: linear-gradient(145deg, #ffffff, #f4faf8);
    min-height: 105px;
    box-shadow: 0 3px 12px rgba(30,60,70,.06);
}
.metric-label {
    color: #60747a;
    font-size: .82rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .03em;
}
.metric-value {
    color: #143d3a;
    font-weight: 800;
    font-size: 1.65rem;
    margin-top: .3rem;
}

.pipeline-wrap {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: .35rem;
    margin: .65rem 0 1rem 0;
}
.step-fixed {
    background: #dceeff;
    color: #135a93;
    border: 1px solid #a9d5f6;
    padding: .42rem .68rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: .85rem;
}
.step-agent {
    background: #eee3ff;
    color: #673ab7;
    border: 1px solid #d3baf8;
    padding: .42rem .68rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: .85rem;
}
.step-rag {
    background: #def7ed;
    color: #14755d;
    border: 1px solid #abe8d6;
    padding: .42rem .68rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: .85rem;
}
.arrow {
    color: #74888e;
    font-weight: 800;
}

.badge-success {
    display:inline-block;
    padding:.25rem .55rem;
    border-radius:999px;
    background:#dff6e8;
    color:#177245;
    font-weight:750;
}
.badge-fail {
    display:inline-block;
    padding:.25rem .55rem;
    border-radius:999px;
    background:#ffe2e2;
    color:#a52d2d;
    font-weight:750;
}
.badge-neutral {
    display:inline-block;
    padding:.25rem .55rem;
    border-radius:999px;
    background:#edf0f3;
    color:#58646b;
    font-weight:750;
}
.framework-fixed {
    border-left: 5px solid #268bd2;
    padding: .75rem 1rem;
    background: #f3f9ff;
    border-radius: 10px;
}
.framework-agent {
    border-left: 5px solid #8b5cf6;
    padding: .75rem 1rem;
    background: #faf7ff;
    border-radius: 10px;
}
.framework-rag {
    border-left: 5px solid #2a9d8f;
    padding: .75rem 1rem;
    background: #f3fbf8;
    border-radius: 10px;
}
.small-note {
    color:#6e7d82;
    font-size:.86rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

TEXT_DEFAULT = "Not recorded"


def clean_text(value: Any) -> str:
    if value is None:
        return TEXT_DEFAULT
    try:
        if pd.isna(value):
            return TEXT_DEFAULT
    except Exception:
        pass
    value = str(value).strip()
    return value if value else TEXT_DEFAULT


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("", TEXT_DEFAULT)
    )


def infer_framework(row: pd.Series) -> tuple[str, str, str]:
    """
    Return (framework, execution_mode, use_case).

    Newer logs may have explicit framework/execution columns. Older Page 9
    records historically used the 'application' field for labels such as
    'LangChain fixed ...' and 'LangGraph agent ...'. We separate those here
    so use case and framework are not mixed in the dashboard.
    """
    explicit_framework = clean_text(row.get("framework", ""))
    explicit_mode = clean_text(row.get("execution_mode", ""))
    application = clean_text(row.get("application", ""))

    app_lower = application.lower()
    fw_lower = explicit_framework.lower()

    if explicit_framework != TEXT_DEFAULT:
        if "langgraph" in fw_lower:
            return "LangGraph", (
                explicit_mode if explicit_mode != TEXT_DEFAULT else "Agentic"
            ), application
        if "langchain" in fw_lower:
            return "LangChain", (
                explicit_mode if explicit_mode != TEXT_DEFAULT else "Fixed"
            ), application
        if "application" in fw_lower or "rag" in fw_lower:
            return "Application RAG", (
                explicit_mode if explicit_mode != TEXT_DEFAULT else "Fixed RAG"
            ), application

    if app_lower.startswith("langgraph") or "langgraph agent" in app_lower:
        return "LangGraph", "Agentic", TEXT_DEFAULT

    if app_lower.startswith("langchain") or "langchain fixed" in app_lower:
        return "LangChain", "Fixed", TEXT_DEFAULT

    # Ordinary Ask GeoAI runs are application/use-case runs. Unless the
    # framework was explicitly logged we avoid inventing one.
    return (
        "Not recorded",
        explicit_mode if explicit_mode != TEXT_DEFAULT else TEXT_DEFAULT,
        application,
    )


def normalize_approach(value: Any) -> str:
    value = clean_text(value)
    mapping = {
        "vector": "Vector",
        "rewrite": "Rewrite",
        "rerank": "Rerank",
        "rewrite_rerank": "Rewrite + Rerank",
    }
    return mapping.get(value.lower(), value)


def pipeline_steps(
    framework: str,
    retrieval_approach: str = TEXT_DEFAULT,
) -> tuple[list[str], str]:
    if framework == "LangChain":
        return (
            [
                "Question",
                "Rewrite",
                "Retrieve",
                "Rerank",
                "Build context",
                "Generate",
            ],
            "fixed",
        )

    if framework == "LangGraph":
        return (
            [
                "Question",
                "Planner",
                "Choose bounded tool",
                "Observe",
                "Planner",
                "Next action",
                "Final answer",
            ],
            "agent",
        )

    if framework == "Application RAG":
        approach = retrieval_approach.lower()

        if approach == "vector":
            return ["Question", "Vector retrieval", "Context", "Generate"], "rag"
        if approach == "rewrite":
            return [
                "Question", "Rewrite", "Vector retrieval", "Context", "Generate"
            ], "rag"
        if approach == "rerank":
            return [
                "Question", "Vector candidates", "FlashRank",
                "Context", "Generate"
            ], "rag"
        if approach in {"rewrite + rerank", "rewrite_rerank"}:
            return [
                "Question", "Rewrite", "Vector candidates", "FlashRank",
                "Context", "Generate"
            ], "rag"

        return [
            "Question", "Retrieve knowledge", "Build context", "Generate answer"
        ], "rag"

    approach = retrieval_approach.lower()

    if approach == "vector":
        return ["Question", "Vector retrieval", "Context", "Generate"], "rag"

    if approach == "rewrite":
        return [
            "Question",
            "Rewrite",
            "Vector retrieval",
            "Context",
            "Generate",
        ], "rag"

    if approach == "rerank":
        return [
            "Question",
            "Vector candidates",
            "FlashRank",
            "Context",
            "Generate",
        ], "rag"

    if approach in {"rewrite + rerank", "rewrite_rerank"}:
        return [
            "Question",
            "Rewrite",
            "Vector candidates",
            "FlashRank",
            "Context",
            "Generate",
        ], "rag"

    return [
        "Question",
        "Retrieve knowledge",
        "Build context",
        "Generate answer",
    ], "rag"


def render_pipeline(steps: list[str], kind: str) -> None:
    css_class = {
        "fixed": "step-fixed",
        "agent": "step-agent",
        "rag": "step-rag",
    }.get(kind, "step-rag")

    parts: list[str] = ['<div class="pipeline-wrap">']
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(
            f'<span class="{css_class}">{step}</span>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def metric_card(label: str, value: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        "</div>"
    )


def status_badge(status: str) -> str:
    status_clean = clean_text(status)
    if status_clean.lower() == "success":
        return f'<span class="badge-success">✓ {status_clean}</span>'
    if status_clean.lower() in {"failed", "failure", "error"}:
        return f'<span class="badge-fail">✕ {status_clean}</span>'
    return f'<span class="badge-neutral">{status_clean}</span>'


def safe_optional(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row.index:
            value = clean_text(row.get(name))
            if value != TEXT_DEFAULT:
                return value
    return TEXT_DEFAULT


def display_jsonish(value: Any) -> None:
    if value is None:
        st.caption(TEXT_DEFAULT)
        return
    if isinstance(value, (dict, list)):
        st.json(value)
        return
    text = clean_text(value)
    if text == TEXT_DEFAULT:
        st.caption(TEXT_DEFAULT)
        return
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            st.json(parsed)
            return
    except Exception:
        pass
    st.code(text)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
<div class="geoscope-hero">
  <h2>📊 Monitoring & AI Run Explorer</h2>
  <p>
    Move from high-level system health to a single run, then inspect the
    use case, orchestration framework, pipeline steps, context and quality signals.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
The goal of this page is no longer only to answer **"How many runs happened?"**.
It should help answer:

> **What produced this answer, which pipeline was used, which prompt/context
> configuration was supplied, and where should we investigate if quality drops?**

### How is this observability dashboard produced?

GeoScope separates **AI execution**, **logging/storage**, **quality evaluation**
and **visualization**:

```text
                    AI EXECUTION
                         │
        ┌────────────────┼────────────────┐
        │                │                │
 Application RAG     LangChain        LangGraph
 fixed flow          fixed chain      bounded agent
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                  Run metadata
        question · model · prompt · retrieval
        top-k · chunks · context size · trace
                         ↓
                DuckDB run store
                  logs/runs.duckdb
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
        LLM-as-a-judge       Human feedback
              │                     │
              └──────────┬──────────┘
                         ↓
                    dlt logging
          evaluation / feedback event tables
                         ↓
              DuckDB monitoring store
          logs/geoscope_monitoring.duckdb
                         ↓
                recent_runs() joins
             everything through run_id
                         ↓
                    STREAMLIT
              Monitoring & Run Explorer
```

**Technology roles**

- **Application RAG / LangChain / LangGraph** — produce the AI execution.
- **DuckDB** — persists local run and observability records.
- **dlt** — appends structured evaluation and human-feedback events while
  allowing those event schemas to evolve.
- **LLM-as-a-judge** — scores answer quality independently.
- **Streamlit** — renders the monitoring, drill-down and comparison interface.

This separation matters: **LangChain/LangGraph do not perform the monitoring**.
They produce executions that GeoScope records and later analyzes.

Older GeoScope runs were created before all observability fields existed.
Missing historical values are therefore shown explicitly as **Not recorded**
rather than invented.
"""
)


# =============================================================================
# LOAD DATA
# =============================================================================

try:
    runs = recent_runs(limit=300)
except ModuleNotFoundError as exc:
    st.error(
        "A required package is missing. Activate the project virtual "
        "environment and run: python -m pip install duckdb"
    )
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not load monitoring data: {exc}")
    st.stop()

if not isinstance(runs, pd.DataFrame):
    runs = pd.DataFrame(runs)

base_expected = {
    "created_at": pd.NaT,
    "run_id": "",
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

# Future-ready fields: they are displayed if the logging layer starts returning
# them, but are not falsely populated for old runs.
optional_ai_fields = {
    "framework": "",
    "execution_mode": "",
    "retrieval_approach": "",
    "approach": "",
    "prompt_id": "",
    "prompt_version": "",
    "model": "",
    "top_k": None,
    "candidate_k": None,
    "context_tokens": None,
    "context_characters": None,
    "chunk_count": None,
    "sources": None,
    "retrieved_chunks": None,
    "rewritten_query": "",
    "retrieval_query": "",
    "answer": "",
    "trace": None,
    "trajectory": None,
    "judge_verdict": "",
    "groundedness": None,
    "relevance": None,
    "technical_correctness": None,
    "human_feedback": "",
}

for col, default in {**base_expected, **optional_ai_fields}.items():
    if col not in runs.columns:
        runs[col] = default

if runs.empty:
    st.info(
        "No runs have been logged yet. Ask a question from Page 3 or run "
        "Page 9, then return here."
    )
    st.stop()

for col in [
    "question",
    "application",
    "crop",
    "season",
    "aoi_summary",
    "status",
    "framework",
    "execution_mode",
    "retrieval_approach",
    "approach",
    "prompt_id",
    "prompt_version",
    "model",
    "rewritten_query",
    "retrieval_query",
    "answer",
    "judge_verdict",
    "human_feedback",
]:
    runs[col] = clean_text_series(runs[col])

runs["latency_seconds"] = pd.to_numeric(
    runs["latency_seconds"], errors="coerce"
)
runs["stac_scene_count"] = pd.to_numeric(
    runs["stac_scene_count"], errors="coerce"
).fillna(0).astype(int)
runs["created_at"] = pd.to_datetime(
    runs["created_at"], errors="coerce", utc=True
)

# Normalize use case vs orchestration framework.
inferred = runs.apply(infer_framework, axis=1, result_type="expand")
inferred.columns = ["framework_normalized", "execution_normalized", "use_case"]
runs = pd.concat([runs, inferred], axis=1)

runs["retrieval_normalized"] = runs.apply(
    lambda row: normalize_approach(
        row["retrieval_approach"]
        if row["retrieval_approach"] != TEXT_DEFAULT
        else row["approach"]
    ),
    axis=1,
)

st.markdown("### 🧹 Run visibility")

show_not_recorded = st.checkbox(
    "Show historical / incomplete runs (`Not recorded`)",
    value=False,
    help=(
        "Turn this on to include older runs created before the new observability "
        "fields were logged. No data is deleted; this only changes what is shown."
    ),
)

all_runs = runs.copy()

instrumented_mask = (
    runs["framework_normalized"].ne("Not recorded")
    & runs["prompt_id"].ne("Not recorded")
    & runs["retrieval_normalized"].ne("Not recorded")
    & pd.to_numeric(runs["chunk_count"], errors="coerce").notna()
)

if not show_not_recorded:
    runs = runs[instrumented_mask].copy()

    st.caption(
        f"Showing **{len(runs)}** fully instrumented runs out of "
        f"**{len(all_runs)}** stored runs. "
        "Enable the checkbox to include historical / incomplete runs."
    )

    if runs.empty:
        st.warning(
            "No fully instrumented runs are available yet. "
            "Enable the checkbox to view historical runs, or create a new run "
            "from Page 3 or Page 6."
        )
else:
    incomplete_count = int((~instrumented_mask).sum())
    st.caption(
        f"Showing all **{len(runs)}** stored runs, including "
        f"**{incomplete_count}** historical / incomplete runs."
    )


# =============================================================================
# TABS
# =============================================================================

overview_tab, explorer_tab, framework_tab, context_tab, runtime_tab = st.tabs(
    [
        "✨ Overview",
        "🔎 Run Explorer",
        "🔀 Framework & Pipeline",
        "🧠 Prompt & Context",
        "⚙️ Runtime",
    ]
)


# =============================================================================
# OVERVIEW
# =============================================================================

with overview_tab:
    total_runs = len(runs)
    successes = int(runs["status"].str.lower().eq("success").sum())
    failures = int(
        runs["status"].str.lower().isin(["failed", "failure", "error"]).sum()
    )
    avg_latency = runs["latency_seconds"].mean()
    success_rate = (
        100.0 * successes / total_runs if total_runs else 0.0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        metric_card("Runs", str(total_runs)),
        unsafe_allow_html=True,
    )
    m2.markdown(
        metric_card("Success rate", f"{success_rate:.1f}%"),
        unsafe_allow_html=True,
    )
    m3.markdown(
        metric_card("Failures", str(failures)),
        unsafe_allow_html=True,
    )
    m4.markdown(
        metric_card(
            "Average latency",
            f"{avg_latency:.1f} s" if pd.notna(avg_latency) else "N/A",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 🛰️ Use case vs 🔀 framework")
    st.caption(
        "These are intentionally separated. Crop monitoring / Urban heat are "
        "EO use cases. LangChain / LangGraph are orchestration frameworks."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Runs by EO use case")
        use_case_counts = (
            runs["use_case"]
            .value_counts()
            .rename_axis("EO use case")
            .to_frame("Runs")
        )
        st.bar_chart(use_case_counts, horizontal=True)

    with right:
        st.markdown("#### Runs by orchestration framework")
        framework_counts = (
            runs["framework_normalized"]
            .value_counts()
            .rename_axis("Framework")
            .to_frame("Runs")
        )
        st.bar_chart(framework_counts, horizontal=True)

    st.info(
        "ℹ️ `Not recorded` is expected for older runs created before framework "
        "metadata was logged explicitly. Page 9 legacy runs are recognized when "
        "their old application label contains LangChain or LangGraph."
    )

    st.markdown("### Recent runs")
    overview_cols = [
        "created_at",
        "use_case",
        "framework_normalized",
        "execution_normalized",
        "question",
        "latency_seconds",
        "status",
    ]
    st.dataframe(
        runs[overview_cols].head(30),
        use_container_width=True,
        hide_index=True,
        column_config={
            "use_case": "EO use case",
            "framework_normalized": "Framework",
            "execution_normalized": "Execution",
            "latency_seconds": st.column_config.NumberColumn(
                "Latency (s)",
                format="%.2f",
            ),
        },
    )


# =============================================================================
# RUN EXPLORER — drill down
# =============================================================================

with explorer_tab:
    st.markdown("## 🔎 Drill down from use case to one run")

    st.markdown(
        """
```text
Use case
   ↓
Framework
   ↓
Question / run
   ↓
Pipeline steps
   ↓
Prompt + context + retrieval
   ↓
Answer / judge / feedback
```
"""
    )

    f1, f2 = st.columns(2)

    use_case_values = ["All"] + sorted(
        runs["use_case"].dropna().astype(str).unique().tolist()
    )
    with f1:
        selected_use_case = st.selectbox(
            "1 · EO use case",
            use_case_values,
            key="monitor_use_case",
        )

    filtered_runs = runs.copy()
    if selected_use_case != "All":
        filtered_runs = filtered_runs[
            filtered_runs["use_case"] == selected_use_case
        ]

    framework_values = ["All"] + sorted(
        filtered_runs["framework_normalized"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    with f2:
        selected_framework = st.selectbox(
            "2 · Framework",
            framework_values,
            key="monitor_framework",
        )

    if selected_framework != "All":
        filtered_runs = filtered_runs[
            filtered_runs["framework_normalized"] == selected_framework
        ]

    st.caption(f"Matching runs: **{len(filtered_runs)}**")

    if filtered_runs.empty:
        st.warning("No logged run matches this selection.")
    else:
        run_options: dict[str, int] = {}

        for pos, (_, row) in enumerate(filtered_runs.iterrows()):
            timestamp = (
                row["created_at"].strftime("%Y-%m-%d %H:%M")
                if pd.notna(row["created_at"])
                else "unknown time"
            )
            question = clean_text(row["question"])
            fw = row["framework_normalized"]
            label = (
                f"{timestamp} · {fw} · "
                f"{question[:100]}"
            )
            run_options[label] = pos

        selected_run_label = st.selectbox(
            "3 · Select a question / run",
            list(run_options.keys()),
            key="monitor_run",
        )

        row = filtered_runs.iloc[run_options[selected_run_label]]

        top1, top2, top3, top4 = st.columns(4)
        top1.markdown(
            metric_card("Use case", clean_text(row["use_case"])),
            unsafe_allow_html=True,
        )
        top2.markdown(
            metric_card(
                "Framework",
                clean_text(row["framework_normalized"]),
            ),
            unsafe_allow_html=True,
        )
        top3.markdown(
            metric_card(
                "Execution",
                clean_text(row["execution_normalized"]),
            ),
            unsafe_allow_html=True,
        )
        top4.markdown(
            metric_card(
                "Latency",
                (
                    f"{row['latency_seconds']:.1f} s"
                    if pd.notna(row["latency_seconds"])
                    else "N/A"
                ),
            ),
            unsafe_allow_html=True,
        )

        st.markdown("### 🧭 Pipeline used")

        steps, kind = pipeline_steps(
            row["framework_normalized"],
            row["retrieval_normalized"],
        )
        render_pipeline(steps, kind)

        if row["framework_normalized"] == "LangChain":
            st.markdown(
                """
<div class="framework-fixed">
<b>LangChain fixed pipeline</b><br>
The application controls the sequence. The same known stages are executed
in a predictable order.
</div>
""",
                unsafe_allow_html=True,
            )

        elif row["framework_normalized"] == "LangGraph":
            st.markdown(
                """
<div class="framework-agent">
<b>LangGraph agentic workflow</b><br>
The planner chooses the next permitted action from bounded tools according
to the question and observed state.
</div>
""",
                unsafe_allow_html=True,
            )

        elif row["framework_normalized"] == "Application RAG":
            st.markdown(
                """
<div class="framework-rag">
<b>GeoScope application RAG pipeline</b><br>
This run was executed by the standard Page 3 application flow. It is a
deterministic RAG pipeline and should not be labelled LangChain unless the
LangChain implementation was actually invoked.
</div>
""",
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
<div class="framework-rag">
<b>RAG / framework metadata not recorded</b><br>
The run is visible, but the historical log does not contain enough metadata
to claim whether LangChain or LangGraph orchestrated it.
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("### 📋 Run details")
        d1, d2, d3 = st.columns([1.3, 1, 1])

        with d1:
            st.markdown("#### Question")
            st.write(clean_text(row["question"]))
            st.markdown(
                status_badge(row["status"]),
                unsafe_allow_html=True,
            )

        with d2:
            st.markdown("#### Retrieval")
            st.write(
                f"**Approach:** "
                f"{clean_text(row['retrieval_normalized'])}"
            )
            st.write(
                f"**Top-k:** {clean_text(row['top_k'])}"
            )
            st.write(
                f"**Candidate-k:** {clean_text(row['candidate_k'])}"
            )

        with d3:
            st.markdown("#### Geographic context")
            st.write(
                f"**AOI:** {clean_text(row['aoi_summary'])}"
            )
            st.write(
                f"**STAC scenes:** {row['stac_scene_count']}"
            )
            st.write(
                f"**Dates:** {clean_text(row['start_date'])} "
                f"→ {clean_text(row['end_date'])}"
            )

        st.markdown("### 🧠 Prompt & context for this run")

        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("#### Prompt metadata")
            st.write(
                f"**Prompt ID:** {clean_text(row['prompt_id'])}"
            )
            st.write(
                f"**Version:** {clean_text(row['prompt_version'])}"
            )
            st.write(
                f"**Model:** {clean_text(row['model'])}"
            )

            rewritten = safe_optional(
                row,
                "rewritten_query",
                "retrieval_query",
            )
            with st.expander("View rewritten / retrieval query"):
                if rewritten == TEXT_DEFAULT:
                    st.caption(
                        "Not recorded for this historical run."
                    )
                else:
                    st.code(rewritten)

        with pc2:
            st.markdown("#### Context metadata")
            st.write(
                f"**Chunks:** {clean_text(row['chunk_count'])}"
            )
            st.write(
                f"**Estimated context tokens:** "
                f"{clean_text(row['context_tokens'])}"
            )
            st.write(
                f"**Context characters:** "
                f"{clean_text(row['context_characters'])}"
            )

            source_value = (
                row["retrieved_chunks"]
                if clean_text(row["retrieved_chunks"]) != TEXT_DEFAULT
                else row["sources"]
            )
            with st.expander("View logged chunks / sources"):
                display_jsonish(source_value)

        st.markdown("### ✅ Output & quality")

        q1, q2 = st.columns(2)

        with q1:
            st.markdown("#### Answer")
            answer = clean_text(row["answer"])
            if answer == TEXT_DEFAULT:
                st.caption(
                    "Answer text was not returned by the current "
                    "`recent_runs()` monitoring query."
                )
            else:
                st.write(answer)

        with q2:
            st.markdown("#### Evaluation")
            verdict = clean_text(row["judge_verdict"])
            st.write(f"**Judge verdict:** {verdict}")
            st.write(
                f"**Groundedness:** "
                f"{clean_text(row['groundedness'])}"
            )
            st.write(
                f"**Relevance:** {clean_text(row['relevance'])}"
            )
            st.write(
                f"**Technical correctness:** "
                f"{clean_text(row['technical_correctness'])}"
            )
            st.write(
                f"**Human feedback:** "
                f"{clean_text(row['human_feedback'])}"
            )

        trajectory_value = (
            row["trajectory"]
            if clean_text(row["trajectory"]) != TEXT_DEFAULT
            else row["trace"]
        )

        with st.expander("🔀 View actual logged framework trajectory / trace"):
            if clean_text(trajectory_value) == TEXT_DEFAULT:
                st.caption(
                    "No per-step trajectory was recorded for this run. "
                    "The pipeline shown above is the known GeoScope pipeline "
                    "definition for that framework, not a fabricated execution trace."
                )
            else:
                display_jsonish(trajectory_value)


# =============================================================================
# FRAMEWORK / PIPELINE
# =============================================================================

with framework_tab:
    st.markdown("## 🔀 Framework & pipeline")
    st.caption(
        "GeoScope currently contains three execution patterns. They must remain "
        "separate in Monitoring so the project never claims a framework was used "
        "when it was not."
    )

    c_app, c_lc, c_lg = st.columns(3)

    with c_app:
        st.markdown("### 🟢 Application RAG")
        render_pipeline(
            ["Question", "Rewrite", "Retrieve", "Rerank", "Context", "Generate"],
            "rag",
        )
        st.write(
            "The standard Ask GeoAI flow. The application code determines the "
            "sequence directly."
        )

    with c_lc:
        st.markdown("### 🔵 LangChain — fixed")
        render_pipeline(
            [
                "Question", "Rewrite", "Retrieve",
                "Rerank", "Build context", "Generate"
            ],
            "fixed",
        )
        st.write(
            "The same predictable concept expressed through LangChain "
            "Runnables for framework-based orchestration."
        )

    with c_lg:
        st.markdown("### 🟣 LangGraph — agentic")
        render_pipeline(
            [
                "Question", "Planner", "Tool",
                "Observe", "Planner", "Answer"
            ],
            "agent",
        )
        st.write(
            "A bounded planner can choose the next permitted action according "
            "to the current state and tool observations."
        )

    st.success(
        "Key principle: **not every AI workflow needs an agent**. "
        "Framework choice is an engineering decision that should be evaluated "
        "against quality, latency and complexity."
    )

    st.markdown("### Inspect runs by execution framework")

    available_frameworks = [
        value
        for value in ["Application RAG", "LangChain", "LangGraph", "Not recorded"]
        if value in set(runs["framework_normalized"].astype(str))
    ]

    if not available_frameworks:
        st.info("No framework metadata is available yet.")
    else:
        framework_choice = st.radio(
            "Framework",
            available_frameworks,
            horizontal=True,
            key="framework_inspection",
        )

        framework_subset = runs[
            runs["framework_normalized"] == framework_choice
        ]

        st.dataframe(
            framework_subset[
                [
                    "created_at",
                    "use_case",
                    "question",
                    "retrieval_normalized",
                    "latency_seconds",
                    "status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "use_case": "EO use case",
                "retrieval_normalized": "Retrieval",
                "latency_seconds": st.column_config.NumberColumn(
                    "Latency (s)",
                    format="%.2f",
                ),
            },
        )


# =============================================================================
# PROMPT & CONTEXT — current state + future-ready logged fields
# =============================================================================

with context_tab:
    st.markdown("## 🧠 Prompt & Context Engineering")
    st.markdown(
        """
This tab prepares GeoScope for the next experiment layer:

**Prompt version + retrieved chunks + context size + framework → answer → judge**

The current historical run log does not yet contain all of those fields.
They are shown whenever available and otherwise remain explicitly
**Not recorded**.
"""
    )

    with st.expander("🔍 Where do these values come from?", expanded=True):
        st.markdown(
            """
For every **new Page 3 run**, GeoScope now records:

```text
question
use case / AOI / STAC context
model
prompt ID + version
retrieval approach
original + rewritten query
top-k + candidate-k
number of final chunks
exact context characters
estimated context tokens
answer
latency + status
known pipeline stages
```

Page 4 then writes the **LLM judge scores and verdict** through dlt, while
Page 3 writes **human feedback** through dlt. `recent_runs()` joins those event
records back to the original execution using `run_id`.

Historical runs remain incomplete because those fields did not exist when the
runs were created.
"""
        )

    st.markdown("### What should eventually be compared?")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🟢 Prompt")
        st.write("Prompt ID / version")
        st.write("Fixed vs planner prompt")
        st.write("Model")
    with c2:
        st.markdown("#### 🟠 Context")
        st.write("Number of chunks")
        st.write("Top-k / candidate-k")
        st.write("Estimated context tokens / exact characters")
    with c3:
        st.markdown("#### 🟣 Quality")
        st.write("Groundedness")
        st.write("Relevance")
        st.write("Judge verdict")

    st.markdown("### Logged prompt / context fields")

    prompt_view = runs[
        [
            "created_at",
            "question",
            "framework_normalized",
            "prompt_id",
            "prompt_version",
            "retrieval_normalized",
            "top_k",
            "candidate_k",
            "chunk_count",
            "context_tokens",
            "judge_verdict",
        ]
    ].copy()

    st.dataframe(
        prompt_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "framework_normalized": "Framework",
            "retrieval_normalized": "Retrieval",
        },
    )

    st.info(
        "Next engineering step: extend `log_run()` / monitoring storage so "
        "these prompt and context fields are persisted for every new run. "
        "Then this tab can compare prompt versions and context sizes against "
        "judge quality rather than only displaying metadata."
    )


# =============================================================================
# RUNTIME
# =============================================================================

with runtime_tab:
    st.markdown("## ⚙️ Operational monitoring")
    st.caption(
        "Runtime metrics remain useful, but they are separated from AI quality "
        "and orchestration so the dashboard is easier to interpret."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Latency over time")
        latency_data = (
            runs.dropna(
                subset=["created_at", "latency_seconds"]
            )
            .sort_values("created_at")
            .set_index("created_at")[["latency_seconds"]]
        )
        if latency_data.empty:
            st.info("No valid latency data.")
        else:
            st.line_chart(latency_data)

    with right:
        st.markdown("### Status distribution")
        status_counts = (
            runs["status"]
            .value_counts()
            .rename_axis("Status")
            .to_frame("Runs")
        )
        st.bar_chart(status_counts, horizontal=True)

    st.markdown("### Run log")
    runtime_cols = [
        "created_at",
        "question",
        "use_case",
        "framework_normalized",
        "latency_seconds",
        "status",
        "stac_scene_count",
    ]

    st.dataframe(
        runs[runtime_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "use_case": "EO use case",
            "framework_normalized": "Framework",
            "latency_seconds": st.column_config.NumberColumn(
                "Latency (s)",
                format="%.2f",
            ),
        },
    )


st.divider()
st.caption(
    "GeoScope monitoring principle: separate domain use case, orchestration "
    "framework, retrieval/context configuration, quality evaluation, and "
    "runtime behavior."
)
