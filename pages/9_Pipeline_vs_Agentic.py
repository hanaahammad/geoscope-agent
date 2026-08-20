from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui import apply_global_style

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic_geoscope import run_agentic_workflow
from src.langchain_pipeline import run_standard_pipeline
from src.monitoring import log_run

st.set_page_config(
    page_title="Pipeline vs Agentic",
    page_icon="🧭",
    layout="wide",
)
apply_global_style()


def render_flow(steps, flow_class="fixed-flow"):
    """Render a lightweight execution flow directly in Streamlit."""
    parts = []

    for index, (title, subtitle) in enumerate(steps):
        parts.append(
            f'<div class="flow-step">'
            f'<div class="flow-title">{title}</div>'
            f'<div class="flow-subtitle">{subtitle}</div>'
            f'</div>'
        )

        if index < len(steps) - 1:
            parts.append(
                '<div class="flow-arrow">→</div>'
            )

    html = f"""
<style>
.geoscope-flow {{
    display: flex;
    align-items: stretch;
    gap: 7px;
    flex-wrap: wrap;
    margin: 0.5rem 0 1rem 0;
}}
.geoscope-flow .flow-step {{
    flex: 1 1 90px;
    min-width: 90px;
    padding: 11px 9px;
    border: 1px solid rgba(86, 108, 91, 0.30);
    border-radius: 10px;
    background: rgba(224, 235, 225, 0.72);
}}
.geoscope-flow.agentic-flow .flow-step {{
    background: rgba(238, 231, 214, 0.78);
    border-color: rgba(126, 106, 76, 0.30);
}}
.geoscope-flow .flow-title {{
    font-size: 0.88rem;
    font-weight: 700;
    margin-bottom: 3px;
}}
.geoscope-flow .flow-subtitle {{
    font-size: 0.74rem;
    line-height: 1.25;
    opacity: 0.78;
}}
.geoscope-flow .flow-arrow {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 16px;
    font-size: 1.25rem;
    font-weight: 700;
    opacity: 0.55;
}}
@media (max-width: 850px) {{
    .geoscope-flow .flow-arrow {{
        flex-basis: 100%;
        height: 10px;
        transform: rotate(90deg);
    }}
}}
</style>
<div class="geoscope-flow {flow_class}">
{''.join(parts)}
</div>
"""

    st.html(html)

st.title("🧭 Standard Pipeline vs Agentic GeoScope")
st.caption(
    "A side-by-side implementation showing when a deterministic LangChain "
    "pipeline is enough and when an agentic LangGraph workflow adds value."
)

st.markdown(
    """
| Standard RAG pipeline | Agentic workflow |
|---|---|
| Fixed sequence chosen by the developer | Next action selected by an LLM planner |
| Rewrite → retrieve → rerank → generate | Plan → call a tool → observe → plan again |
| Predictable and easy to evaluate | Can branch according to available AOI/STAC evidence |
| Best when the required steps are known | Useful when the path depends on the question and current state |

**GeoScope principle:** not every request should be agentic. A simple technical
question can use the standard RAG pipeline. A request that asks GeoScope to
inspect geographic context, check imagery availability, and then recommend an
analysis can benefit from tool-using agentic orchestration.
"""
)

st.markdown("### Execution anatomy")

fixed_view, agent_view = st.columns(2)

with fixed_view:
    st.markdown("#### Fixed LangChain pipeline")
    st.caption(
        "The application defines the sequence. Every request follows "
        "the same reproducible chain."
    )
    render_flow(
        [
            ("Question", "User input"),
            ("Rewrite", "Query transformation"),
            ("Retrieve", "Chroma candidates"),
            ("Rerank", "FlashRank"),
            ("Context", "Evidence + geo context"),
            ("Generate", "Grounded answer"),
        ],
        "fixed-flow",
    )

with agent_view:
    st.markdown("#### Agentic LangGraph workflow")
    st.caption(
        "The LLM planner selects the next permitted action from the "
        "current question and GeoScope state."
    )
    render_flow(
        [
            ("Question", "User goal"),
            ("Planner", "Choose action"),
            ("Tool", "Inspect / STAC / RAG"),
            ("Observe", "Read tool result"),
            ("Planner", "Choose next step"),
            ("Answer", "Stop when sufficient"),
        ],
        "agentic-flow",
    )

st.info(
    "**What LangChain adds:** a structured, composable and reproducible "
    "way to connect query rewriting, retrieval, reranking, context building "
    "and generation. It does not make the LLM more accurate by itself.\n\n"
    "**What LangGraph adds:** stateful orchestration and conditional routing. "
    "The planner can select the next action from a bounded set of tools "
    "instead of always following the same sequence."
)

left, right = st.columns([0.7, 1.3])

with left:
    st.markdown("### Test options")
    st.caption(
        "Run one approach independently, or use **Compare both** to execute "
        "the same question through both paths and highlight the actions."
    )
    top_k = st.slider("Final sources", 1, 8, 5)
    candidate_k = st.slider("Candidates before reranking", 5, 30, 15)

with right:
    example_questions = {
        "1 — Simple knowledge question":
            "What is NDVI and which Sentinel-2 bands are used?",
        "2 — Knowledge synthesis":
            "How can Sentinel-2 and Landsat be combined for long-term crop monitoring?",
        "3 — Context-aware analysis":
            "Check whether the current AOI has suitable Sentinel-2 scenes and tell me if NDVI analysis is possible.",
        "4 — Decision requiring context":
            "What should I verify before recommending a time-series analysis for this AOI?",
        "Custom question":
            "",
    }

    question_choice = st.selectbox(
        "Example question",
        list(example_questions.keys()),
        help=(
            "Questions 1–2 are usually well suited to a fixed RAG pipeline. "
            "Questions 3–4 better illustrate where agentic orchestration "
            "can inspect context and choose tools."
        ),
    )

    question = st.text_area(
        "Question",
        value=example_questions[question_choice],
        height=140,
        placeholder="Enter your own GeoScope question...",
    )

    st.caption(
        "Tip: run Question 1 with the fixed pipeline, then Question 3 or 4 "
        "with the agentic workflow. Use Compare both for the clearest demo."
    )

scenes = st.session_state.get("stac_scenes", [])
aoi_geojson = st.session_state.get("aoi_geojson")
aoi_summary = st.session_state.get("aoi_summary", "No AOI selected.")
unique_dates = sorted({s.get("date") for s in scenes if s.get("date")})

c1, c2, c3 = st.columns(3)
c1.metric("AOI available", "Yes" if aoi_geojson else "No")
c2.metric("STAC items", len(scenes))
c3.metric("Distinct dates", len(unique_dates))


def _run_standard():
    started = time.perf_counter()
    run_id = str(uuid.uuid4())

    geo_context = (
        f"AOI: {aoi_summary}\n"
        f"STAC scene items: {len(scenes)}\n"
        f"Distinct acquisition dates: {len(unique_dates)}\n"
        f"Time-series possible: {'yes' if len(unique_dates) >= 2 else 'no'}"
    )

    with st.spinner("Running deterministic LangChain pipeline..."):
        result = run_standard_pipeline(
            question,
            geo_context=geo_context,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    elapsed = time.perf_counter() - started

    output = {
        "engine": "LangChain fixed pipeline",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "latency": elapsed,
    }

    log_run(
        run_id=run_id,
        question=question,
        answer=output["answer"],
        sources=output["sources"],
        latency_seconds=elapsed,
        status="success",
        application=output["engine"],
        aoi_summary=aoi_summary,
        aoi_geojson=aoi_geojson,
        stac_scene_count=len(scenes),
        start_date=str(st.session_state.get("start_date", "")),
        end_date=str(st.session_state.get("end_date", "")),
        max_cloud_cover=st.session_state.get("max_cloud_cover"),
    )

    return output


def _run_agentic():
    started = time.perf_counter()
    run_id = str(uuid.uuid4())

    with st.spinner("Agent is planning and using tools..."):
        result = run_agentic_workflow(
            question,
            aoi_geojson=aoi_geojson,
            aoi_summary=aoi_summary,
            scenes=scenes,
            start_date=str(st.session_state.get("start_date", "")),
            end_date=str(st.session_state.get("end_date", "")),
            max_cloud_cover=float(
                st.session_state.get("max_cloud_cover", 40.0) or 40.0
            ),
            top_k=top_k,
            candidate_k=candidate_k,
        )

    elapsed = time.perf_counter() - started

    output = {
        "engine": "LangGraph agentic workflow",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "latency": elapsed,
    }

    log_run(
        run_id=run_id,
        question=question,
        answer=output["answer"],
        sources=output["sources"],
        latency_seconds=elapsed,
        status="success",
        application=output["engine"],
        aoi_summary=aoi_summary,
        aoi_geojson=aoi_geojson,
        stac_scene_count=len(result.get("scenes", scenes)),
        start_date=str(st.session_state.get("start_date", "")),
        end_date=str(st.session_state.get("end_date", "")),
        max_cloud_cover=st.session_state.get("max_cloud_cover"),
    )

    return output


b1, b2, b3 = st.columns(3)

with b1:
    run_standard = st.button(
        "Run fixed LangChain",
        use_container_width=True,
    )

with b2:
    run_agentic = st.button(
        "Run agentic LangGraph",
        use_container_width=True,
    )

with b3:
    run_both = st.button(
        "Compare both",
        type="primary",
        use_container_width=True,
    )

try:
    if run_standard:
        st.session_state["pipeline_vs_agentic_standard"] = _run_standard()

    if run_agentic:
        st.session_state["pipeline_vs_agentic_agentic"] = _run_agentic()

    if run_both:
        st.session_state["pipeline_vs_agentic_standard"] = _run_standard()
        st.session_state["pipeline_vs_agentic_agentic"] = _run_agentic()

except Exception as exc:
    st.error(f"Execution failed: {exc}")


standard_result = st.session_state.get("pipeline_vs_agentic_standard")
agentic_result = st.session_state.get("pipeline_vs_agentic_agentic")

if standard_result or agentic_result:
    st.divider()
    st.subheader("Execution results")

    if standard_result and agentic_result:
        st.caption(
            "The two detailed outputs are shown in separate tabs to keep long "
            "answers, code blocks and source text readable without column overlap."
        )

        result_tab_fixed, result_tab_agentic = st.tabs(
            [
                "Fixed LangChain result",
                "Agentic LangGraph result",
            ]
        )

        with result_tab_fixed:
            st.markdown("### Fixed LangChain pipeline")
            f1, f2 = st.columns(2)
            f1.metric("Latency", f"{standard_result['latency']:.2f}s")
            f2.metric(
                "Sources used",
                len(standard_result.get("sources", [])),
            )

            st.markdown("**Execution trace**")
            trace_df = pd.DataFrame(standard_result.get("trace", []))
            if not trace_df.empty:
                st.dataframe(
                    trace_df,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**Answer**")
            st.markdown(
                standard_result.get("answer", "")
            )

            st.markdown("**Evidence used**")
            for index, source in enumerate(
                standard_result.get("sources", []),
                start=1,
            ):
                title = source.get("file_name", "Unknown source")
                with st.expander(f"Source {index} — {title}"):
                    st.write(source.get("text", ""))

        with result_tab_agentic:
            st.markdown("### Agentic LangGraph workflow")
            a1, a2 = st.columns(2)
            a1.metric("Latency", f"{agentic_result['latency']:.2f}s")
            a2.metric(
                "Sources used",
                len(agentic_result.get("sources", [])),
            )

            st.markdown("**Execution trace**")
            trace_df = pd.DataFrame(agentic_result.get("trace", []))
            if not trace_df.empty:
                st.dataframe(
                    trace_df,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**Answer**")
            st.markdown(
                agentic_result.get("answer", "")
            )

            st.markdown("**Evidence used**")
            for index, source in enumerate(
                agentic_result.get("sources", []),
                start=1,
            ):
                title = source.get("file_name", "Unknown source")
                with st.expander(f"Source {index} — {title}"):
                    st.write(source.get("text", ""))

        st.markdown("### Comparison summary")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Aspect": "Execution path",
                        "Fixed LangChain": "Always follows the predefined sequence",
                        "Agentic LangGraph": "Planner selects the next permitted action",
                    },
                    {
                        "Aspect": "Branching",
                        "Fixed LangChain": "No",
                        "Agentic LangGraph": "Yes, within bounded tools",
                    },
                    {
                        "Aspect": "Best fit",
                        "Fixed LangChain": "Known, repeatable RAG tasks",
                        "Agentic LangGraph": "Tasks whose next step depends on context",
                    },
                    {
                        "Aspect": "Latency",
                        "Fixed LangChain": f"{standard_result['latency']:.2f}s",
                        "Agentic LangGraph": f"{agentic_result['latency']:.2f}s",
                    },
                    {
                        "Aspect": "Sources used",
                        "Fixed LangChain": len(standard_result.get("sources", [])),
                        "Agentic LangGraph": len(agentic_result.get("sources", [])),
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:
        result = standard_result or agentic_result
        st.metric("Execution engine", result["engine"])
        st.metric("Latency", f"{result['latency']:.2f}s")

        st.subheader("Execution trace")
        trace_df = pd.DataFrame(result.get("trace", []))
        if not trace_df.empty:
            st.dataframe(trace_df, use_container_width=True, hide_index=True)

        st.subheader("Answer")
        st.write(result.get("answer", ""))

        st.subheader("Evidence used")
        for index, source in enumerate(result.get("sources", []), start=1):
            title = source.get("file_name", "Unknown source")
            with st.expander(f"Source {index} — {title}"):
                st.write(source.get("text", ""))

st.divider()
st.markdown(
    """
### How to interpret the comparison

The purpose of this page is not to show that agentic AI is always better.
It demonstrates **when a fixed pipeline is sufficient and when agentic
orchestration adds value**.

- **Fixed LangChain pipeline:** the developer defines the execution order.
  It is predictable, reproducible, easier to test, and appropriate when the
  required steps are already known.
- **Agentic LangGraph workflow:** an LLM planner selects the next action from
  a bounded set of tools according to the question and the current GeoScope
  context. This is useful when the workflow may need to branch.
- **Governance principle:** the agent is deliberately bounded. STAC search,
  retrieval, reranking, and answer generation remain deterministic tools.
  The planner chooses *which tool to use next*; it does not replace the
  underlying validated operations.

**Key takeaway:** use the simplest orchestration pattern that solves the task.
Agentic AI adds value when adaptive decision-making is needed; otherwise the
fixed RAG pipeline is preferable.
"""
)
