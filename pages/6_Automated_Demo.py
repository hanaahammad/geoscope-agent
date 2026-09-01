from __future__ import annotations

import inspect
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_runner import DemoConfig, run_automated_demo
from src.llm_provider import get_generation_model
from src.retrieval import APPROACH_LABELS


st.set_page_config(
    page_title="Automated Demo",
    page_icon="▶️",
    layout="wide",
)

apply_global_style()

st.title("▶️ GeoScope Automated Demonstration")

st.markdown(
    """
This page executes the main GeoScope workflow from one button:

```text
Place or fallback AOI
→ Sentinel-2 STAC search
→ distinct-date validation
→ query rewriting
→ Chroma retrieval
→ FlashRank reranking
→ grounded LLM answer
→ optional NDVI GeoTIFF
→ monitoring log
```

Use **Standard demo** for a fast presentation. Enable GeoTIFF only when
the network is stable, because the raster step downloads remote data.
"""
)


def _try_log_demo(
    result: dict,
    elapsed: float,
) -> tuple[bool, str, str]:
    """Log the automated demo with the richer Page 5 observability schema."""
    try:
        from src.monitoring import log_run
    except Exception as exc:
        return False, f"Monitoring import unavailable: {exc}", ""

    run_id = str(uuid.uuid4())
    sources = result.get("sources", [])

    context_text = "\n\n".join(
        str(source.get("text", ""))
        for source in sources
        if source.get("text")
    )
    context_characters = len(context_text)
    estimated_context_tokens = (
        max(1, round(context_characters / 4))
        if context_characters
        else 0
    )

    config = result.get("config", {})
    retrieval_approach = config.get("retrieval_approach", "")
    original_query = result.get("question", "")
    rewritten_query = result.get("rewritten_query", "")

    values = {
        "run_id": run_id,
        "question": original_query,
        "answer": result.get("answer", ""),
        "sources": sources,
        "latency_seconds": elapsed,
        "status": "success",
        "error_message": "",

        "application": "Automated GeoScope demo",
        "crop": "Wheat",
        "season": "Winter",
        "aoi_summary": result.get("aoi_label", ""),
        "aoi_geojson": result.get("aoi_geojson"),
        "stac_scene_count": result.get("scene_count", 0),
        "start_date": config.get("start_date", ""),
        "end_date": config.get("end_date", ""),
        "max_cloud_cover": config.get("max_cloud_cover"),

        "framework": "Application RAG",
        "execution_mode": "Automated fixed RAG demo",
        "model": get_generation_model(),
        "prompt_id": "automated_demo_grounded_rag",
        "prompt_version": "1.0",

        "retrieval_approach": retrieval_approach,
        "original_query": original_query,
        "rewritten_query": (
            rewritten_query
            if rewritten_query and rewritten_query != original_query
            else ""
        ),
        "top_k": config.get("top_k"),
        "candidate_k": (
            config.get("candidate_k")
            if retrieval_approach in {"rerank", "rewrite_rerank"}
            else None
        ),
        "chunk_count": len(sources),
        "context_characters": context_characters,
        "estimated_context_tokens": estimated_context_tokens,

        "trace": result.get("steps", []),
        "step_count": len(result.get("steps", [])),
    }

    try:
        accepted = inspect.signature(log_run).parameters
        filtered = {k: v for k, v in values.items() if k in accepted}
        log_run(**filtered)
        return True, "Run written to the GeoScope monitoring store.", run_id
    except Exception as exc:
        return False, f"Demo completed, but monitoring logging failed: {exc}", run_id


with st.expander(
    "Architecture used in this demonstration",
    expanded=True,
):
    st.code(
        """
PDF / HTML documents
        │
        ▼
Extract → clean → chunk → Ollama embeddings
        │
        ▼
Chroma vector store
        │
User question
        ▼
LLM query rewriting
        ▼
Vector candidate retrieval
        ▼
FlashRank reranking
        ▼
Top context chunks ───────────────┐
                                  │
Text/map AOI → Sentinel-2 STAC ───┤
                                  ▼
                         Ollama / OpenAI answer
                                  │
                         Red / NIR / NDVI
                                  ▼
                         Downloadable GeoTIFF

Feedback + runs → DuckDB/dlt → Streamlit monitoring
""".strip(),
        language="text",
    )

st.subheader("Demo configuration")

left, middle, right = st.columns(3)

with left:
    place_query = st.text_input(
        "Place",
        value="Kom Ombo, Aswan, Egypt",
    )

    use_text_geocoding = st.checkbox(
        "Try text geocoding first",
        value=True,
        help=(
            "If Nominatim fails, the demo uses the bundled Kom Ombo "
            "study-area box."
        ),
    )

with middle:
    start_date = st.date_input(
        "Start date",
        value=datetime(2025, 11, 1).date(),
    )
    end_date = st.date_input(
        "End date",
        value=datetime(2026, 3, 31).date(),
    )

with right:
    max_cloud_cover = st.slider(
        "Maximum cloud cover (%)",
        0,
        100,
        40,
    )
    scene_limit = st.slider(
        "Maximum scene items",
        3,
        20,
        10,
    )

question = st.text_area(
    "Demonstration question",
    value=(
        "Which Sentinel-2 bands and processing workflow should I use "
        "to assess wheat vegetation condition in this area, and is the "
        "available temporal coverage sufficient for a time series?"
    ),
    height=120,
)

c1, c2, c3 = st.columns(3)

with c1:
    retrieval_approach = st.selectbox(
        "Retrieval pipeline",
        list(APPROACH_LABELS.keys()),
        format_func=lambda key: APPROACH_LABELS[key],
        index=3,
    )

with c2:
    top_k = st.slider(
        "Final context chunks",
        1,
        10,
        5,
    )

with c3:
    candidate_k = st.slider(
        "Candidates before reranking",
        5,
        30,
        15,
        disabled=retrieval_approach
        not in {"rerank", "rewrite_rerank"},
    )

st.info(
    """
**What happens during this automated demo?**

GeoScope executes the workflow visibly, step by step:

**AOI → STAC search → temporal validation → query rewriting → retrieval →
reranking → grounded generation → optional GeoTIFF → monitoring log**

This page uses the **Application RAG** workflow. It is deterministic application
code, not LangChain or LangGraph. Page 9 is where those framework
implementations are compared explicitly.
"""
)

generate_geotiff = st.checkbox(
    "Include real GeoTIFF generation",
    value=False,
    help=(
        "Enable for the full demonstration. Keep disabled for a faster "
        "and safer recorded walkthrough."
    ),
)

product = st.radio(
    "GeoTIFF product",
    ["NDVI", "Red", "NIR"],
    horizontal=True,
    disabled=not generate_geotiff,
)

if st.button(
    "▶ Run automated demonstration",
    type="primary",
    use_container_width=True,
):
    config = DemoConfig(
        place_query=place_query,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        max_cloud_cover=max_cloud_cover,
        scene_limit=scene_limit,
        retrieval_approach=retrieval_approach,
        top_k=top_k,
        candidate_k=candidate_k,
        product=product,
        generate_geotiff=generate_geotiff,
        use_text_geocoding=use_text_geocoding,
    )

    st.markdown("### ▶ Live execution")

    total_steps = 7
    step_order = {
        "AOI": 1,
        "AOI warning": 1,
        "STAC": 2,
        "Retrieval": 4,
        "Generation": 5,
        "GeoTIFF": 6,
        "Completed": 7,
    }

    progress_values = {
        "AOI": 12,
        "AOI warning": 15,
        "STAC": 30,
        "Retrieval": 58,
        "Generation": 78,
        "GeoTIFF": 90,
        "Completed": 100,
    }

    step_titles = {
        "AOI": "Resolve Area of Interest",
        "AOI warning": "Resolve Area of Interest",
        "STAC": "Search satellite catalogue",
        "Retrieval": "Retrieve and rerank knowledge",
        "Generation": "Generate grounded answer",
        "GeoTIFF": "Generate optional GeoTIFF",
        "Completed": "Log and complete run",
    }

    step_explanations = {
        "AOI": (
            "GeoScope resolves the study area. If text geocoding is unavailable, "
            "the bundled Kom Ombo fallback AOI is used."
        ),
        "AOI warning": (
            "The preferred AOI path was unavailable, so GeoScope uses the safe "
            "fallback instead of stopping the demonstration."
        ),
        "STAC": (
            "GeoScope queries the satellite catalogue and checks distinct "
            "acquisition dates. Several scene items do not necessarily mean "
            "several dates."
        ),
        "Retrieval": (
            "GeoScope prepares the question for retrieval. Depending on the "
            "selected pipeline, it may rewrite the query, retrieve Chroma "
            "candidates and rerank them with FlashRank."
        ),
        "Generation": (
            "The strongest evidence chunks are assembled into context and passed "
            "to the generation model to produce a grounded answer."
        ),
        "GeoTIFF": (
            "GeoScope attempts the optional raster-processing step. This stage "
            "depends on remote raster access and network conditions."
        ),
        "Completed": (
            "Execution metadata, prompt version, retrieval configuration, "
            "context size, latency and the actual demo steps are logged for "
            "Page 5 Monitoring."
        ),
    }

    # Visible placeholders are created before execution starts so the user
    # immediately sees that the run has begun.
    counter_box = st.empty()
    progress = st.progress(0)
    status_box = st.empty()
    narrative_box = st.container()
    messages: list[dict[str, str]] = []

    counter_box.markdown(
        f"### Step 1/{total_steps} — Starting GeoScope demo..."
    )
    status_box.info(
        "Initializing the automated workflow and preparing the configured AOI, "
        "dates and retrieval settings."
    )
    progress.progress(3)

    def update_progress(step: str, message: str) -> None:
        messages.append({"step": step, "message": message})

        current_step = step_order.get(step, 1)
        title = step_titles.get(step, step)

        counter_box.markdown(
            f"### Step {current_step}/{total_steps} — {title}"
        )
        progress.progress(progress_values.get(step, 5))

        explanation = step_explanations.get(
            step,
            "GeoScope is executing the next workflow stage.",
        )

        status_box.info(
            f"**{message}**\n\n{explanation}"
        )

        with narrative_box:
            st.markdown(
                f"**Step {current_step}/{total_steps} · {title}**  \n"
                f"{message}  \n"
                f"<span style='color:#667085'>{explanation}</span>",
                unsafe_allow_html=True,
            )


    started = time.perf_counter()

    try:
        result = run_automated_demo(
            question=question,
            config=config,
            callback=update_progress,
        )
        elapsed = time.perf_counter() - started

        # The demo runner does not always emit a separate callback for every
        # conceptual step, so complete the visible counter here.
        counter_box.markdown(
            f"### Step 7/{total_steps} — Log and complete run"
        )
        progress.progress(96)
        status_box.info(
            "The AI workflow is complete. GeoScope is now writing the run "
            "metadata to the monitoring store."
        )

        logged, log_message, run_id = _try_log_demo(
            result,
            elapsed,
        )

        result["elapsed_seconds"] = elapsed
        result["monitoring_logged"] = logged
        result["monitoring_message"] = log_message
        result["run_id"] = run_id
        result["progress_messages"] = messages
        result["completed_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        st.session_state[
            "automated_demo_result"
        ] = result

        # Reuse the result in the normal evaluation page.
        st.session_state["aoi_geojson"] = result[
            "aoi_geojson"
        ]
        st.session_state["aoi_summary"] = result[
            "aoi_label"
        ]
        st.session_state["stac_scenes"] = result[
            "scenes"
        ]
        st.session_state["last_question"] = result[
            "question"
        ]
        st.session_state[
            "last_augmented_question"
        ] = result["augmented_question"]
        st.session_state["last_answer"] = result[
            "answer"
        ]
        st.session_state["last_sources"] = result[
            "sources"
        ]
        st.session_state["last_run_id"] = run_id
        st.session_state["last_retrieval_approach"] = retrieval_approach
        st.session_state["last_prompt_id"] = "automated_demo_grounded_rag"
        st.session_state["last_prompt_version"] = "1.0"
        st.session_state["last_framework"] = "Application RAG"
        st.session_state["last_execution_mode"] = "Automated fixed RAG demo"
        st.session_state["last_model"] = get_generation_model()
        st.session_state["last_top_k"] = top_k
        st.session_state["last_candidate_k"] = (
            candidate_k
            if retrieval_approach in {"rerank", "rewrite_rerank"}
            else None
        )

        counter_box.markdown(
            f"### Step 7/{total_steps} — Complete ✅"
        )
        progress.progress(100)
        status_box.success(
            f"Demonstration completed in {elapsed:.2f} seconds. "
            "The run is now available in Page 5 Monitoring."
        )

    except Exception as exc:
        elapsed = time.perf_counter() - started
        status_box.error(
            f"Automated demo failed after {elapsed:.2f} seconds: {exc}"
        )

result = st.session_state.get(
    "automated_demo_result"
)

if result:
    st.divider()
    st.subheader("Demonstration results")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Scene items",
        result["scene_count"],
    )
    m2.metric(
        "Distinct dates",
        result["distinct_date_count"],
    )
    m3.metric(
        "Time series",
        (
            "Possible"
            if result["time_series_available"]
            else "Insufficient"
        ),
    )
    m4.metric(
        "Runtime",
        f"{result['elapsed_seconds']:.1f}s",
    )

    st.success(
        f"AOI: {result['aoi_label']} "
        f"({result['aoi_source']})"
    )

    if not result["time_series_available"]:
        st.warning(
            "The selected scene items do not provide at least two "
            "distinct acquisition dates. GeoScope will not describe "
            "them as a time series."
        )

    st.markdown("### Executed steps")
    st.dataframe(
        pd.DataFrame(result["steps"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Query rewriting")
    st.write("**Original demonstration question**")
    st.code(result["question"])
    st.write("**Query used by retrieval**")
    st.code(result["rewritten_query"])

    st.markdown("### Grounded answer")
    st.markdown(result["answer"])

    st.markdown("### Retrieved and reranked evidence")

    source_rows = [
        {
            "final_rank": source.get("rank"),
            "vector_rank": source.get("vector_rank"),
            "file": source.get("file_name"),
            "page": source.get("page_number"),
            "vector_distance": source.get("distance"),
            "rerank_score": source.get("rerank_score"),
        }
        for source in result["sources"]
    ]

    st.dataframe(
        pd.DataFrame(source_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"{result['monitoring_message']} "
        f"Run ID: {result.get('run_id', 'Not available')}"
    )

    if result.get("geotiff_bytes"):
        summary = result["geotiff_summary"]

        st.markdown("### Generated GeoTIFF")
        st.json(summary)

        st.download_button(
            "Download generated GeoTIFF",
            data=result["geotiff_bytes"],
            file_name=result["geotiff_filename"],
            mime="image/tiff",
            type="primary",
            use_container_width=True,
        )
