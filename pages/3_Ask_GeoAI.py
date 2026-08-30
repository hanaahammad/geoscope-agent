from __future__ import annotations

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

from src.dlt_logging import log_user_feedback
from src.generation import generate_answer
from src.llm_provider import generate_text, get_generation_model
from src.monitoring import log_run
from src.retrieval import (
    APPROACH_LABELS,
    search_documents,
)
from src.reranking import (
    FLASHRANK_MODEL_NAME,
    check_flashrank_ready,
    flashrank_local_cache_available,
)



POST_PROCESSING_INSTRUCTIONS = """
You post-process an existing GeoScope answer.

Do not perform new retrieval and do not introduce new technical claims.
Preserve Earth Observation terminology, dataset names, band names,
indices, numerical values, limitations, and source references exactly
where relevant.

If translating, use clear professional language appropriate for an
Earth Observation researcher.
""".strip()


def transform_existing_answer(
    answer: str,
    action: str,
    target_language: str,
) -> str:
    if not answer.strip():
        raise ValueError("There is no GeoScope answer to transform.")

    if action == "Summarize":
        task = (
            "Summarize the existing answer concisely. Preserve the "
            "technical meaning, limitations, numerical values, and citations."
        )
    elif action == "Translate":
        task = (
            f"Translate the existing answer into {target_language}. "
            "Preserve technical terminology, numerical values, and citations."
        )
    else:
        task = (
            f"First summarize the existing answer, then translate the "
            f"summary into {target_language}. Preserve technical terminology, "
            "numerical values, limitations, and citations."
        )

    prompt = f"""
TASK:
{task}

EXISTING GEOSCOPE ANSWER:
{answer}
""".strip()

    return generate_text(
        instructions=POST_PROCESSING_INSTRUCTIONS,
        prompt=prompt,
        model=get_generation_model(),
    )


st.set_page_config(
    page_title="Ask GeoAI",
    page_icon="🤖",
    layout="wide",
)

apply_global_style()

st.title("🤖 Step 3 — Ask GeoAI")
st.caption("Ask a new question, or continue working with the previous result saved in this Streamlit session.")
st.caption(
    "Query rewriting, semantic retrieval, FlashRank reranking, "
    "AOI/STAC context, and grounded generation."
)


aoi_geojson = st.session_state.get("aoi_geojson")
aoi_summary = st.session_state.get(
    "aoi_summary",
    "No AOI description available.",
)
scenes = st.session_state.get("stac_scenes", [])

EXAMPLE_QUESTIONS = {
    "Agriculture and crop monitoring": [
        "Which datasets and workflow should I use to monitor wheat in the selected AOI?",
        "How can Sentinel-1 SAR support crop monitoring in cloudy conditions?",
        "How can Sentinel-1 and Sentinel-2 be combined for crop monitoring?",
        "Which vegetation indices are useful for detecting crop stress?",
        "How can satellite time series support crop phenology analysis?",
        "Which datasets are suitable for field-level crop classification?",
        "How can crop calendars improve satellite image selection?",
        "Which data are most suitable for monitoring irrigation patterns?",
        "What are the limitations of using optical imagery for crop monitoring?",
        "How can MODIS complement Sentinel-2 for seasonal crop monitoring?",
    ],
    "Sentinel-1 and SAR": [
        "What Sentinel-1 preprocessing steps are required before crop analysis?",
        "Which Sentinel-1 polarization is useful for agricultural monitoring?",
        "How can SAR backscatter changes indicate crop growth stages?",
        "Why is Sentinel-1 useful when cloud cover is persistent?",
        "How can Sentinel-1 support flood assessment in agricultural areas?",
        "What are the main interpretation challenges of SAR imagery?",
    ],
    "Sentinel-2 and optical": [
        "Which Sentinel-2 bands are useful for vegetation monitoring?",
        "Why are Sentinel-2 red-edge bands useful for crop analysis?",
        "Which preprocessing steps are required for Sentinel-2 Level-2A data?",
        "How should cloudy Sentinel-2 observations be handled?",
        "What spatial resolution does Sentinel-2 provide for crop monitoring?",
        "Which Sentinel-2 indices can support land-cover classification?",
    ],
    "Landsat and urban heat": [
        "Which Landsat product should I use for land-surface temperature analysis?",
        "How can Landsat support urban heat mapping in the selected AOI?",
        "What is the difference between surface reflectance and surface temperature?",
        "How can Landsat and ECOSTRESS complement each other for urban heat studies?",
        "What are the limitations of thermal remote sensing in cities?",
        "Which variables should be combined with land-surface temperature for urban heat analysis?",
    ],
    "MODIS and phenology": [
        "What is the MOD13 vegetation index product?",
        "How can MODIS support crop phenology analysis?",
        "When is MODIS preferable to Sentinel-2?",
        "What are the spatial-resolution limitations of MODIS?",
        "How can MODIS time series support regional crop monitoring?",
    ],
    "Land-cover change and floods": [
        "Which datasets are suitable for land-cover change detection?",
        "How can Sentinel-1 support flood detection?",
        "How can Sentinel-2 and Landsat be combined for long-term change analysis?",
        "What preprocessing is required before land-cover change detection?",
        "Which data are suitable for monitoring water-body changes?",
        "How can cloud cover affect flood and land-cover analysis?",
    ],
    "GeoAI and foundation models": [
        "What is a geospatial foundation model?",
        "How can foundation-model embeddings be used in remote sensing?",
        "Can a remote-sensing foundation model replace domain-specific validation?",
        "How can foundation models support land-cover classification?",
        "What is the difference between a vision-language model and a remote-sensing foundation model?",
        "What are the main limitations of remote-sensing foundation models?",
    ],
    "AOI-aware and STAC": [
        "Which available Sentinel-2 scene has the lowest cloud cover?",
        "Are the available scenes suitable for crop monitoring in this AOI?",
        "Which datasets cover the selected AOI during the selected period?",
        "How should the recommendation change if cloud cover is high?",
        "Are the available scenes suitable for field-level analysis?",
        "Which scene dates are most appropriate for seasonal crop monitoring?",
    ],
}

if "ask_question" not in st.session_state:
    st.session_state["ask_question"] = (
        "Which datasets and workflow should I use "
        "to monitor wheat in the selected AOI?"
    )

main_col, explanation_col = st.columns([1.55, 0.85], gap="large")

with main_col:
    st.markdown("### Example questions")

    example_col1, example_col2 = st.columns([0.42, 0.58])

    with example_col1:
        example_group = st.selectbox(
            "Question category",
            list(EXAMPLE_QUESTIONS.keys()),
        )

    with example_col2:
        selected_example = st.selectbox(
            "Choose an example",
            EXAMPLE_QUESTIONS[example_group],
        )

    if st.button(
        "Use selected example",
        use_container_width=True,
    ):
        st.session_state["ask_question"] = selected_example
        st.rerun()

with explanation_col:
    st.markdown(
        """
        <div style="
            background:#F3ECDD;
            border:1px solid #D5C7A8;
            border-left:6px solid #8A7653;
            border-radius:12px;
            padding:1rem 1.1rem;
            margin-top:0.15rem;
        ">
          <div style="
              font-size:1.05rem;
              font-weight:700;
              color:#493F2E;
              margin-bottom:0.55rem;
          ">
            💡 What we do here
          </div>

          <div style="font-size:0.89rem; line-height:1.45; color:#493F2E;">
            <b>Ask GeoAI</b> is the RAG execution step. GeoScope does not send
            the question directly to the LLM. It first finds and ranks evidence,
            then gives that evidence to the model.
          </div>

          <div style="
              margin:0.8rem 0 0.45rem 0;
              font-size:0.90rem;
              font-weight:700;
              color:#493F2E;
          ">
            Retrieval options
          </div>

          <div style="font-size:0.84rem; line-height:1.48; color:#493F2E;">
            <b>Vector</b> — semantic search using the original query.<br>
            <b>Rewrite</b> — improves retrieval wording before vector search.<br>
            <b>Rerank</b> — retrieves candidates, then FlashRank reorders them.<br>
            <b>Rewrite + Rerank</b> — full advanced retrieval pipeline.
          </div>

          <div style="
              margin:0.8rem 0 0.45rem 0;
              font-size:0.90rem;
              font-weight:700;
              color:#493F2E;
          ">
            How to read the results
          </div>

          <div style="font-size:0.84rem; line-height:1.48; color:#493F2E;">
            <b>Original input</b> — user question + GeoScope context.<br>
            <b>Retrieval query</b> — wording used for semantic search only.<br>
            <b>Vector rank</b> — initial semantic-search position.<br>
            <b>Final rank</b> — position after reranking.<br>
            <b>Rerank score</b> — estimated relevance to the question.
          </div>

          <div style="
              margin-top:0.85rem;
              padding-top:0.7rem;
              border-top:1px solid #D5C7A8;
              font-size:0.82rem;
              line-height:1.4;
              color:#5A4D37;
          ">
            <b>Governance:</b> exposing the retrieval path improves
            transparency and explainability because the user can inspect
            how evidence was selected before the answer was generated.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

left, right = st.columns([0.78, 1.22])

with left:
    application = st.selectbox(
        "Application",
        [
            "Crop monitoring",
            "Crop classification",
            "Urban heat",
            "Land-cover change",
            "Flood assessment",
            "GeoAI foundation models",
        ],
    )

    crop = st.selectbox(
        "Crop",
        [
            "Any",
            "Wheat",
            "Maize",
            "Rice",
            "Cotton",
            "Potato",
        ],
    )

    season = st.selectbox(
        "Season",
        [
            "Any",
            "Winter",
            "Summer",
            "All",
        ],
    )

    approach = st.selectbox(
        "Retrieval approach",
        list(APPROACH_LABELS.keys()),
        format_func=lambda key: APPROACH_LABELS[key],
        index=3,
        help=(
            "The full approach rewrites the query, retrieves a wider "
            "candidate set from Chroma, and reranks it with FlashRank."
        ),
    )

    top_k = st.slider(
        "Final context chunks",
        1,
        10,
        5,
    )

    candidate_k = st.slider(
        "Vector candidates before reranking",
        max(top_k, 5),
        30,
        max(15, top_k * 3),
        disabled=approach not in {
            "rerank",
            "rewrite_rerank",
        },
    )

    if approach in {"rerank", "rewrite_rerank"}:
        if flashrank_local_cache_available():
            st.caption(
                f"✅ FlashRank local cache detected: `{FLASHRANK_MODEL_NAME}`"
            )
        else:
            st.warning(
                "⚠️ FlashRank local model cache was not detected. "
                "GeoScope will try to initialize the reranker when you run "
                "the query. If the model cannot be downloaded in this "
                "environment, the run will stop cleanly and you can select "
                "**Vector search** or **Query rewriting + vector search**."
            )

with right:
    question = st.text_area(
        "Question",
        key="ask_question",
        height=170,
    )

st.write(
    f"AOI available: **{'Yes' if aoi_geojson else 'No'}** · "
    f"STAC scene items: **{len(scenes)}**"
)

if aoi_geojson:
    st.info(f"**AOI used:** {aoi_summary}")

unique_dates = sorted(
    {
        scene.get("date")
        for scene in scenes
        if scene.get("date")
    }
)

if scenes:
    st.write(
        f"Distinct acquisition dates: **{len(unique_dates)}** · "
        f"Time series possible: "
        f"**{'Yes' if len(unique_dates) >= 2 else 'No'}**"
    )


if st.button(
    "Run GeoScope",
    type="primary",
    use_container_width=True,
):
    run_id = str(uuid.uuid4())
    started = time.perf_counter()

    answer = ""
    sources = []
    status = "success"
    error_message = ""

    try:
        if not question.strip():
            raise ValueError("Please enter a question.")

        if approach in {"rerank", "rewrite_rerank"}:
            with st.spinner("Checking FlashRank reranker..."):
                flashrank_ready, flashrank_message = (
                    check_flashrank_ready()
                )

            if not flashrank_ready:
                raise RuntimeError(
                    "FlashRank reranking is unavailable for this run. "
                    "Please select **Vector search** or "
                    "**Query rewriting + vector search**, or install/cache "
                    f"`{FLASHRANK_MODEL_NAME}` and retry.\n\n"
                    f"{flashrank_message}"
                )

        scene_lines = [
            (
                f"- {scene.get('item_id')} | "
                f"date={scene.get('date')} | "
                f"cloud_cover={scene.get('cloud_cover')}"
            )
            for scene in scenes[:5]
        ]

        scene_summary = (
            "\n".join(scene_lines)
            if scene_lines
            else "No STAC scenes were selected."
        )

        augmented_question = f"""
User question:
{question}

Selected filters:
- Application: {application}
- Crop: {crop}
- Season: {season}
- AOI supplied: {"yes" if aoi_geojson else "no"}
- AOI description: {aoi_summary}
- STAC scene items: {len(scenes)}
- Distinct acquisition dates: {len(unique_dates)}
- Distinct dates: {", ".join(unique_dates) if unique_dates else "None"}
- Time-series analysis possible: {"yes" if len(unique_dates) >= 2 else "no"}

Available Sentinel-2 scenes:
{scene_summary}
""".strip()

        with st.spinner(
            "Rewriting, retrieving, and reranking knowledge..."
        ):
            sources = search_documents(
                augmented_question,
                top_k=top_k,
                approach=approach,
                candidate_k=(
                    candidate_k
                    if approach in {
                        "rerank",
                        "rewrite_rerank",
                    }
                    else None
                ),
            )

        with st.spinner("Generating the grounded answer..."):
            answer = generate_answer(
                question=augmented_question,
                retrieved_chunks=sources,
            )

    except Exception as exc:
        status = "failed"
        error_message = str(exc)

    latency = time.perf_counter() - started

    log_run(
        run_id=run_id,
        question=question,
        answer=answer,
        sources=sources,
        latency_seconds=latency,
        status=status,
        error_message=error_message,
        application=application,
        crop=crop,
        season=season,
        aoi_summary=aoi_summary,
        aoi_geojson=aoi_geojson,
        stac_scene_count=len(scenes),
        start_date=st.session_state.get("start_date", ""),
        end_date=st.session_state.get("end_date", ""),
        max_cloud_cover=st.session_state.get(
            "max_cloud_cover"
        ),
    )

    if status == "success":
        st.session_state["last_run_id"] = run_id
        st.session_state["last_question"] = question
        st.session_state["last_augmented_question"] = (
            augmented_question
        )
        st.session_state["last_answer"] = answer
        st.session_state["last_sources"] = sources
        st.session_state["last_retrieval_approach"] = (
            approach
        )
        st.session_state["last_latency"] = latency
        st.session_state["last_application"] = application
        st.session_state["last_crop"] = crop
        st.session_state["last_season"] = season
    else:
        st.session_state["last_run_error"] = error_message


# ---------------------------------------------------------------------------
# Persisted result rendering
# ---------------------------------------------------------------------------
# Streamlit reruns the full script when any button is clicked.
# Therefore the answer must be rendered from session_state, not only inside
# the "Run GeoScope" button block. This keeps the original answer visible
# while summary / translation / feedback actions are executed.

active_answer = st.session_state.get("last_answer")
active_sources = st.session_state.get("last_sources", [])
active_run_id = st.session_state.get("last_run_id")
active_question = st.session_state.get("last_question", question)
active_augmented_question = st.session_state.get(
    "last_augmented_question",
    "",
)
active_approach = st.session_state.get(
    "last_retrieval_approach",
    approach,
)
active_latency = st.session_state.get(
    "last_latency",
    0.0,
)

if st.session_state.get("last_run_error"):
    st.error(st.session_state["last_run_error"])
    st.session_state.pop("last_run_error", None)

if active_answer and active_run_id:
    st.divider()

    st.info(
        "A previous GeoScope result is available from the current session. "
        "You can review it below, reuse it for summary/translation, or clear it "
        "before starting a new analysis."
    )

    previous_col, clear_col = st.columns([4, 1])

    with previous_col:
        st.markdown("### Previous GeoScope result")
        st.write(f"**Question:** {active_question}")
        st.caption(
            f"Run ID: {active_run_id} · "
            f"Retrieval: {APPROACH_LABELS.get(active_approach, active_approach)} · "
            f"Latency: {active_latency:.2f} seconds"
        )

    with clear_col:
        st.write("")
        if st.button(
            "Clear previous result",
            key=f"clear_previous_{active_run_id}",
            use_container_width=True,
        ):
            keys_to_clear = [
                "last_run_id",
                "last_question",
                "last_augmented_question",
                "last_answer",
                "last_sources",
                "last_retrieval_approach",
                "last_latency",
                "last_application",
                "last_crop",
                "last_season",
                "last_user_feedback",
            ]

            for key in keys_to_clear:
                st.session_state.pop(key, None)

            # Remove any post-processing result attached to this run.
            st.session_state.pop(
                f"post_processed_answer_{active_run_id}",
                None,
            )

            st.rerun()

    st.subheader("Answer")
    st.markdown(active_answer)
    st.caption(
        f"Run ID: {active_run_id} · "
        f"Latency: {active_latency:.2f} seconds"
    )

    if active_sources:
        original_query = active_sources[0].get(
            "original_query",
            active_augmented_question,
        )
        retrieval_query = active_sources[0].get(
            "retrieval_query",
            original_query,
        )

        with st.expander(
            "Inspect retrieval pipeline",
            expanded=True,
        ):
            st.write(
                f"**Approach:** "
                f"{APPROACH_LABELS[active_approach]}"
            )
            st.caption(
                "This section explains the retrieval path before answer "
                "generation. The rewritten query is used only for retrieval; "
                "the final answer remains grounded in the original user intent "
                "and retrieved evidence."
            )
            st.write("**Original retrieval input**")
            st.code(original_query)
            st.write("**Query used for vector search**")
            st.code(retrieval_query)
            st.info(
                "Interpretation: GeoScope first prepares a retrieval-friendly "
                "query, then performs semantic search, and finally optionally "
                "reranks the candidate chunks with FlashRank to improve source "
                "ordering."
            )

    st.subheader("Retrieved and reranked sources")
    st.caption(
        "These are the evidence chunks used to ground the answer. "
        "Compare vector rank with final rank to see how reranking changed "
        "the order. Lower vector distance generally means stronger semantic "
        "similarity."
    )

    rows = [
        {
            "final_rank": source.get("rank"),
            "vector_rank": source.get("vector_rank"),
            "file": source.get("file_name"),
            "page": source.get("page_number"),
            "vector_distance": round(
                source.get("distance", 0.0),
                4,
            ),
            "rerank_score": (
                round(
                    source.get("rerank_score"),
                    4,
                )
                if source.get("rerank_score")
                is not None
                else None
            ),
        }
        for source in active_sources
    ]

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    for index, source in enumerate(
        active_sources,
        start=1,
    ):
        with st.expander(
            f"Source {index} — "
            f"{source.get('file_name', 'Unknown')}"
        ):
            st.write(source.get("text", ""))

    st.divider()
    st.subheader("Quick human feedback")
    st.caption(
        "Human feedback is captured here at the point of use. "
        "Formal LLM-as-a-judge evaluation remains in Step 4."
    )

    feedback_key = f"feedback_rating_{active_run_id}"
    comment_key = f"feedback_comment_{active_run_id}"

    rating = st.radio(
        "Was this answer useful?",
        ["👍 Yes", "👎 No"],
        horizontal=True,
        key=feedback_key,
    )

    comment = st.text_area(
        "Optional comment",
        placeholder=(
            "What was useful, unclear, or should be improved?"
        ),
        key=comment_key,
    )

    if st.button(
        "Save feedback",
        key=f"save_feedback_{active_run_id}",
        use_container_width=True,
    ):
        try:
            log_user_feedback(
                {
                    "run_id": active_run_id,
                    "feedback_timestamp": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "rating": rating,
                    "comment": comment,
                    "question": active_question,
                }
            )

            st.session_state["last_user_feedback"] = {
                "run_id": active_run_id,
                "rating": rating,
                "comment": comment,
            }

            st.success(
                "Feedback saved. It will be available "
                "for governance and monitoring metrics."
            )
        except Exception as exc:
            st.error(
                f"Could not save feedback: {exc}"
            )

    st.divider()
    st.markdown("### Summarize or translate")
    st.caption(
        "This extends the current page with an additional transformed view. "
        "The full original grounded answer above stays visible and unchanged. "
        "Retrieval is not rerun."
    )

    pp_action_col, pp_language_col, pp_button_col = st.columns(
        [1.0, 0.9, 1.0]
    )

    with pp_action_col:
        post_action = st.selectbox(
            "Action",
            [
                "Summarize",
                "Translate",
                "Summarize + Translate",
            ],
            key=f"post_action_{active_run_id}",
        )

    with pp_language_col:
        target_language = st.selectbox(
            "Target language",
            [
                "English",
                "French",
                "Arabic",
                "Spanish",
                "German",
            ],
            key=f"target_language_{active_run_id}",
            disabled=(post_action == "Summarize"),
        )

    with pp_button_col:
        st.write("")
        st.write("")
        run_post_processing = st.button(
            "Run transformation",
            key=f"run_post_processing_{active_run_id}",
            use_container_width=True,
        )

    if run_post_processing:
        try:
            with st.spinner(
                "Transforming the existing answer..."
            ):
                transformed_answer = (
                    transform_existing_answer(
                        answer=active_answer,
                        action=post_action,
                        target_language=target_language,
                    )
                )

            st.session_state[
                f"post_processed_answer_{active_run_id}"
            ] = {
                "action": post_action,
                "language": target_language,
                "answer": transformed_answer,
            }

        except Exception as exc:
            st.error(
                f"Post-processing failed: {exc}"
            )

    post_result = st.session_state.get(
        f"post_processed_answer_{active_run_id}"
    )

    if post_result:
        label = post_result["action"]

        if post_result["action"] != "Summarize":
            label += (
                f" — {post_result['language']}"
            )

        st.markdown(f"#### {label}")
        st.markdown(post_result["answer"])
        st.info(
            "This is an additional transformed view. "
            "The original grounded answer remains visible above and was not "
            "replaced. Retrieval, source selection, and evidence were not "
            "rerun."
        )
