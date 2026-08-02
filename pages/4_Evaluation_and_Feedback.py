from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dlt_logging import (
    log_generation_evaluation,
    log_retrieval_evaluation,
    log_user_feedback,
)
from src.evaluation import (
    evaluate_generation,
    evaluate_retrieval,
    load_ground_truth,
)


st.set_page_config(
    page_title="Evaluation and Feedback",
    page_icon="✅",
    layout="wide",
)

apply_global_style()

st.title("✅ Step 4 — Evaluation and Feedback")

st.markdown(
    """
### What is implemented on this page?

GeoScope is evaluated at two levels:

1. **Retrieval evaluation** checks whether the expected source document
   appears in the top results and how highly it is ranked.
2. **Generation evaluation** checks whether the final answer is relevant,
   grounded, complete, technically correct, and properly supported.

Evaluation records and user feedback are loaded into DuckDB through a
small **dlt** pipeline so they can later be monitored and compared.
"""
)

card1, card2 = st.columns(2)

with card1:
    st.info(
        """
### 🔎 Retrieval Evaluation

**Question:** Did GeoScope retrieve the correct knowledge?

**Metrics**
- Hit Rate
- Mean Reciprocal Rank
- Expected source rank
- Failed questions
"""
    )

with card2:
    st.info(
        """
### 🧠 Generation Evaluation

**Question:** Did GeoScope produce a good answer from the retrieved context?

**Metrics**
- Relevance
- Groundedness
- Completeness
- Technical correctness
- Citation quality
- Geographic relevance
"""
    )

st.divider()

st.subheader("Domain")

st.markdown(
    """
GeoScope covers **remote sensing, Earth observation, agriculture, and
GeoAI**. Typical questions concern Sentinel-1 and Sentinel-2, Landsat,
MODIS, ECOSTRESS, crop monitoring, urban heat, land-cover change, and
remote-sensing foundation models.
"""
)

retrieval_tab, generation_tab, examples_tab = st.tabs(
    [
        "Retrieval Evaluation",
        "Generation Evaluation",
        "Question Examples",
    ]
)

with retrieval_tab:
    st.markdown(
        """
## Principle

Each ground-truth question is linked to an expected source document.
GeoScope retrieves the top-*k* document chunks.

- A **hit** occurs when the expected document appears anywhere in the
  top-*k* results.
- **Hit Rate** measures the proportion of questions with a hit.
- **MRR** rewards systems that rank the expected source near the top.

For example, rank 1 gives `1.0`, rank 2 gives `0.5`, and rank 4 gives
`0.25`.
"""
    )

    with st.expander("Representative evaluation code"):
        st.code(
            """
rank = find_expected_rank(results, expected_document)
hit = rank is not None
reciprocal_rank = 1 / rank if rank else 0
""".strip(),
            language="python",
        )

    try:
        ground_truth = load_ground_truth()
    except Exception as exc:
        st.error(str(exc))
        ground_truth = pd.DataFrame()

    if not ground_truth.empty:
        domains = ["All"] + sorted(
            ground_truth["domain"].dropna().unique().tolist()
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            selected_domain = st.selectbox(
                "Domain",
                domains,
            )

        with c2:
            selected_difficulty = st.selectbox(
                "Difficulty",
                ["All", "Beginner", "Intermediate", "Advanced"],
            )

        with c3:
            top_k = st.slider(
                "Top-k",
                1,
                10,
                5,
            )

        filtered = ground_truth.copy()

        if selected_domain != "All":
            filtered = filtered[
                filtered["domain"] == selected_domain
            ]

        if selected_difficulty != "All":
            filtered = filtered[
                filtered["difficulty"] == selected_difficulty
            ]

        st.write(f"Questions selected: **{len(filtered)}**")

        with st.expander("View ground-truth questions"):
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
            )

        if st.button(
            "Run retrieval evaluation",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Running retrieval evaluation..."
                ):
                    metrics, details = evaluate_retrieval(
                        filtered,
                        top_k=top_k,
                    )

                st.session_state["retrieval_metrics"] = metrics
                st.session_state["retrieval_details"] = details

                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                records = details.assign(
                    evaluation_timestamp=timestamp,
                    top_k=top_k,
                ).to_dict(orient="records")

                log_retrieval_evaluation(records)

            except Exception as exc:
                st.error(str(exc))

        metrics = st.session_state.get("retrieval_metrics")
        details = st.session_state.get("retrieval_details")

        if metrics:
            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Questions",
                metrics["questions_evaluated"],
            )
            m2.metric(
                "Hit Rate",
                f"{metrics['hit_rate']:.3f}",
            )
            m3.metric(
                "MRR",
                f"{metrics['mrr']:.3f}",
            )
            m4.metric(
                "Failures",
                metrics["failures"],
            )

            st.markdown(
                f"""
**Interpretation:** GeoScope found the expected source for
**{metrics['hit_rate']:.1%}** of the evaluated questions. The MRR of
**{metrics['mrr']:.3f}** indicates how close the correct documents were
to the top of the ranking.
"""
            )

        if isinstance(details, pd.DataFrame):
            st.subheader("Detailed retrieval results")
            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True,
            )

            failures = details[~details["hit"]]

            if not failures.empty:
                st.subheader("Questions requiring improvement")
                st.dataframe(
                    failures[
                        [
                            "question_id",
                            "question",
                            "expected_document",
                            "retrieved_documents",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

with generation_tab:
    st.markdown(
        """
## Principle

Good retrieval does not guarantee a good answer. An independent judge
model evaluates the latest GeoScope answer against the question and
retrieved context.
"""
    )

    judge_options = {
        "llama3.1:8b — recommended local judge": "llama3.1:8b",
        "qwen2.5:7b-instruct — faster, but also the generator": "qwen2.5:7b-instruct",
        "deepseek-r1:8b — slower reasoning judge": "deepseek-r1:8b",
        "OpenAI — future option": "openai",
    }

    selected_label = st.selectbox(
        "Judge model",
        list(judge_options.keys()),
    )

    judge_model = judge_options[selected_label]

    if judge_model == "llama3.1:8b":
        st.success(
            "Recommended: use Qwen for generation and Llama 3.1 as a "
            "separate local judge to reduce self-evaluation bias."
        )
    elif judge_model == "qwen2.5:7b-instruct":
        st.warning(
            "This is fast, but the same model family may be evaluating "
            "its own answer style."
        )
    elif judge_model == "deepseek-r1:8b":
        st.info(
            "Useful for difficult examples, but it can be slower on local hardware."
        )
    else:
        st.warning(
            "OpenAI support is planned but not enabled in this MVP."
        )

    question = st.session_state.get("last_question")
    answer = st.session_state.get("last_answer")
    sources = st.session_state.get("last_sources", [])
    run_id = st.session_state.get("last_run_id")
    aoi_supplied = bool(st.session_state.get("aoi_geojson"))

    if not answer:
        st.info(
            "Run a question from the Ask GeoAI page before evaluating generation."
        )
    else:
        st.markdown("### Latest answer")
        st.markdown(answer)

        with st.expander("Representative judge call"):
            st.code(
                """
evaluation = evaluate_generation(
    question=question,
    answer=answer,
    retrieved_chunks=sources,
    judge_model=judge_model,
)
""".strip(),
                language="python",
            )

        if st.button(
            "Run generation evaluation",
            type="primary",
            use_container_width=True,
            disabled=(judge_model == "openai"),
        ):
            try:
                with st.spinner(
                    f"Evaluating with {judge_model}..."
                ):
                    evaluation = evaluate_generation(
                        question=question,
                        answer=answer,
                        retrieved_chunks=sources,
                        judge_model=judge_model,
                        aoi_supplied=aoi_supplied,
                    )

                st.session_state["last_evaluation"] = evaluation

                log_generation_evaluation(
                    {
                        "run_id": run_id,
                        "evaluation_timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "judge_model": judge_model,
                        "question": question,
                        **evaluation,
                    }
                )

            except Exception as exc:
                st.error(str(exc))

        evaluation = st.session_state.get("last_evaluation")

        if evaluation:
            names = [
                "relevance",
                "groundedness",
                "completeness",
                "technical_correctness",
                "citation_quality",
                "geographic_relevance",
            ]

            columns = st.columns(3)

            for index, name in enumerate(names):
                columns[index % 3].metric(
                    name.replace("_", " ").title(),
                    evaluation.get(name),
                )

            st.metric(
                "Overall",
                evaluation.get("overall"),
            )

            st.write(evaluation.get("comment", ""))

    st.divider()
    st.subheader("User feedback")

    rating = st.radio(
        "Was this answer useful?",
        ["👍 Yes", "👎 No"],
        horizontal=True,
    )

    comment = st.text_area(
        "Optional comment",
        placeholder="What was useful or what should be improved?",
    )

    if st.button("Save feedback"):
        log_user_feedback(
            {
                "run_id": run_id,
                "feedback_timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "rating": rating,
                "comment": comment,
                "question": question,
            }
        )
        st.success("Feedback saved through dlt.")

with examples_tab:
    st.markdown(
        """
## Example questions

Use these examples for demonstrations, manual testing, and future
ground-truth expansion.
"""
    )

    examples = {
        "Agriculture": [
            "How can Sentinel-1 SAR support crop monitoring?",
            "Which datasets are suitable for wheat monitoring?",
            "How can optical and SAR time series be combined?",
            "How can crop calendars improve image selection?",
            "Which indicators may reveal crop stress?",
        ],
        "Sentinel-2": [
            "Which Sentinel-2 bands support vegetation analysis?",
            "Why are red-edge bands useful for crop monitoring?",
            "Which preprocessing steps are needed for Sentinel-2?",
            "How should cloud-contaminated observations be handled?",
        ],
        "Landsat and thermal": [
            "Which Landsat product supports land-surface temperature?",
            "How is surface reflectance different from surface temperature?",
            "How can Landsat support urban heat analysis?",
            "What are the limitations of thermal remote sensing?",
        ],
        "MODIS and phenology": [
            "What is the MOD13 product?",
            "When is MODIS preferable to Sentinel-2?",
            "How can MODIS support crop phenology analysis?",
        ],
        "GeoAI foundation models": [
            "What is a geospatial foundation model?",
            "How are foundation-model embeddings used in remote sensing?",
            "Can a foundation model replace domain validation?",
            "How can foundation models support land-cover classification?",
        ],
        "AOI-aware questions": [
            "Which available scene has the lowest cloud cover?",
            "Which datasets cover the selected AOI?",
            "Are the available scenes suitable for wheat monitoring?",
            "How should the recommendation change under persistent clouds?",
        ],
    }

    selected_group = st.selectbox(
        "Question group",
        list(examples.keys()),
    )

    for item in examples[selected_group]:
        st.code(item)
