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
    active_judge_configuration,
    compare_retrieval_approaches,
    evaluate_generation,
    load_ground_truth,
)
from src.retrieval import APPROACH_LABELS


st.set_page_config(
    page_title="Evaluation and Feedback",
    page_icon="✅",
    layout="wide",
)

apply_global_style()

st.title("✅ Step 4 — Evaluation and Feedback")

st.markdown(
    """
GeoScope now evaluates **four real retrieval pipelines** on the same
ground-truth questions:

1. Vector search
2. Query rewriting + vector search
3. Vector search + FlashRank reranking
4. Query rewriting + vector search + FlashRank reranking

Generation is evaluated independently with an LLM judge, while explicit
👍/👎 feedback is stored through the existing dlt/DuckDB workflow.
"""
)

retrieval_tab, generation_tab, examples_tab = st.tabs(
    [
        "Retrieval comparison",
        "Generation evaluation",
        "Question examples",
    ]
)

with retrieval_tab:
    try:
        ground_truth = load_ground_truth()
    except Exception as exc:
        st.error(str(exc))
        ground_truth = pd.DataFrame()

    if not ground_truth.empty:
        c1, c2, c3 = st.columns(3)

        domains = ["All"] + sorted(
            ground_truth["domain"]
            .dropna()
            .unique()
            .tolist()
        )

        with c1:
            selected_domain = st.selectbox(
                "Domain",
                domains,
            )

        with c2:
            selected_difficulty = st.selectbox(
                "Difficulty",
                [
                    "All",
                    "Beginner",
                    "Intermediate",
                    "Advanced",
                ],
            )

        with c3:
            top_k = st.slider(
                "Final top-k",
                1,
                10,
                5,
            )

        candidate_k = st.slider(
            "Candidate chunks before FlashRank",
            max(top_k, 5),
            30,
            max(15, top_k * 3),
        )

        filtered = ground_truth.copy()

        if selected_domain != "All":
            filtered = filtered[
                filtered["domain"]
                == selected_domain
            ]

        if selected_difficulty != "All":
            filtered = filtered[
                filtered["difficulty"]
                == selected_difficulty
            ]

        st.write(
            f"Questions selected: **{len(filtered)}**"
        )

        with st.expander(
            "View ground-truth questions"
        ):
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "The comparison calls the rewriting LLM for two approaches "
            "and may take several minutes on a local machine."
        )

        if st.button(
            "Compare all retrieval approaches",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Evaluating vector search, rewriting, "
                    "reranking, and the combined pipeline..."
                ):
                    metrics_df, details_df = (
                        compare_retrieval_approaches(
                            filtered,
                            top_k=top_k,
                            candidate_k=candidate_k,
                        )
                    )

                st.session_state[
                    "retrieval_comparison_metrics"
                ] = metrics_df
                st.session_state[
                    "retrieval_comparison_details"
                ] = details_df

                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                records = details_df.assign(
                    evaluation_timestamp=timestamp,
                    top_k=top_k,
                    candidate_k=candidate_k,
                ).to_dict(orient="records")

                log_retrieval_evaluation(records)

            except Exception as exc:
                st.error(str(exc))

        metrics_df = st.session_state.get(
            "retrieval_comparison_metrics"
        )
        details_df = st.session_state.get(
            "retrieval_comparison_details"
        )

        if isinstance(metrics_df, pd.DataFrame):
            st.subheader("Approach comparison")

            display_metrics = metrics_df[
                [
                    "approach_label",
                    "questions_evaluated",
                    "hit_rate",
                    "mrr",
                    "failures",
                ]
            ].copy()

            display_metrics["hit_rate"] = (
                display_metrics["hit_rate"]
                .round(3)
            )
            display_metrics["mrr"] = (
                display_metrics["mrr"]
                .round(3)
            )

            st.dataframe(
                display_metrics,
                use_container_width=True,
                hide_index=True,
            )

            best = metrics_df.iloc[0]
            st.success(
                f"Best measured approach: "
                f"**{best['approach_label']}** — "
                f"Hit Rate {best['hit_rate']:.3f}, "
                f"MRR {best['mrr']:.3f}."
            )

        if isinstance(details_df, pd.DataFrame):
            st.subheader("Detailed results")

            selected_approach = st.selectbox(
                "Inspect one approach",
                list(APPROACH_LABELS.keys()),
                format_func=lambda key: (
                    APPROACH_LABELS[key]
                ),
            )

            inspected = details_df[
                details_df["approach"]
                == selected_approach
            ]

            st.dataframe(
                inspected,
                use_container_width=True,
                hide_index=True,
            )

            failures = inspected[
                ~inspected["hit"]
            ]

            if not failures.empty:
                st.subheader(
                    "Questions requiring improvement"
                )
                st.dataframe(
                    failures[
                        [
                            "question_id",
                            "question",
                            "rewritten_query",
                            "expected_document",
                            "retrieved_documents",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

with generation_tab:
    config = active_judge_configuration()

    st.info(
        f"Active judge provider: **{config['provider']}** · "
        f"model: **{config['model']}**"
    )

    question = st.session_state.get(
        "last_augmented_question"
    ) or st.session_state.get("last_question")
    answer = st.session_state.get("last_answer")
    sources = st.session_state.get(
        "last_sources",
        [],
    )
    run_id = st.session_state.get("last_run_id")
    aoi_supplied = bool(
        st.session_state.get("aoi_geojson")
    )

    if not answer:
        st.info(
            "Run a question from Ask GeoAI before "
            "evaluating generation."
        )

    else:
        st.markdown("### Latest answer")
        st.markdown(answer)

        if st.button(
            "Run generation evaluation",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    f"Evaluating with "
                    f"{config['model']}..."
                ):
                    evaluation = evaluate_generation(
                        question=question,
                        answer=answer,
                        retrieved_chunks=sources,
                        aoi_supplied=aoi_supplied,
                    )

                st.session_state[
                    "last_evaluation"
                ] = evaluation

                log_generation_evaluation(
                    {
                        "run_id": run_id,
                        "evaluation_timestamp": (
                            datetime.now(
                                timezone.utc
                            ).isoformat()
                        ),
                        "judge_model": config["model"],
                        "judge_provider": (
                            config["provider"]
                        ),
                        "question": question,
                        **evaluation,
                    }
                )

            except Exception as exc:
                st.error(str(exc))

        evaluation = st.session_state.get(
            "last_evaluation"
        )

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
                    name.replace(
                        "_",
                        " ",
                    ).title(),
                    evaluation.get(name),
                )

            st.metric(
                "Overall",
                evaluation.get("overall"),
            )
            st.write(
                evaluation.get("comment", "")
            )

    st.divider()
    st.subheader("User feedback")

    rating = st.radio(
        "Was this answer useful?",
        ["👍 Yes", "👎 No"],
        horizontal=True,
    )

    comment = st.text_area(
        "Optional comment",
        placeholder=(
            "What was useful or what "
            "should be improved?"
        ),
    )

    if st.button("Save feedback"):
        log_user_feedback(
            {
                "run_id": run_id,
                "feedback_timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "rating": rating,
                "comment": comment,
                "question": question,
            }
        )
        st.success("Feedback saved through dlt.")

with examples_tab:
    st.markdown(
        """
Use the existing `data/evaluation_questions.csv` questions for automated
comparison. Add new rows whenever GeoScope gains another document or
domain. Each row must identify the expected source document.
"""
    )

    if 'ground_truth' in locals() and not ground_truth.empty:
        st.dataframe(
            ground_truth,
            use_container_width=True,
            hide_index=True,
        )
