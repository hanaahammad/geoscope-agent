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

from src.workflow_store import (
    get_project,
    save_snapshot,
    update_step,
)

from src.dlt_logging import (
    log_generation_evaluation,
    log_retrieval_evaluation,
)
from src.evaluation import (
    active_judge_configuration,
    compare_retrieval_approaches,
    evaluate_generation,
    load_ground_truth,
)
from src.retrieval import APPROACH_LABELS



def _score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_judge_verdict(evaluation: dict) -> tuple[str, str]:
    """
    Convert 1–5 judge scores into an operational verdict.

    PASS:
        strong overall result and no critical groundedness problem.
    NEEDS REVIEW:
        usable but one or more dimensions are borderline.
    FAIL:
        severe overall or groundedness weakness.

    This is an application rule layered on top of the raw judge scores; the
    original scores remain visible and are never replaced.
    """
    overall = _score(evaluation.get("overall"))
    groundedness = _score(evaluation.get("groundedness"))
    technical = _score(evaluation.get("technical_correctness"))
    relevance = _score(evaluation.get("relevance"))

    if (
        (overall is not None and overall <= 2.0)
        or (groundedness is not None and groundedness <= 2.0)
    ):
        if groundedness is not None and groundedness <= 2.0:
            return "FAIL", "Unsupported / weakly grounded answer"
        return "FAIL", "Low overall answer quality"

    borderline = [
        ("Groundedness", groundedness),
        ("Technical correctness", technical),
        ("Relevance", relevance),
    ]
    weak_dimensions = [
        name for name, score in borderline
        if score is not None and score <= 3.0
    ]

    if (
        (overall is not None and overall < 4.0)
        or weak_dimensions
    ):
        category = (
            "Borderline: " + ", ".join(weak_dimensions)
            if weak_dimensions
            else "Borderline overall quality"
        )
        return "NEEDS REVIEW", category

    return "PASS", "No critical quality issue detected"


st.set_page_config(
    page_title="Evaluation and Feedback",
    page_icon="✅",
    layout="wide",
)

apply_global_style()

def sync_evaluation_to_active_project() -> None:
    """
    Immediately reflect evaluation progress in an active GeoScope project.

    This avoids requiring the user to manually save Page 7 before the
    Evaluation milestone is shown as completed.
    """
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        return

    project = get_project(project_id)
    if not project:
        return

    # Keep the existing saved project snapshot and update it with the
    # evaluation-related state currently available in Streamlit.
    snapshot = dict(project.get("snapshot") or {})

    keys_to_sync = [
        "last_evaluation",
        "retrieval_comparison_metrics",
        "retrieval_comparison_details",
        "last_run_id",
        "last_question",
        "last_augmented_question",
        "last_answer",
        "last_sources",
        "aoi_geojson",
        "aoi_summary",
        "stac_scenes",
    ]

    for key in keys_to_sync:
        if key in st.session_state:
            value = st.session_state[key]

            # Pandas DataFrames cannot be stored directly in the JSON snapshot.
            if isinstance(value, pd.DataFrame):
                snapshot[key] = value.to_dict(orient="records")
            else:
                snapshot[key] = value

    snapshot["evaluation_completed"] = True

    save_snapshot(project_id, snapshot)
    update_step(
        project_id,
        6,
        "COMPLETED",
        "Evaluation completed from Page 4.",
    )



st.title("✅ Step 4 — Evaluation and Feedback")

st.markdown(
    """
GeoScope evaluates **retrieval first** and **generation second**.

The retrieval experiment compares four real ways of finding evidence for the
same ground-truth questions. The generation experiment then asks an independent
LLM judge to evaluate the answer produced from that evidence.

### What does *query rewriting* mean?

Query rewriting happens **before vector retrieval**. The LLM does not rewrite
the final answer. It reformulates the user's question into a search-oriented
query that may match the indexed technical documents more effectively.

```text
Original user question
        ↓
Optional query rewrite
        ↓
Vector search in Chroma
        ↓
Optional FlashRank reranking
        ↓
Retrieved evidence
        ↓
Answer generation
```

A rewrite is **not automatically better**. A precise question may already be a
good retrieval query, while a short or conversational question may benefit from
additional technical wording. That is why GeoScope measures both alternatives.
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
    st.markdown("### 🔎 Retrieval experiment")

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**1 · Vector**")
        st.code("Original question → Chroma → top chunks", language=None)

        st.markdown("**2 · Rewrite**")
        st.code(
            "Original question → LLM rewrite → Chroma → top chunks",
            language=None,
        )

    with p2:
        st.markdown("**3 · Rerank**")
        st.code(
            "Original question → Chroma candidates → FlashRank → top chunks",
            language=None,
        )

        st.markdown("**4 · Rewrite + Rerank**")
        st.code(
            "Question → LLM rewrite → Chroma candidates → FlashRank → top chunks",
            language=None,
        )

    with st.expander("💡 Example — why might a query be rewritten?"):
        st.markdown(
            """
**User question**

`How can I detect crop stress?`

**Possible retrieval-oriented rewrite**

`Earth Observation crop stress detection using Sentinel-2 vegetation and moisture indices such as NDVI and NDMI`

The rewritten form is used **only to search the knowledge base**. The original
user intent remains the basis for the final answer.

The experiment tells us whether this reformulation actually helps retrieval.
"""
        )

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

        selected_count = len(filtered)

        st.write(
            f"Questions selected: **{selected_count}**"
        )

        with st.expander(
            "View ground-truth questions"
        ):
            if filtered.empty:
                st.caption(
                    "No ground-truth question matches the current Domain + "
                    "Difficulty combination."
                )
            else:
                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True,
                )

        if filtered.empty:
            st.warning(
                "No questions match these filters. Change **Domain** or "
                "**Difficulty** before running the comparison."
            )
        else:
            st.info(
                "The **Rewrite** and **Rewrite + Rerank** approaches call the "
                "rewriting LLM before retrieval. On a local model this can make "
                "the four-way comparison take several minutes."
            )

        run_comparison = st.button(
            "Compare all retrieval approaches",
            type="primary",
            use_container_width=True,
            disabled=filtered.empty,
        )

        if run_comparison:
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

                if (
                    isinstance(details_df, pd.DataFrame)
                    and not details_df.empty
                ):
                    records = details_df.assign(
                        evaluation_timestamp=timestamp,
                        top_k=top_k,
                        candidate_k=candidate_k,
                    ).to_dict(orient="records")

                    log_retrieval_evaluation(records)

                st.session_state["evaluation_completed"] = True
                sync_evaluation_to_active_project()

            except Exception as exc:
                st.error(str(exc))

        metrics_df = st.session_state.get(
            "retrieval_comparison_metrics"
        )
        details_df = st.session_state.get(
            "retrieval_comparison_details"
        )

        metrics_required = {
            "approach_label",
            "questions_evaluated",
            "hit_rate",
            "mrr",
            "failures",
        }

        if (
            isinstance(metrics_df, pd.DataFrame)
            and not metrics_df.empty
            and metrics_required.issubset(metrics_df.columns)
        ):
            st.subheader("Approach comparison")
            st.caption(
                "**Hit Rate** asks whether the expected source was retrieved. "
                "**MRR** also rewards retrieving it near the top of the ranking."
            )

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
        elif isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            st.warning(
                "Stored retrieval metrics use an older/incompatible result "
                "schema. Run the comparison again to refresh them."
            )

        details_required = {
            "approach",
            "hit",
            "question_id",
            "question",
        }

        if (
            isinstance(details_df, pd.DataFrame)
            and not details_df.empty
            and details_required.issubset(details_df.columns)
        ):
            st.subheader("Detailed results")

            available_approaches = [
                key
                for key in APPROACH_LABELS.keys()
                if key in set(details_df["approach"].dropna())
            ]

            if not available_approaches:
                st.info(
                    "No approach details are available for the current result."
                )
            else:
                selected_approach = st.selectbox(
                    "Inspect one approach",
                    available_approaches,
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
                    ~inspected["hit"].fillna(False).astype(bool)
                ]

                if not failures.empty:
                    st.subheader(
                        "Questions requiring improvement"
                    )
                    st.caption(
                        "These are **retrieval failures** for the selected "
                        "approach: the expected source document was not found "
                        "in the final retrieved set. This does not automatically "
                        "mean the rewritten query itself is wrong."
                    )

                    preferred_columns = [
                        "question_id",
                        "question",
                        "rewritten_query",
                        "expected_document",
                        "retrieved_documents",
                    ]
                    visible_columns = [
                        col
                        for col in preferred_columns
                        if col in failures.columns
                    ]

                    st.dataframe(
                        failures[visible_columns],
                        use_container_width=True,
                        hide_index=True,
                    )

                    if "rewritten_query" in failures.columns:
                        changed = (
                            failures["rewritten_query"]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            != failures["question"]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                        )

                        st.caption(
                            f"Query rewrite changed the wording for "
                            f"**{int(changed.sum())} of {len(failures)}** "
                            "failed questions in this view."
                        )

        elif isinstance(details_df, pd.DataFrame) and not details_df.empty:
            st.warning(
                "Stored detailed results use an older/incompatible schema "
                "(for example, the `approach` column is missing). "
                "Run the retrieval comparison again to refresh the session."
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

                judge_verdict, failure_category = (
                    classify_judge_verdict(evaluation)
                )

                evaluation["judge_verdict"] = judge_verdict
                evaluation["failure_category"] = failure_category

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

                st.session_state["evaluation_completed"] = True
                sync_evaluation_to_active_project()

            except Exception as exc:
                st.error(str(exc))

        evaluation = st.session_state.get(
            "last_evaluation"
        )

        if evaluation:
            judge_verdict = evaluation.get("judge_verdict")
            failure_category = evaluation.get("failure_category")

            if judge_verdict:
                if judge_verdict == "PASS":
                    st.success(
                        f"Judge verdict: **PASS** · {failure_category}"
                    )
                elif judge_verdict == "NEEDS REVIEW":
                    st.warning(
                        f"Judge verdict: **NEEDS REVIEW** · {failure_category}"
                    )
                else:
                    st.error(
                        f"Judge verdict: **FAIL** · {failure_category}"
                    )

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
    st.info(
        "Human 👍/👎 feedback is collected directly in Step 3 — Ask GeoAI. "
        "This page is reserved for structured retrieval evaluation and "
        "LLM-as-a-judge generation evaluation."
    )

with examples_tab:
    st.markdown(
        """
The retrieval benchmark is based on `data/evaluation_questions.csv`.

Each row represents a **ground-truth retrieval test**:

- `question` — the question sent to the retrieval pipeline;
- `expected_document` — the source that should be found;
- `domain` and `difficulty` — optional filters for targeted experiments.

For rewrite-based approaches, GeoScope first generates a
`rewritten_query`. The benchmark then checks whether that reformulation
helped the retrieval pipeline find the expected document.

A failed row means **the expected document was not present in the final
retrieved set**. It should be investigated as a retrieval/configuration
issue rather than automatically labelled a "bad rewrite".
"""
    )

    if 'ground_truth' in locals() and not ground_truth.empty:
        st.dataframe(
            ground_truth,
            use_container_width=True,
            hide_index=True,
        )
