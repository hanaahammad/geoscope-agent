from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DB = PROJECT_ROOT / "logs" / "runs.duckdb"
JUDGE_MODEL = "llama3.1:8b"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

st.set_page_config(page_title="Evaluation", page_icon="✅", layout="wide")

st.title("✅ Step 4 — Evaluation and User Feedback")
st.caption("Evaluate the latest answer and collect explicit user feedback.")

question = st.session_state.get("last_question")
answer = st.session_state.get("last_answer")
sources = st.session_state.get("last_sources", [])
run_id = st.session_state.get("last_run_id")

if not answer:
    st.info("Run a question first from the Ask GeoAI page.")
    st.stop()

st.subheader("Latest answer")
st.markdown(answer)

if st.button("Run LLM-as-a-judge evaluation", type="primary"):
    context = "\n\n".join(
        source.get("text", "")
        for source in sources[:5]
    )

    prompt = f"""
You are evaluating a GeoAI RAG answer.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

Return valid JSON only with integer scores from 1 to 5:
{{
  "relevance": 1,
  "groundedness": 1,
  "completeness": 1,
  "technical_correctness": 1,
  "citation_quality": 1,
  "overall": 1,
  "comment": "short explanation"
}}
""".strip()

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": JUDGE_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=240,
        )
        response.raise_for_status()

        evaluation = json.loads(
            response.json().get("response", "{}")
        )

        st.session_state["last_evaluation"] = evaluation

        cols = st.columns(6)
        metric_names = [
            "relevance",
            "groundedness",
            "completeness",
            "technical_correctness",
            "citation_quality",
            "overall",
        ]

        for column, name in zip(cols, metric_names):
            column.metric(name.replace("_", " ").title(), evaluation.get(name))

        st.write(evaluation.get("comment", ""))

    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("User feedback")

rating = st.radio(
    "Was this answer useful?",
    ["👍 Yes", "👎 No"],
    horizontal=True,
)

comment = st.text_area(
    "Optional feedback",
    placeholder="What was useful or what should be improved?",
)

if st.button("Save feedback"):
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(LOG_DB)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                run_id VARCHAR,
                created_at TIMESTAMP,
                rating VARCHAR,
                comment VARCHAR
            )
            """
        )

        con.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?)",
            [
                run_id,
                datetime.now(timezone.utc).replace(tzinfo=None),
                rating,
                comment,
            ],
        )

    st.success("Feedback saved.")
