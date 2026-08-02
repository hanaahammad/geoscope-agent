from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import recent_runs

LOG_DB = PROJECT_ROOT / "logs" / "runs.duckdb"

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")

st.title("📊 Step 5 — Monitoring")
st.caption("Review runs, latency, failures, and user feedback.")

runs = recent_runs(limit=100)

if runs.empty:
    st.info("No runs have been logged yet.")
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total runs", len(runs))

    with c2:
        st.metric(
            "Average latency",
            f"{runs['latency_seconds'].mean():.2f} s",
        )

    with c3:
        success_count = int(
            (runs["status"] == "success").sum()
        )
        st.metric("Successful runs", success_count)

    st.subheader("Recent runs")
    st.dataframe(
        runs,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Latency")
    st.line_chart(
        runs.sort_values("created_at").set_index("created_at")[
            "latency_seconds"
        ]
    )

if LOG_DB.exists():
    with duckdb.connect(str(LOG_DB)) as con:
        tables = {
            row[0]
            for row in con.execute("SHOW TABLES").fetchall()
        }

        if "feedback" in tables:
            feedback = con.execute(
                """
                SELECT *
                FROM feedback
                ORDER BY created_at DESC
                """
            ).df()

            st.subheader("User feedback")
            st.dataframe(
                feedback,
                use_container_width=True,
                hide_index=True,
            )
