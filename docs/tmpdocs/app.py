from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="GeoScope Agent",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 GeoScope Agent")
st.subheader("AOI-aware GeoAI assistant for remote-sensing dataset selection")

st.markdown(
    """
GeoScope helps users select suitable Earth-observation datasets and analysis
workflows based on:

- the selected Area of Interest;
- the application;
- crop and season;
- satellite-scene availability;
- curated technical documentation.

The application combines a local document knowledge base, live STAC metadata,
local Ollama models, retrieval evaluation, monitoring, and user feedback.
"""
)

st.info(
    "Use the pages in the left sidebar in order. "
    "The recommended walkthrough is shown below."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
### 1. Prepare the knowledge base

- Ingest PDF and HTML files
- Inspect extracted chunks
- Build or update the vector index
"""
    )

with col2:
    st.markdown(
        """
### 2. Select an AOI and search scenes

- Draw a polygon or rectangle
- Set dates and cloud threshold
- Query Sentinel-2 through STAC
"""
    )

with col3:
    st.markdown(
        """
### 3. Ask, evaluate, and monitor

- Retrieve supporting documents
- Generate an answer locally
- Evaluate quality
- Review logs and feedback
"""
    )

st.divider()

st.markdown(
    """
## Capstone requirement mapping

| Requirement | GeoScope implementation |
|---|---|
| Dataset or API-backed source | Local GeoAI documents and Earth Search STAC |
| Data ingestion | PDF and HTML extraction and chunking |
| Knowledge base | Chroma vector database |
| Application flow | AOI filtering, retrieval, prompt construction, Ollama generation |
| Evaluation | Retrieval checks and LLM-as-a-judge |
| Interface | Streamlit multipage application |
| Feedback | User rating and optional comment |
| Monitoring | DuckDB logs and Streamlit dashboard |
"""
)

st.markdown(
    """
## Example question

> Which datasets and workflow should I use to monitor wheat in the selected AOI?

Start with **1_Data_Preparation** from the sidebar.
"""
)
