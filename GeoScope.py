from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="GeoScope Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_style()

st.markdown(
    """
<style>
:root {
    --card-radius: 18px;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    background:
        radial-gradient(circle at top right, rgba(88, 166, 255, 0.20), transparent 32%),
        linear-gradient(135deg, rgba(14, 116, 144, 0.18), rgba(34, 197, 94, 0.10));
    border: 1px solid rgba(120, 120, 120, 0.20);
    margin-bottom: 1.4rem;
}

.hero h1 {
    margin: 0;
    font-size: 3rem;
    line-height: 1.05;
}

.hero p {
    font-size: 1.12rem;
    margin-top: 0.8rem;
    max-width: 880px;
    opacity: 0.86;
}

.pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    margin-right: 0.4rem;
    margin-top: 0.45rem;
    border-radius: 999px;
    border: 1px solid rgba(120, 120, 120, 0.25);
    background: rgba(255, 255, 255, 0.04);
    font-size: 0.86rem;
}

.feature-card {
    min-height: 210px;
    padding: 1.25rem 1.2rem;
    border-radius: var(--card-radius);
    border: 1px solid rgba(120, 120, 120, 0.20);
    background: rgba(255, 255, 255, 0.025);
}

.feature-card h3 {
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.step-card {
    padding: 1rem 1.1rem;
    border-radius: 16px;
    border: 1px solid rgba(120, 120, 120, 0.18);
    margin-bottom: 0.75rem;
    background: rgba(255, 255, 255, 0.02);
}

.step-number {
    font-size: 0.78rem;
    font-weight: 700;
    opacity: 0.72;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.small-muted {
    opacity: 0.72;
    font-size: 0.92rem;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(120, 120, 120, 0.18);
    padding: 0.8rem 1rem;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.025);
}

.stButton > button {
    border-radius: 12px;
    font-weight: 650;
    min-height: 2.7rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120, 120, 120, 0.16);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <div class="small-muted">LLM Zoomcamp Capstone</div>
    <h1>🌍 GeoScope Agent</h1>
    <p>
        An AOI-aware GeoAI assistant that combines remote-sensing knowledge,
        live satellite metadata, local RAG, evaluation, and monitoring in one
        guided application.
    </p>
    <span class="pill">AOI-aware</span>
    <span class="pill">STAC-powered</span>
    <span class="pill">Local Ollama</span>
    <span class="pill">Evaluated RAG</span>
    <span class="pill">Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Knowledge sources", "PDF + HTML")
m2.metric("Satellite metadata", "STAC")
m3.metric("Generation", "Local LLM")
m4.metric("Evaluation", "Hit Rate + MRR + Judge")

st.markdown("## Why GeoScope?")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
<div class="feature-card">
<h3>🛰️ Remote-sensing guidance</h3>
<p>
GeoScope retrieves curated technical content about Sentinel, Landsat,
MODIS, ECOSTRESS, agriculture, and GeoAI.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
<div class="feature-card">
<h3>🗺️ Geographic context</h3>
<p>
The user draws an Area of Interest, then GeoScope narrows the answer using
location, time, cloud cover, crop, and application.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
<div class="feature-card">
<h3>✅ Transparent AI workflow</h3>
<p>
The app exposes retrieved sources, evaluation metrics, judge scores,
latency, feedback, and monitoring instead of returning an unexplained answer.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("## Who can use it?")

u1, u2 = st.columns([1.1, 0.9])

with u1:
    st.markdown(
        """
- Students learning remote sensing or GeoAI
- GIS and Earth-observation analysts
- Agricultural and environmental specialists
- Urban planners and researchers
- Data scientists exploring geospatial applications
"""
    )

with u2:
    st.info(
        "GeoScope supports dataset selection and workflow planning. "
        "It does not replace scientific validation or expert review."
    )

st.markdown("## What is remote sensing?")

st.markdown(
    """
Remote sensing means observing the Earth without direct physical contact.
Satellites record reflected or emitted energy from vegetation, water, land,
buildings, and the atmosphere. These measurements can support crop monitoring,
flood mapping, urban heat analysis, land-cover change, and environmental studies.
"""
)

with st.expander("Key acronyms"):
    st.markdown(
        """
| Acronym | Meaning |
|---|---|
| **AOI** | Area of Interest |
| **STAC** | SpatioTemporal Asset Catalog |
| **RAG** | Retrieval-Augmented Generation |
| **LLM** | Large Language Model |
| **GeoAI** | Geospatial Artificial Intelligence |
| **SAR** | Synthetic Aperture Radar |
| **MRR** | Mean Reciprocal Rank |
"""
    )

st.markdown("## Guided workflow")

steps = [
    ("Step 1", "Prepare the knowledge base", "Upload or ingest documents, then build the vector index."),
    ("Step 2", "Select AOI and search STAC", "Draw an area, choose dates, and inspect available Sentinel-2 scenes."),
    ("Step 3", "Ask GeoAI", "Choose an example or write a question and generate a sourced answer."),
    ("Step 4", "Evaluate", "Measure retrieval quality and score the final answer with an LLM judge."),
    ("Step 5", "Monitor", "Review runs, AOI context, latency, failures, and user feedback."),
]

for number, title, description in steps:
    st.markdown(
        f"""
<div class="step-card">
<div class="step-number">{number}</div>
<h3>{title}</h3>
<p>{description}</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("## Frequently asked questions")

with st.expander("What is GeoScope?"):
    st.write(
        "GeoScope is a local, AOI-aware RAG application for remote-sensing "
        "research and dataset-selection support."
    )

with st.expander("Does GeoScope download full satellite images?"):
    st.write(
        "No. The MVP searches STAC metadata and preview assets. "
        "It does not download or process full satellite scenes."
    )

with st.expander("Can I add new documents?"):
    st.write(
        "Yes. Open Step 1 — Data Preparation, upload PDF or HTML files, "
        "run ingestion, and rebuild the vector index."
    )

with st.expander("Is the answer always correct?"):
    st.write(
        "No. GeoScope shows its sources and evaluation results so the user "
        "can verify the recommendation."
    )

st.markdown("## Capstone requirement mapping")

st.markdown(
    """
| Requirement | GeoScope implementation |
|---|---|
| Dataset or API source | Curated documents and Earth Search STAC |
| Data ingestion | PDF/HTML upload, extraction, and chunking |
| Knowledge base | Chroma vector database |
| Application flow | AOI + STAC + retrieval + Ollama generation |
| Evaluation | Hit Rate, MRR, and LLM-as-a-judge |
| Interface | Streamlit multipage app |
| Feedback | User rating and comments |
| Monitoring | DuckDB and Streamlit dashboard |
"""
)
