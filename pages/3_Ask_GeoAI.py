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

from src.generation import generate_answer
from src.monitoring import log_run
from src.retrieval import search_documents


st.set_page_config(
    page_title="Ask GeoAI",
    page_icon="🤖",
    layout="wide",
)

apply_global_style()

st.title("🤖 Step 3 — Ask GeoAI")
st.caption(
    "Combine AOI context, STAC metadata, document retrieval, "
    "and local generation."
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

st.divider()

left, right = st.columns([0.75, 1.25])

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

    top_k = st.slider(
        "Retrieved document chunks",
        1,
        10,
        5,
    )

with right:
    question = st.text_area(
        "Question",
        key="ask_question",
        height=170,
    )

st.write(
    f"AOI available: **{'Yes' if aoi_geojson else 'No'}** · "
    f"STAC scenes available: **{len(scenes)}**"
)

if aoi_geojson:
    st.info(f"**AOI used in this question:** {aoi_summary}")

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

        scene_lines = []

        for scene in scenes[:5]:
            scene_lines.append(
                f"- {scene.get('item_id')} | "
                f"date={scene.get('date')} | "
                f"cloud_cover={scene.get('cloud_cover')}"
            )

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

Available Sentinel-2 scenes:
{scene_summary}
""".strip()

        with st.spinner("Retrieving relevant knowledge..."):
            sources = search_documents(
                augmented_question,
                top_k=top_k,
            )

        with st.spinner("Generating the answer locally..."):
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
        st.session_state["last_answer"] = answer
        st.session_state["last_sources"] = sources

        st.subheader("Answer")
        st.markdown(answer)
        st.caption(
            f"Run ID: {run_id} · "
            f"Latency: {latency:.2f} seconds"
        )

        st.subheader("Retrieved sources")

        rows = [
            {
                "file": source.get("file_name"),
                "page": source.get("page_number"),
                "distance": round(
                    source.get("distance", 0),
                    4,
                ),
                "preview": source.get("text", "")[:300],
            }
            for source in sources
        ]

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.error(error_message)
