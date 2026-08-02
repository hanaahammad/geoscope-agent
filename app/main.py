


from __future__ import annotations

import sys
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_vector_index import build_vector_index
from src.generation import generate_answer
from src.ingest_documents import ingest_documents
from src.monitoring import log_run, recent_runs
from src.retrieval import search_documents
from src.stac_search import search_sentinel2


st.set_page_config(
    page_title="GeoScope Agent",
    layout="wide",
)

st.title("GeoScope Agent")
st.caption(
    "AOI-aware GeoAI assistant using local Ollama models, "
    "Chroma retrieval, STAC metadata, and Streamlit"
)


def extract_aoi(map_output: dict | None) -> dict | None:
    """Return the latest drawn GeoJSON geometry."""
    if not map_output:
        return None

    drawing = map_output.get("last_active_drawing")

    if not drawing:
        return None

    return drawing.get("geometry")


def build_augmented_question(
    question: str,
    application: str,
    crop: str,
    season: str,
    scenes: list[dict],
) -> str:
    """Add user filters and STAC scene metadata to the retrieval query."""
    scene_summary = ""

    if scenes:
        lines = []

        for scene in scenes[:5]:
            lines.append(
                f"- {scene.get('item_id')} | "
                f"date={scene.get('date')} | "
                f"cloud_cover={scene.get('cloud_cover')} | "
                f"collection={scene.get('collection')}"
            )

        scene_summary = "\nAvailable STAC scenes:\n" + "\n".join(lines)

    return f"""
User question:
{question}

Selected context:
- Application: {application}
- Crop: {crop}
- Season: {season}
{scene_summary}
""".strip()


tab_ask, tab_kb, tab_monitoring = st.tabs(
    [
        "Ask GeoAI",
        "Knowledge Base",
        "Monitoring",
    ]
)


with tab_ask:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("1. Select Area of Interest")

        fmap = folium.Map(
            location=[30.5, 30.8],
            zoom_start=7,
            control_scale=True,
        )

        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
            },
            edit_options={
                "edit": True,
                "remove": True,
            },
        ).add_to(fmap)

        map_output = st_folium(
            fmap,
            width=None,
            height=480,
            returned_objects=["last_active_drawing"],
            key="aoi_map",
        )

        aoi_geojson = extract_aoi(map_output)

        if aoi_geojson:
            st.success("AOI captured.")
            with st.expander("View AOI GeoJSON"):
                st.json(aoi_geojson)
        else:
            st.info("Draw a polygon or rectangle on the map.")

    with right:
        st.subheader("2. Define the analysis")

        application = st.selectbox(
            "Application",
            [
                "Crop monitoring",
                "Crop classification",
                "Urban heat",
                "Land-cover change",
                "Flood assessment",
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

        default_end = date.today()
        default_start = default_end - timedelta(days=60)

        start_date = st.date_input(
            "Start date",
            value=default_start,
        )

        end_date = st.date_input(
            "End date",
            value=default_end,
        )

        max_cloud_cover = st.slider(
            "Maximum cloud cover (%)",
            min_value=0,
            max_value=100,
            value=20,
        )

        scene_limit = st.slider(
            "Maximum STAC scenes",
            min_value=1,
            max_value=20,
            value=5,
        )

        top_k = st.slider(
            "Retrieved document chunks",
            min_value=1,
            max_value=10,
            value=5,
        )

        question = st.text_area(
            "Question",
            value=(
                "Which datasets and workflow should I use "
                "to monitor wheat in this selected area?"
            ),
            height=120,
        )

    st.divider()

    if st.button(
        "Run GeoScope",
        type="primary",
        use_container_width=True,
    ):
        run_id = str(uuid.uuid4())
        started = time.perf_counter()

        answer = ""
        sources = []
        scenes = []
        status = "success"
        error_message = ""

        try:
            if not question.strip():
                raise ValueError("Please enter a question.")

            if not aoi_geojson:
                raise ValueError(
                    "Please draw an AOI before running the analysis."
                )

            with st.spinner("Searching Sentinel-2 STAC metadata..."):
                scenes = search_sentinel2(
                    aoi_geometry=aoi_geojson,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    max_cloud_cover=max_cloud_cover,
                    limit=scene_limit,
                )

            augmented_question = build_augmented_question(
                question=question,
                application=application,
                crop=crop,
                season=season,
                scenes=scenes,
            )

            with st.spinner("Retrieving relevant documents..."):
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
        )

        if status == "success":
            st.subheader("Answer")
            st.markdown(answer)
            st.caption(
                f"Run ID: {run_id} · "
                f"Latency: {latency:.2f} seconds"
            )

            st.subheader("Matching Sentinel-2 scenes")

            if scenes:
                scene_rows = [
                    {
                        "item_id": scene.get("item_id"),
                        "date": scene.get("date"),
                        "cloud_cover": scene.get("cloud_cover"),
                        "collection": scene.get("collection"),
                    }
                    for scene in scenes
                ]

                st.dataframe(
                    pd.DataFrame(scene_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                previews = [
                    scene
                    for scene in scenes
                    if scene.get("thumbnail")
                ][:3]

                for scene in previews:
                    st.image(
                        scene["thumbnail"],
                        caption=(
                            f"{scene.get('date')} · "
                            f"Cloud cover: "
                            f"{scene.get('cloud_cover')}%"
                        ),
                    )
            else:
                st.warning(
                    "No matching Sentinel-2 scenes were found "
                    "for the selected filters."
                )

            st.subheader("Retrieved knowledge sources")

            source_rows = [
                {
                    "file": item.get("file_name"),
                    "page": item.get("page_number"),
                    "distance": round(
                        item.get("distance", 0),
                        4,
                    ),
                    "preview": item.get("text", "")[:300],
                }
                for item in sources
            ]

            st.dataframe(
                pd.DataFrame(source_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.error(error_message)


with tab_kb:
    st.subheader("Knowledge-base administration")

    st.warning(
        "Run these actions only when documents change. "
        "Vector indexing can take several minutes."
    )

    if st.button(
        "1. Ingest PDF and HTML documents",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Extracting and chunking documents..."
            ):
                result = ingest_documents()

            st.success(
                f"Processed {result['documents_processed']} "
                f"documents and created "
                f"{result['chunks_created']} chunks."
            )
            st.json(result)

        except Exception as exc:
            st.error(str(exc))

    if st.button(
        "2. Build or update vector index",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Creating embeddings with Ollama..."
            ):
                result = build_vector_index()

            st.success(
                f"Indexed {result['records']} chunks."
            )
            st.json(result)

        except Exception as exc:
            st.error(str(exc))


with tab_monitoring:
    st.subheader("Recent runs")

    try:
        runs = recent_runs()

        if runs.empty:
            st.info("No runs have been logged yet.")
        else:
            st.dataframe(
                runs,
                use_container_width=True,
                hide_index=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total runs",
                    len(runs),
                )

            with col2:
                st.metric(
                    "Average latency",
                    f"{runs['latency_seconds'].mean():.2f} s",
                )

            with col3:
                successful = (
                    runs["status"] == "success"
                ).sum()

                st.metric(
                    "Successful runs",
                    int(successful),
                )

    except Exception as exc:
        st.error(str(exc))
