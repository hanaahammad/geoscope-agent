from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.io import MemoryFile
from shapely.geometry import shape
from streamlit_folium import st_folium

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geotiff_processing import generate_product_geotiff
from src.monitoring import log_run
from src.stac_search import search_dataset


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Vegetation Condition Classification",
    page_icon="🌿",
    layout="wide",
)

apply_global_style()

st.title("🌿 Vegetation Condition Classification")
st.caption(
    "From an AOI and Sentinel-2 scene to NDVI, vegetation-condition classes, "
    "a classified GeoTIFF, statistics, and a mapped result."
)

st.markdown(
    """
This page demonstrates a **bounded geospatial execution task** rather than a
generic land-cover classifier.

The workflow is:

```text
Task request
→ inspect current AOI
→ find / reuse Sentinel-2 imagery
→ read Red + NIR
→ calculate NDVI
→ classify NDVI into vegetation-signal classes
→ calculate area statistics
→ create classified GeoTIFF
→ map + download result
→ log execution for Monitoring
```

**Important scientific limitation:** these classes describe **NDVI vegetation
signal**, not crop species, land-cover classes, or guaranteed crop health.
High NDVI often indicates stronger green vegetation signal, but interpretation
still depends on crop type, growth stage, soil, clouds, irrigation, season,
and field conditions.
"""
)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TASK = (
    "Using the current AOI, find a suitable Sentinel-2 image and create a "
    "vegetation-condition classification from NDVI."
)

CLASS_DEFINITIONS = [
    {
        "code": 1,
        "label": "Negative / non-vegetation signal",
        "min": -1.0,
        "max": 0.0,
        "display": "NDVI < 0.00",
        "rgba": (101, 139, 190, 210),
    },
    {
        "code": 2,
        "label": "Very low vegetation signal",
        "min": 0.0,
        "max": 0.2,
        "display": "0.00 ≤ NDVI < 0.20",
        "rgba": (214, 180, 107, 210),
    },
    {
        "code": 3,
        "label": "Low vegetation signal",
        "min": 0.2,
        "max": 0.4,
        "display": "0.20 ≤ NDVI < 0.40",
        "rgba": (205, 214, 99, 210),
    },
    {
        "code": 4,
        "label": "Moderate vegetation signal",
        "min": 0.4,
        "max": 0.6,
        "display": "0.40 ≤ NDVI < 0.60",
        "rgba": (111, 190, 88, 210),
    },
    {
        "code": 5,
        "label": "High vegetation signal",
        "min": 0.6,
        "max": 1.000001,
        "display": "NDVI ≥ 0.60",
        "rgba": (24, 111, 61, 220),
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def classify_ndvi_geotiff(
    ndvi_bytes: bytes,
) -> tuple[bytes, pd.DataFrame, np.ndarray]:
    """
    Convert an NDVI GeoTIFF into a categorical vegetation-condition GeoTIFF.

    Output codes:
        0 = nodata
        1..5 = vegetation-signal classes
    """
    with MemoryFile(ndvi_bytes) as src_mem:
        with src_mem.open() as src:
            ndvi = src.read(1).astype("float32")
            profile = src.profile.copy()
            nodata = src.nodata

            valid = np.isfinite(ndvi)
            if nodata is not None:
                valid &= ndvi != nodata

            classified = np.zeros(ndvi.shape, dtype=np.uint8)

            # Mutually exclusive thresholds.
            classified[valid & (ndvi < 0.0)] = 1
            classified[valid & (ndvi >= 0.0) & (ndvi < 0.2)] = 2
            classified[valid & (ndvi >= 0.2) & (ndvi < 0.4)] = 3
            classified[valid & (ndvi >= 0.4) & (ndvi < 0.6)] = 4
            classified[valid & (ndvi >= 0.6)] = 5

            valid_count = int(valid.sum())

            rows: list[dict[str, Any]] = []
            for class_def in CLASS_DEFINITIONS:
                count = int((classified == class_def["code"]).sum())
                percentage = (
                    100.0 * count / valid_count
                    if valid_count
                    else 0.0
                )
                rows.append(
                    {
                        "Class": class_def["code"],
                        "Vegetation condition": class_def["label"],
                        "NDVI range": class_def["display"],
                        "Pixels": count,
                        "Percent": percentage,
                    }
                )

            profile.update(
                dtype="uint8",
                count=1,
                nodata=0,
                compress="deflate",
            )

            with MemoryFile() as dst_mem:
                with dst_mem.open(**profile) as dst:
                    dst.write(classified, 1)
                    dst.set_band_description(
                        1,
                        "NDVI vegetation-condition class",
                    )
                    dst.update_tags(
                        classification_method="NDVI threshold classification",
                        class_1="Negative / non-vegetation signal",
                        class_2="Very low vegetation signal",
                        class_3="Low vegetation signal",
                        class_4="Moderate vegetation signal",
                        class_5="High vegetation signal",
                        note=(
                            "Vegetation-signal classes only; not crop-type or "
                            "general land-cover classification."
                        ),
                    )

                output_bytes = dst_mem.read()

    return output_bytes, pd.DataFrame(rows), classified


def classification_rgba(classified: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*classified.shape, 4), dtype=np.uint8)

    for class_def in CLASS_DEFINITIONS:
        mask = classified == class_def["code"]
        rgba[mask] = class_def["rgba"]

    rgba[classified == 0] = (0, 0, 0, 0)
    return rgba


def render_classified_map(
    aoi: dict[str, Any],
    classified: np.ndarray,
) -> None:
    geometry = shape(aoi)
    minx, miny, maxx, maxy = geometry.bounds
    centroid = geometry.centroid

    fmap = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=11,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    folium.GeoJson(
        aoi,
        name="AOI",
        style_function=lambda _: {
            "color": "#1E6F5C",
            "weight": 3,
            "fillOpacity": 0.02,
        },
    ).add_to(fmap)

    overlay_bounds = [
        [miny, minx],
        [maxy, maxx],
    ]

    ImageOverlay(
        image=classification_rgba(classified),
        bounds=overlay_bounds,
        opacity=0.82,
        name="Vegetation condition classes",
        interactive=True,
        cross_origin=False,
    ).add_to(fmap)

    legend_rows = ""
    for class_def in CLASS_DEFINITIONS:
        r, g, b, _ = class_def["rgba"]
        legend_rows += f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
          <span style="
            width:14px;height:14px;border-radius:3px;
            display:inline-block;background:rgb({r},{g},{b});">
          </span>
          <span>{class_def['label']} — {class_def['display']}</span>
        </div>
        """

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 28px;
        right: 24px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        border: 1px solid #aaa;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.22);
        font-size: 12px;
        min-width: 270px;">
      <div style="font-weight:700;font-size:13px;margin-bottom:7px;">
        NDVI vegetation-condition classes
      </div>
      {legend_rows}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.fit_bounds(
        [[miny, minx], [maxy, maxx]],
        padding=(18, 18),
        max_zoom=13,
    )

    folium.LayerControl(collapsed=False).add_to(fmap)

    st_folium(
        fmap,
        width=None,
        height=580,
        key="vegetation_condition_classification_map",
    )


def scene_label(scene: dict[str, Any], index: int) -> str:
    cloud = scene.get("cloud_cover")
    cloud_text = (
        f"{float(cloud):.1f}% cloud"
        if cloud is not None
        else "cloud n/a"
    )
    return (
        f"{index + 1}. {scene.get('date', 'date n/a')} · "
        f"{cloud_text} · {scene.get('item_id', 'scene')}"
    )


def log_classification_run(
    *,
    run_id: str,
    task_request: str,
    aoi: dict[str, Any],
    aoi_summary: str,
    scene: dict[str, Any],
    class_stats: pd.DataFrame,
    elapsed: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
) -> None:
    summary_lines = [
        (
            f"{row['Vegetation condition']}: "
            f"{float(row['Percent']):.1f}%"
        )
        for _, row in class_stats.iterrows()
    ]

    answer = (
        "Vegetation-condition classification completed from Sentinel-2 NDVI.\n\n"
        + "\n".join(summary_lines)
        + "\n\nThis is NDVI threshold classification, not crop-type or "
        "general land-cover classification."
    )

    trace = [
        {"step": 1, "name": "Inspect AOI"},
        {"step": 2, "name": "Select Sentinel-2 scene"},
        {"step": 3, "name": "Read Red + NIR"},
        {"step": 4, "name": "Calculate NDVI"},
        {"step": 5, "name": "Classify NDVI"},
        {"step": 6, "name": "Calculate class statistics"},
        {"step": 7, "name": "Create classified GeoTIFF"},
    ]

    log_run(
        run_id=run_id,
        question=task_request,
        answer=answer,
        sources=[],
        latency_seconds=elapsed,
        status="success",
        error_message="",
        application="Vegetation condition classification",
        crop="",
        season="",
        aoi_summary=aoi_summary,
        aoi_geojson=aoi,
        stac_scene_count=1,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        framework="Application geospatial workflow",
        execution_mode="Deterministic raster classification",
        model="No LLM used for pixel classification",
        prompt_id="vegetation_condition_task",
        prompt_version="1.0",
        retrieval_approach="Not applicable",
        original_query=task_request,
        rewritten_query="",
        top_k=None,
        candidate_k=None,
        chunk_count=0,
        context_characters=0,
        estimated_context_tokens=0,
        trace=trace,
        step_count=len(trace),
    )


# =============================================================================
# CURRENT CONTEXT
# =============================================================================

aoi = st.session_state.get("aoi_geojson")
aoi_summary = st.session_state.get(
    "aoi_summary",
    "Current AOI from GeoScope session.",
)

existing_scenes = st.session_state.get("stac_scenes", [])

st.markdown("## 1. Task and current context")

task_request = st.text_area(
    "Ask GeoScope to perform the bounded vegetation-classification task",
    value=DEFAULT_TASK,
    height=90,
    help=(
        "This page currently supports one bounded action: Sentinel-2 NDVI "
        "vegetation-condition classification. It does not interpret arbitrary "
        "geospatial instructions."
    ),
)

c1, c2, c3 = st.columns(3)

with c1:
    if aoi:
        st.success("AOI available")
        st.caption(aoi_summary)
    else:
        st.error("No AOI available")
        st.caption("Create an AOI on Page 2 or Page 7 first.")

with c2:
    st.metric("Scenes already in session", len(existing_scenes))

with c3:
    st.metric("Execution type", "Deterministic EO")

st.info(
    "**Why no LLM for the pixels?** The assistant coordinates the task, but "
    "the actual classification is performed deterministically from NDVI values. "
    "An LLM should not invent pixel classes."
)


# =============================================================================
# SEARCH / SCENE SELECTION
# =============================================================================

st.markdown("## 2. Imagery")

default_end = date.today()
default_start = default_end - timedelta(days=90)

s1, s2, s3, s4 = st.columns(4)

with s1:
    start_date = st.date_input(
        "Start date",
        value=default_start,
        key="vegclass_start",
    )

with s2:
    end_date = st.date_input(
        "End date",
        value=default_end,
        key="vegclass_end",
    )

with s3:
    max_cloud_cover = st.slider(
        "Maximum cloud cover (%)",
        0,
        100,
        20,
        key="vegclass_cloud",
    )

with s4:
    scene_limit = st.slider(
        "Maximum scenes",
        1,
        10,
        5,
        key="vegclass_scene_limit",
    )


sentinel_session_scenes = [
    scene
    for scene in existing_scenes
    if (
        scene.get("dataset_name") == "Sentinel-2 Level-2A"
        or str(scene.get("collection", "")).startswith("sentinel-2")
    )
]

if "vegclass_scenes" not in st.session_state:
    st.session_state["vegclass_scenes"] = sentinel_session_scenes


search_col, clear_col = st.columns([3, 1])

with search_col:
    if st.button(
        "🔎 Search Sentinel-2 for current AOI",
        type="primary",
        use_container_width=True,
        disabled=not bool(aoi),
    ):
        try:
            with st.spinner("Searching Earth Search STAC for Sentinel-2 scenes..."):
                scenes = search_dataset(
                    dataset_name="Sentinel-2 Level-2A",
                    aoi_geometry=aoi,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    max_cloud_cover=max_cloud_cover,
                    limit=scene_limit,
                )

            st.session_state["vegclass_scenes"] = scenes

            if scenes:
                st.success(f"Found {len(scenes)} Sentinel-2 scene(s).")
            else:
                st.warning(
                    "No Sentinel-2 scenes matched the current AOI/date/cloud filters."
                )
        except Exception as exc:
            st.error(f"STAC search failed: {exc}")

with clear_col:
    if st.button(
        "Clear scene list",
        use_container_width=True,
    ):
        st.session_state["vegclass_scenes"] = []
        st.rerun()


scenes = st.session_state.get("vegclass_scenes", [])

selected_scene: dict[str, Any] | None = None

if scenes:
    selected_index = st.selectbox(
        "Select scene",
        options=list(range(len(scenes))),
        format_func=lambda idx: scene_label(scenes[idx], idx),
    )
    selected_scene = scenes[selected_index]

    scene_cols = st.columns(4)
    scene_cols[0].metric("Date", selected_scene.get("date") or "n/a")
    scene_cols[1].metric(
        "Cloud cover",
        (
            f"{float(selected_scene['cloud_cover']):.1f}%"
            if selected_scene.get("cloud_cover") is not None
            else "n/a"
        ),
    )
    scene_cols[2].metric(
        "Collection",
        selected_scene.get("collection") or "n/a",
    )
    scene_cols[3].metric(
        "Assets",
        len(selected_scene.get("assets", {})),
    )
else:
    st.warning(
        "No Sentinel-2 scene is currently selected. Search for scenes above."
    )


# =============================================================================
# EXECUTION
# =============================================================================

st.markdown("## 3. Execute vegetation-condition classification")

with st.expander("Classification method", expanded=False):
    method_df = pd.DataFrame(
        [
            {
                "Class": item["code"],
                "Vegetation condition": item["label"],
                "NDVI range": item["display"],
            }
            for item in CLASS_DEFINITIONS
        ]
    )
    st.dataframe(
        method_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Thresholds are intentionally simple and transparent for the capstone "
        "demonstration. A production agronomic model would require calibration "
        "and validation for crop, season, sensor, and local conditions."
    )


if st.button(
    "🌿 Run NDVI classification",
    type="primary",
    use_container_width=True,
    disabled=(not bool(aoi) or selected_scene is None),
):
    total_steps = 7
    counter = st.empty()
    progress = st.progress(0)
    status = st.empty()

    started = time.perf_counter()
    run_id = str(uuid.uuid4())

    try:
        counter.markdown(f"### Step 1/{total_steps} — Validate AOI and scene")
        progress.progress(8)
        status.info(
            "Using the active AOI and selected Sentinel-2 scene as the "
            "geospatial execution context."
        )

        time.sleep(0.05)

        counter.markdown(f"### Step 2/{total_steps} — Read Red and NIR")
        progress.progress(20)
        status.info(
            "GeoScope resolves the Sentinel-2 Red and NIR STAC assets. "
            "These raster values are clipped to the AOI."
        )

        counter.markdown(f"### Step 3/{total_steps} — Calculate NDVI")
        progress.progress(35)
        status.info(
            "NDVI = (NIR - Red) / (NIR + Red). "
            "This is deterministic raster processing, not an LLM estimate."
        )

        ndvi_bytes, ndvi_filename, ndvi_summary = generate_product_geotiff(
            scene=selected_scene,
            aoi_geometry=aoi,
            product="NDVI",
        )

        counter.markdown(f"### Step 4/{total_steps} — Classify NDVI")
        progress.progress(55)
        status.info(
            "Each valid NDVI pixel is assigned to one transparent "
            "vegetation-signal class."
        )

        classified_bytes, class_stats, classified_array = (
            classify_ndvi_geotiff(ndvi_bytes)
        )

        counter.markdown(f"### Step 5/{total_steps} — Calculate statistics")
        progress.progress(70)
        status.info(
            "GeoScope counts valid pixels in every class and calculates "
            "their percentage of the classified AOI."
        )

        counter.markdown(f"### Step 6/{total_steps} — Build GeoTIFF + map")
        progress.progress(86)
        status.info(
            "A categorical GeoTIFF is created with class codes and metadata. "
            "The same classification is prepared for the interactive map."
        )

        item_id = selected_scene.get("item_id", "sentinel2")
        item_date = selected_scene.get("date", "unknown-date")
        classified_filename = (
            f"{item_id}_{item_date}_vegetation_condition_classification.tif"
        )

        elapsed = time.perf_counter() - started

        counter.markdown(f"### Step 7/{total_steps} — Log run")
        progress.progress(95)
        status.info(
            "GeoScope records this geospatial execution in Monitoring. "
            "No LLM is credited with the pixel classification."
        )

        log_classification_run(
            run_id=run_id,
            task_request=task_request,
            aoi=aoi,
            aoi_summary=aoi_summary,
            scene=selected_scene,
            class_stats=class_stats,
            elapsed=elapsed,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            max_cloud_cover=max_cloud_cover,
        )

        st.session_state["vegclass_result"] = {
            "run_id": run_id,
            "ndvi_bytes": ndvi_bytes,
            "ndvi_filename": ndvi_filename,
            "ndvi_summary": ndvi_summary,
            "classified_bytes": classified_bytes,
            "classified_filename": classified_filename,
            "class_stats": class_stats.to_dict(orient="records"),
            "classified_array": classified_array,
            "scene": selected_scene,
            "elapsed_seconds": elapsed,
            "task_request": task_request,
        }

        progress.progress(100)
        counter.markdown(f"### Step 7/{total_steps} — Complete ✅")
        status.success(
            f"Vegetation-condition classification completed in "
            f"{elapsed:.2f} seconds. Run ID: {run_id}"
        )

    except Exception as exc:
        elapsed = time.perf_counter() - started
        status.error(
            f"Classification failed after {elapsed:.2f} seconds: {exc}"
        )


# =============================================================================
# RESULTS
# =============================================================================

result = st.session_state.get("vegclass_result")

if result:
    st.divider()
    st.markdown("## 4. Result")

    class_stats = pd.DataFrame(result["class_stats"])

    r1, r2, r3, r4 = st.columns(4)

    high_pct = float(
        class_stats.loc[
            class_stats["Class"] == 5,
            "Percent",
        ].sum()
    )
    moderate_pct = float(
        class_stats.loc[
            class_stats["Class"] == 4,
            "Percent",
        ].sum()
    )
    low_pct = float(
        class_stats.loc[
            class_stats["Class"].isin([2, 3]),
            "Percent",
        ].sum()
    )

    r1.metric("High vegetation signal", f"{high_pct:.1f}%")
    r2.metric("Moderate signal", f"{moderate_pct:.1f}%")
    r3.metric("Low / very low signal", f"{low_pct:.1f}%")
    r4.metric("Runtime", f"{result['elapsed_seconds']:.1f}s")

    st.markdown("### Class distribution")

    display_stats = class_stats.copy()
    display_stats["Percent"] = display_stats["Percent"].map(
        lambda x: f"{float(x):.1f}%"
    )

    st.dataframe(
        display_stats[
            [
                "Class",
                "Vegetation condition",
                "NDVI range",
                "Pixels",
                "Percent",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Classified AOI map")
    render_classified_map(
        aoi,
        result["classified_array"],
    )

    st.markdown("### Download products")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download NDVI GeoTIFF",
            data=result["ndvi_bytes"],
            file_name=result["ndvi_filename"],
            mime="image/tiff",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download classified GeoTIFF",
            data=result["classified_bytes"],
            file_name=result["classified_filename"],
            mime="image/tiff",
            type="primary",
            use_container_width=True,
        )

    with st.expander("Technical metadata"):
        st.write(f"**Run ID:** {result['run_id']}")
        st.write(
            f"**Scene:** {result['scene'].get('item_id', 'n/a')}"
        )
        st.write(
            f"**Scene date:** {result['scene'].get('date', 'n/a')}"
        )
        st.json(result["ndvi_summary"])

    st.warning(
        "Interpretation guardrail: this output is a transparent NDVI threshold "
        "classification. It must not be presented as validated crop health, "
        "crop-type classification, or general land-cover mapping."
    )
