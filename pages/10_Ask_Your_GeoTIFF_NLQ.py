from __future__ import annotations

import json
import re
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
from src.llm_provider import generate_text, get_generation_model
from src.stac_search import search_dataset


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Ask Your GeoTIFF",
    page_icon="🗺️",
    layout="wide",
)

apply_global_style()

st.title("🗺️ Ask Your GeoTIFF")
st.caption(
    "Generate or load a GeoTIFF, inspect the raster with deterministic tools, "
    "and ask natural-language questions grounded in the actual pixel values."
)

st.markdown(
    """
This page demonstrates **natural-language querying of raster data**.

The workflow is:

```text
GeoTIFF
→ inspect metadata and valid pixels
→ calculate deterministic raster statistics
→ user asks a natural-language question
→ GeoScope selects the appropriate raster operation
→ tool result
→ grounded natural-language answer
```

The page also keeps the Sentinel-2 NDVI vegetation-condition classification
as one available raster analysis.

**Important:** the LLM does not calculate pixel statistics and does not
classify pixels. Numerical results are produced by deterministic raster code;
the LLM is used only to explain those computed results in natural language.
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
# GEOTIFF NLQ HELPERS
# =============================================================================

RASTER_QA_INSTRUCTIONS = """
You are GeoScope's raster-results explainer.

Answer the user's question using ONLY the deterministic tool result supplied
to you. Do not invent pixel values, locations, percentages, dates, or
scientific conclusions. If the tool result says the question is unsupported,
say so clearly.

Keep the answer concise and suitable for an Earth Observation analyst.
Distinguish measured raster statistics from interpretation.
""".strip()


def inspect_numeric_geotiff(raster_bytes: bytes) -> dict[str, Any]:
    """Read band 1 and return deterministic metadata/statistics."""
    with MemoryFile(raster_bytes) as mem:
        with mem.open() as src:
            arr = src.read(1).astype("float64")
            nodata = src.nodata

            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata

            values = arr[valid]
            if values.size == 0:
                raise ValueError("The GeoTIFF contains no valid pixels in band 1.")

            bounds = src.bounds
            transform = src.transform

            return {
                "array": arr,
                "valid_mask": valid,
                "values": values,
                "metadata": {
                    "driver": src.driver,
                    "width": int(src.width),
                    "height": int(src.height),
                    "count": int(src.count),
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs) if src.crs else "Not recorded",
                    "nodata": nodata,
                    "resolution_x": abs(float(transform.a)),
                    "resolution_y": abs(float(transform.e)),
                    "bounds": {
                        "left": float(bounds.left),
                        "bottom": float(bounds.bottom),
                        "right": float(bounds.right),
                        "top": float(bounds.top),
                    },
                },
                "statistics": {
                    "valid_pixels": int(values.size),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std_dev": float(np.std(values)),
                },
            }


def _find_number(question: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", question)
    return float(match.group(0)) if match else None


def route_raster_question(
    question: str,
    raster_info: dict[str, Any],
    *,
    treat_as_ndvi: bool,
) -> tuple[str, dict[str, Any]]:
    """
    Map common natural-language raster questions to deterministic operations.

    The router is intentionally bounded: it supports statistics, metadata,
    thresholds, NDVI class distribution, and concise raster summaries.
    """
    q = question.strip().lower()
    stats = raster_info["statistics"]
    meta = raster_info["metadata"]
    values = raster_info["values"]

    # Metadata / geometry
    if any(term in q for term in ["metadata", "crs", "projection", "resolution", "pixel size", "dimensions", "size of the raster", "bounds"]):
        return "metadata", {
            "operation": "raster_metadata",
            **meta,
        }

    # Threshold queries
    threshold = _find_number(q)
    if threshold is not None and any(term in q for term in ["above", "greater than", "over", ">"]):
        count = int(np.sum(values > threshold))
        pct = 100.0 * count / values.size
        return "threshold_above", {
            "operation": "percentage_above_threshold",
            "threshold": threshold,
            "matching_pixels": count,
            "valid_pixels": int(values.size),
            "percentage": pct,
        }

    if threshold is not None and any(term in q for term in ["below", "less than", "under", "<"]):
        count = int(np.sum(values < threshold))
        pct = 100.0 * count / values.size
        return "threshold_below", {
            "operation": "percentage_below_threshold",
            "threshold": threshold,
            "matching_pixels": count,
            "valid_pixels": int(values.size),
            "percentage": pct,
        }

    # NDVI class questions
    if treat_as_ndvi and any(
        term in q
        for term in [
            "vegetation condition",
            "vegetation signal",
            "high vegetation",
            "moderate vegetation",
            "low vegetation",
            "class distribution",
            "classified",
            "classification",
        ]
    ):
        valid = np.isfinite(raster_info["array"])
        if meta["nodata"] is not None:
            valid &= raster_info["array"] != meta["nodata"]
        arr = raster_info["array"]

        class_counts = {
            "negative_non_vegetation": int(np.sum(valid & (arr < 0.0))),
            "very_low": int(np.sum(valid & (arr >= 0.0) & (arr < 0.2))),
            "low": int(np.sum(valid & (arr >= 0.2) & (arr < 0.4))),
            "moderate": int(np.sum(valid & (arr >= 0.4) & (arr < 0.6))),
            "high": int(np.sum(valid & (arr >= 0.6))),
        }
        total = int(np.sum(valid))
        percentages = {
            key: (100.0 * count / total if total else 0.0)
            for key, count in class_counts.items()
        }
        return "ndvi_classes", {
            "operation": "ndvi_class_distribution",
            "valid_pixels": total,
            "class_counts": class_counts,
            "class_percentages": percentages,
            "note": (
                "These are transparent NDVI vegetation-signal thresholds, "
                "not validated crop-health or crop-type classes."
            ),
        }

    # Specific statistics
    if "mean" in q or "average" in q:
        return "mean", {"operation": "raster_mean", "mean": stats["mean"], "valid_pixels": stats["valid_pixels"]}

    if "median" in q:
        return "median", {"operation": "raster_median", "median": stats["median"], "valid_pixels": stats["valid_pixels"]}

    if "standard deviation" in q or "std" in q or "variability" in q:
        return "std", {"operation": "raster_standard_deviation", "std_dev": stats["std_dev"], "valid_pixels": stats["valid_pixels"]}

    if ("minimum" in q or "min " in q or q.startswith("min")) and ("maximum" in q or "max " in q or "range" in q):
        return "range", {
            "operation": "raster_range",
            "minimum": stats["minimum"],
            "maximum": stats["maximum"],
        }

    if "minimum" in q or "lowest" in q or "min value" in q:
        return "minimum", {"operation": "raster_minimum", "minimum": stats["minimum"]}

    if "maximum" in q or "highest" in q or "max value" in q:
        return "maximum", {"operation": "raster_maximum", "maximum": stats["maximum"]}

    if "pixel" in q and any(term in q for term in ["how many", "count", "valid"]):
        return "valid_pixels", {
            "operation": "valid_pixel_count",
            "valid_pixels": stats["valid_pixels"],
            "width": meta["width"],
            "height": meta["height"],
        }

    # A general summary is still deterministic: provide a compact statistics bundle.
    if any(term in q for term in ["summary", "summarize", "describe", "overall", "what does this raster show", "tell me about"]):
        result = {
            "operation": "raster_summary",
            "statistics": stats,
            "metadata": meta,
        }
        if treat_as_ndvi:
            result["interpretation_guardrail"] = (
                "Raster is being treated as NDVI. Numerical statistics are measured; "
                "vegetation interpretation remains contextual."
            )
        return "summary", result

    # Spatial 'where' questions need more than global band statistics.
    if any(term in q for term in ["where", "which location", "which part", "north", "south", "east", "west"]):
        return "unsupported_spatial", {
            "operation": "unsupported_question",
            "message": (
                "This version supports global raster statistics, thresholds, metadata, "
                "and NDVI class distribution. It does not yet compute zonal or directional "
                "spatial summaries for arbitrary 'where' questions."
            ),
        }

    return "fallback_summary", {
        "operation": "raster_summary",
        "statistics": stats,
        "metadata": meta,
        "message": (
            "The question did not map to a more specific bounded raster operation, "
            "so GeoScope returned the measured raster summary."
        ),
    }


def explain_raster_tool_result(question: str, tool_result: dict[str, Any]) -> str:
    """Use the LLM only to verbalize deterministic raster-tool output."""
    prompt = f"""
USER QUESTION:
{question}

DETERMINISTIC RASTER TOOL RESULT:
{json.dumps(tool_result, indent=2, default=str)}

Answer the question directly from this result.
""".strip()

    return generate_text(
        instructions=RASTER_QA_INSTRUCTIONS,
        prompt=prompt,
        model=get_generation_model(),
    )


def log_raster_nlq_run(
    *,
    run_id: str,
    question: str,
    answer: str,
    tool_name: str,
    tool_result: dict[str, Any],
    elapsed: float,
    raster_name: str,
) -> None:
    trace = [
        {"step": 1, "name": "Read GeoTIFF band 1"},
        {"step": 2, "name": "Route natural-language raster question"},
        {"step": 3, "name": tool_name},
        {"step": 4, "name": "Generate grounded explanation from tool result"},
    ]

    log_run(
        run_id=run_id,
        question=question,
        answer=answer,
        sources=[],
        latency_seconds=elapsed,
        status="success",
        error_message="",
        application="Natural-language GeoTIFF analysis",
        crop="",
        season="",
        aoi_summary=st.session_state.get("aoi_summary", ""),
        aoi_geojson=st.session_state.get("aoi_geojson"),
        stac_scene_count=0,
        start_date="",
        end_date="",
        max_cloud_cover=None,
        framework="Application geospatial workflow",
        execution_mode="Natural-language raster query",
        model=get_generation_model(),
        prompt_id="geotiff_nlq_grounded_explanation",
        prompt_version="1.0",
        retrieval_approach="Raster tool routing",
        original_query=question,
        rewritten_query="",
        top_k=None,
        candidate_k=None,
        chunk_count=0,
        context_characters=len(json.dumps(tool_result, default=str)),
        estimated_context_tokens=max(1, len(json.dumps(tool_result, default=str)) // 4),
        trace=trace,
        tool_calls=[{"tool": tool_name, "raster": raster_name}],
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

st.markdown("## 1. Prepare an NDVI GeoTIFF")

task_request = st.text_area(
    "Optional task description",
    value=DEFAULT_TASK,
    height=90,
    help=(
        "This text documents the raster-generation task. Natural-language GeoTIFF questions are available below after a raster is ready."
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

st.markdown("## 3. Generate NDVI + vegetation-condition classification")

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
    st.markdown("## 4. Raster result")

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


# =============================================================================
# NATURAL-LANGUAGE GEOTIFF QUERY
# =============================================================================

st.divider()
st.markdown("## 5. Ask Your GeoTIFF in Natural Language")

st.markdown(
    """
Ask questions about the **actual raster values**, for example:

- `What is the average NDVI?`
- `What are the minimum and maximum values?`
- `What percentage of the AOI has NDVI above 0.6?`
- `How much of the raster has low vegetation signal?`
- `How many valid pixels are there?`
- `What is the CRS and pixel resolution?`
- `Summarize this raster.`

GeoScope first executes a deterministic raster operation and only then asks
the LLM to explain the computed result.
"""
)

source_options = ["Generated NDVI from this page", "Upload a GeoTIFF"]
source_choice = st.radio(
    "1. Choose raster source",
    source_options,
    horizontal=True,
    key="geotiff_nlq_source",
)

qa_bytes: bytes | None = None
qa_name = ""
treat_as_ndvi = False

if source_choice == "Generated NDVI from this page":
    if result:
        qa_bytes = result["ndvi_bytes"]
        qa_name = result["ndvi_filename"]
        treat_as_ndvi = True
        st.success(f"Raster ready: {qa_name}")
    else:
        st.info(
            "No NDVI raster is ready yet. Generate the NDVI product above, "
            "or switch to **Upload a GeoTIFF**."
        )
else:
    uploaded_raster = st.file_uploader(
        "Upload GeoTIFF",
        type=["tif", "tiff"],
        key="geotiff_nlq_upload",
    )
    if uploaded_raster is not None:
        qa_bytes = uploaded_raster.getvalue()
        qa_name = uploaded_raster.name
        treat_as_ndvi = st.checkbox(
            "Treat this raster as NDVI",
            value=False,
            help=(
                "Enable only if band 1 actually contains NDVI values. "
                "This unlocks vegetation-signal class questions."
            ),
        )
        st.success(f"Raster ready: {qa_name}")
    else:
        st.info("Upload a `.tif` or `.tiff` file to enable raster analysis.")

raster_info = None
raster_ready = qa_bytes is not None

st.markdown("### 2. Inspect raster")

if raster_ready:
    try:
        raster_info = inspect_numeric_geotiff(qa_bytes)
        m = raster_info["metadata"]
        s = raster_info["statistics"]

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Valid pixels", f"{s['valid_pixels']:,}")
        i2.metric("Mean", f"{s['mean']:.4f}")
        i3.metric("Minimum", f"{s['minimum']:.4f}")
        i4.metric("Maximum", f"{s['maximum']:.4f}")

        with st.expander("Show raster metadata and statistics", expanded=False):
            st.json(
                {
                    "filename": qa_name,
                    "metadata": m,
                    "statistics": s,
                    "treated_as_ndvi": treat_as_ndvi,
                }
            )
    except Exception as exc:
        st.error(f"Could not inspect the GeoTIFF: {exc}")
        raster_info = None
        raster_ready = False
else:
    st.caption(
        "Raster inspection will appear here after a GeoTIFF is generated or uploaded."
    )

st.markdown("### 3. Ask a natural-language question")

examples = [
    "What is the average NDVI?" if treat_as_ndvi else "What is the average raster value?",
    "What are the minimum and maximum values?",
    "What percentage of the AOI has NDVI above 0.6?" if treat_as_ndvi else "What percentage of pixels are above 0.5?",
    "How many valid pixels are there?",
    "What is the CRS and pixel resolution?",
    "Summarize this raster.",
]

if treat_as_ndvi:
    examples.insert(3, "How much of the raster has high vegetation signal?")

def _copy_example_to_question() -> None:
    selected = st.session_state.get("geotiff_nlq_example", "Write my own question")
    if selected != "Write my own question":
        st.session_state["geotiff_nlq_question"] = selected


selected_example = st.selectbox(
    "Example question",
    ["Write my own question"] + examples,
    key="geotiff_nlq_example",
    disabled=not raster_ready,
    on_change=_copy_example_to_question,
)

# Initialize the question field once the raster becomes ready.
if "geotiff_nlq_question" not in st.session_state:
    st.session_state["geotiff_nlq_question"] = (
        "" if selected_example == "Write my own question" else selected_example
    )

question = st.text_input(
    "Natural-language question about the GeoTIFF",
    key="geotiff_nlq_question",
    placeholder="e.g. What percentage of the AOI has NDVI above 0.6?",
    disabled=not raster_ready,
)

if not raster_ready:
    st.caption("Load or generate a GeoTIFF to enable the question field and analysis.")

ask_disabled = (not raster_ready) or (not question.strip())

if st.button(
    "💬 Ask GeoTIFF",
    type="primary",
    use_container_width=True,
    disabled=ask_disabled,
):
    started_qa = time.perf_counter()
    qa_run_id = str(uuid.uuid4())

    try:
        tool_name, tool_result = route_raster_question(
            question,
            raster_info,
            treat_as_ndvi=treat_as_ndvi,
        )

        with st.spinner(
            f"Running raster tool `{tool_name}` and grounding the answer..."
        ):
            answer = explain_raster_tool_result(question, tool_result)

        elapsed_qa = time.perf_counter() - started_qa

        st.session_state["geotiff_qa_result"] = {
            "question": question,
            "answer": answer,
            "tool_name": tool_name,
            "tool_result": tool_result,
            "raster_name": qa_name,
            "elapsed": elapsed_qa,
            "run_id": qa_run_id,
        }

        try:
            log_raster_nlq_run(
                run_id=qa_run_id,
                question=question,
                answer=answer,
                tool_name=tool_name,
                tool_result=tool_result,
                elapsed=elapsed_qa,
                raster_name=qa_name,
            )
        except Exception as log_exc:
            st.warning(
                f"The answer was produced, but Monitoring logging failed: {log_exc}"
            )

    except Exception as exc:
        st.error(f"GeoTIFF question failed: {exc}")

qa_result = st.session_state.get("geotiff_qa_result")
if qa_result:
    st.markdown("### 4. Grounded answer")
    st.info(qa_result["answer"])

    q1, q2, q3 = st.columns(3)
    q1.metric("Raster", qa_result["raster_name"])
    q2.metric("Tool", qa_result["tool_name"])
    q3.metric("Runtime", f"{qa_result['elapsed']:.2f}s")

    with st.expander("Show deterministic tool result"):
        st.json(qa_result["tool_result"])

    st.caption(
        "The numerical result comes from raster code. "
        "The LLM only converts that computed result into a concise explanation."
    )

with st.expander("Current NLQ scope and limitations", expanded=False):
    st.markdown(
        """
Supported now:

- band-1 statistics;
- raster metadata;
- threshold percentages;
- NDVI vegetation-signal classes;
- raster summaries.

Not yet supported:

- arbitrary zonal questions;
- directional questions such as *Where is vegetation weakest?*;
- comparisons between two dates unless a second-raster comparison tool is added.

Unsupported questions are not silently approximated.
"""
    )
