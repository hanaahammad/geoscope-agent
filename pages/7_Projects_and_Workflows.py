from __future__ import annotations

import io
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.plugins import Draw
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from shapely.geometry import shape
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geotiff_processing import (
    generate_product_geotiff,
    _read_clipped_band,
    _to_geotiff_bytes,
)
from src.llm_provider import generate_text, get_generation_model, get_provider
from src.stac_search import DATASET_CONFIG, search_dataset
from src.ui import apply_global_style
from src.workflow_store import (
    create_project,
    get_project,
    list_projects,
    project_metrics,
    save_snapshot,
    set_project_status,
)

st.set_page_config(
    page_title="GeoScope Project Assistant",
    page_icon="🧭",
    layout="wide",
)
apply_global_style()


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

TASKS: dict[str, dict[str, Any]] = {
    "Crop monitoring": {
        "icon": "🌱",
        "goal": "Monitor vegetation condition through the growing season.",
        "sources": ["Sentinel-2 Level-2A", "Landsat Collection 2 Level-2"],
        "defaults": ["Sentinel-2 Level-2A"],
        "indicator": "NDVI",
        "processing": "Calculate NDVI from Red and NIR bands.",
        "comparison": "Compare vegetation condition across distinct dates.",
        "interpretation": (
            "Higher NDVI generally indicates denser green vegetation. "
            "Low values can indicate sparse vegetation, bare soil, water, or stress."
        ),
        "supported_product": "NDVI",
    },
    "Urban heat island detection": {
        "icon": "🌡️",
        "goal": "Detect and compare surface heat patterns.",
        "sources": ["Landsat Collection 2 Level-2"],
        "defaults": ["Landsat Collection 2 Level-2"],
        "indicator": "Land Surface Temperature (LST)",
        "processing": "Calculate LST from thermal observations.",
        "comparison": "Compare heat intensity and identify hotspots.",
        "interpretation": (
            "Higher LST values indicate hotter surfaces. Interpretation must consider "
            "season, acquisition time, emissivity, and land-cover context."
        ),
        "supported_product": None,
    },
    "Flood mapping": {
        "icon": "🌊",
        "goal": "Detect flood extent using pre-event and post-event observations.",
        "sources": ["Sentinel-1 GRD", "Sentinel-2 Level-2A"],
        "defaults": ["Sentinel-1 GRD", "Sentinel-2 Level-2A"],
        "indicator": "Flood / water mask",
        "processing": "Create water or inundation masks.",
        "comparison": "Compare pre-event and post-event water extent.",
        "interpretation": (
            "Newly inundated area should be separated from permanent water. "
            "Sentinel-1 radar is valuable when clouds affect optical imagery."
        ),
        "supported_product": None,
    },
    "Drought monitoring": {
        "icon": "☀️",
        "goal": "Monitor persistent vegetation stress and drought-related anomalies.",
        "sources": ["Sentinel-2 Level-2A", "Landsat Collection 2 Level-2"],
        "defaults": ["Sentinel-2 Level-2A"],
        "indicator": "NDVI / vegetation anomaly",
        "processing": "Calculate vegetation indicators.",
        "comparison": "Build a multi-date vegetation trend and compare with baseline conditions.",
        "interpretation": (
            "Persistent low vegetation condition is more informative than one low value. "
            "Interpret with rainfall, soil moisture, seasonality, and crop stage."
        ),
        "supported_product": "NDVI",
    },
    "Water extent monitoring": {
        "icon": "💧",
        "goal": "Measure changes in lake, reservoir, or river water extent.",
        "sources": ["Sentinel-2 Level-2A", "Sentinel-1 GRD"],
        "defaults": ["Sentinel-2 Level-2A"],
        "indicator": "Water extent",
        "processing": "Create a water mask.",
        "comparison": "Compare water area across selected dates.",
        "interpretation": (
            "Water-area changes should be interpreted with seasonality, rainfall, "
            "reservoir operations, and sensor limitations."
        ),
        "supported_product": None,
    },
    "Urban growth monitoring": {
        "icon": "🏙️",
        "goal": "Measure built-up expansion between periods.",
        "sources": ["Sentinel-2 Level-2A", "Landsat Collection 2 Level-2"],
        "defaults": ["Sentinel-2 Level-2A", "Landsat Collection 2 Level-2"],
        "indicator": "Built-up change",
        "processing": "Map built-up or land-cover classes.",
        "comparison": "Compare classifications and calculate expansion.",
        "interpretation": (
            "Built-up change should use comparable dates and validated classification "
            "rather than visual interpretation alone."
        ),
        "supported_product": None,
    },
}


PROCESS_EDUCATION: dict[str, dict[str, Any]] = {
    "Crop monitoring": {
        "concept_title": "NDVI — Normalized Difference Vegetation Index",
        "concept": (
            "NDVI is a normalized indicator of green vegetation. Healthy leaves "
            "usually absorb visible red light and reflect strongly in near-infrared."
        ),
        "formula": "NDVI = (NIR - Red) / (NIR + Red)",
        "meaning": (
            "Values usually range from -1 to +1. Higher positive values generally "
            "mean a stronger vegetation signal, but NDVI is not a direct measurement "
            "of crop yield or plant health."
        ),
        "steps": [
            ("1", "Read the Red band", "Red reflectance is sensitive to chlorophyll absorption."),
            ("2", "Read the NIR band", "Healthy vegetation strongly reflects near-infrared energy."),
            ("3", "Clip both bands to the AOI", "Only pixels inside the study area should be analysed."),
            ("4", "Calculate NDVI", "The normalized ratio reduces the effect of overall brightness."),
            ("5", "Summarize the raster", "Mean, minimum, and maximum provide a quick project overview."),
            ("6", "Render and interpret", "Map spatial patterns and relate them to crop stage and field knowledge."),
        ],
        "other_indices": [
            ("EVI", "Vegetation index designed to be less sensitive to saturation and some atmospheric/background effects."),
            ("SAVI", "Vegetation index that reduces soil-background influence, useful when vegetation cover is sparse."),
            ("NDMI", "Uses NIR and SWIR to indicate vegetation/canopy moisture conditions."),
        ],
    },
    "Urban heat island detection": {
        "concept_title": "LST — Land Surface Temperature",
        "concept": (
            "LST is the temperature of the Earth's surface observed by the satellite. "
            "It is not the same as air temperature measured by a weather station."
        ),
        "formula": (
            "Landsat Level-2 Surface Temperature → scaled Kelvin → Celsius. "
            "GeoScope uses the ST asset metadata scale/offset when available."
        ),
        "meaning": (
            "Higher LST means a hotter observed surface. Built-up areas and bare soil "
            "often become hotter, while water, vegetation, irrigation, and shade can "
            "produce cooler surface patterns."
        ),
        "steps": [
            ("1", "Find the Landsat Surface Temperature asset", "Use the Level-2 thermal science product rather than Sentinel-2, which has no thermal band for LST."),
            ("2", "Read and clip it to the AOI", "Restrict the analysis to the selected city or district."),
            ("3", "Apply scale and offset", "The stored raster values must be converted to physical temperature."),
            ("4", "Convert Kelvin to Celsius", "Celsius is easier to interpret in the final dashboard."),
            ("5", "Calculate heat statistics", "Mean, minimum, maximum, and P90 summarize the heat distribution."),
            ("6", "Render the heat map", "Spatial intensity is more useful than one city-wide average."),
        ],
        "other_indices": [
            ("NDVI", "Useful beside LST because vegetation can explain cooler areas."),
            ("NDBI", "A built-up index that can help relate hot surfaces to urban fabric."),
            ("Urban heat intensity", "A comparison between hotter urban surfaces and a suitable cooler/reference area; methodology must be defined carefully."),
        ],
    },
    "Flood mapping": {
        "concept_title": "Flood / water extent",
        "concept": (
            "Flood mapping identifies water present after an event and separates "
            "new inundation from water that was already present before the event."
        ),
        "formula": "Post-event water mask - pre-event permanent water = candidate flood extent",
        "meaning": (
            "A water-looking pixel is not automatically a flood pixel. A valid flood "
            "workflow needs pre-event context, event timing, and preferably cloud-resistant radar."
        ),
        "steps": [
            ("1", "Select pre-event and post-event dates", "A flood is a change, so at least two distinct dates are required."),
            ("2", "Prefer Sentinel-1 under cloud", "Radar can observe the surface even when optical imagery is cloud-covered."),
            ("3", "Create water masks", "Detect likely water for both periods."),
            ("4", "Remove permanent water", "Avoid labeling rivers and lakes as newly flooded."),
            ("5", "Calculate inundated area", "Convert the flood mask into an interpretable area statistic."),
            ("6", "Render the flood extent", "Map where the new inundation occurred."),
        ],
        "other_indices": [
            ("NDWI", "Optical water index using Green and NIR; useful when clouds are limited."),
            ("MNDWI", "Modified water index using Green and SWIR; often useful in built-up areas."),
            ("SAR backscatter change", "Radar-based change signal used in Sentinel-1 flood workflows."),
        ],
    },
    "Drought monitoring": {
        "concept_title": "Vegetation anomaly / drought signal",
        "concept": (
            "Drought monitoring is temporal. A single low NDVI value does not prove "
            "drought; the signal becomes meaningful when vegetation is persistently "
            "below expected seasonal conditions."
        ),
        "formula": "Anomaly = current vegetation indicator - expected/baseline indicator",
        "meaning": (
            "Negative vegetation anomalies can indicate stress, but rainfall, soil "
            "moisture, irrigation, crop calendar, and land cover should also be considered."
        ),
        "steps": [
            ("1", "Calculate vegetation indicators", "NDVI is a practical first vegetation signal."),
            ("2", "Use several distinct dates", "Drought develops over time rather than in one acquisition."),
            ("3", "Build a seasonal trend", "Observe whether vegetation follows or departs from its expected trajectory."),
            ("4", "Create a baseline", "Comparison with normal conditions is stronger than raw values alone."),
            ("5", "Calculate anomalies", "Highlight persistent departures from the baseline."),
            ("6", "Interpret with climate context", "EO vegetation stress should not be interpreted in isolation."),
        ],
        "other_indices": [
            ("NDVI", "General vegetation greenness/vigor indicator."),
            ("NDMI", "Vegetation moisture-related signal using NIR and SWIR."),
            ("SAVI", "Vegetation index useful where exposed soil strongly affects the signal."),
        ],
    },
    "Water extent monitoring": {
        "concept_title": "Water extent",
        "concept": (
            "Water-extent monitoring converts satellite observations into a water mask "
            "and then measures how the mapped water area changes over time."
        ),
        "formula": "Water mask → pixel area × water pixels = mapped water surface area",
        "meaning": (
            "Changes may reflect rainfall, drought, reservoir operations, river dynamics, "
            "seasonality, or classification error."
        ),
        "steps": [
            ("1", "Choose suitable water-sensitive data", "Sentinel-2 is useful when cloud-free; Sentinel-1 is useful under cloud."),
            ("2", "Calculate a water signal", "Use an index or radar-based detection method."),
            ("3", "Create a water mask", "Turn continuous values into mapped water/non-water classes."),
            ("4", "Calculate area", "Translate raster pixels into an interpretable surface-area measure."),
            ("5", "Repeat for other dates", "Water monitoring is fundamentally temporal."),
            ("6", "Render change", "Show where water expanded, contracted, or remained stable."),
        ],
        "other_indices": [
            ("NDWI", "Green/NIR index commonly used to enhance open water."),
            ("MNDWI", "Green/SWIR variant often useful for water in urban environments."),
            ("SAR water mask", "Radar-based water detection, useful during cloudy periods."),
        ],
    },
    "Urban growth monitoring": {
        "concept_title": "Built-up / urban change",
        "concept": (
            "Urban-growth monitoring maps built-up land at two or more comparable "
            "periods and measures where the built footprint expanded."
        ),
        "formula": "Built-up map at T2 - built-up map at T1 = candidate urban expansion",
        "meaning": (
            "A built-up spectral index is an indicator, not a final land-use label. "
            "Classification and validation are still required."
        ),
        "steps": [
            ("1", "Choose comparable dates", "Season and image quality should be similar when possible."),
            ("2", "Calculate built-up indicators", "Indices can highlight candidate urban surfaces."),
            ("3", "Classify built-up / non-built-up", "Translate spectral information into mapped classes."),
            ("4", "Validate the classification", "Avoid confusing bare soil, rock, and bright surfaces with buildings."),
            ("5", "Calculate change area", "Measure where the built footprint increased."),
            ("6", "Render expansion", "Show the spatial pattern and growth direction."),
        ],
        "other_indices": [
            ("NDBI", "Uses SWIR and NIR to highlight many built-up surfaces."),
            ("NDVI", "Helps distinguish vegetation from candidate built-up areas."),
            ("MNDWI", "Can help distinguish water from other urban surfaces."),
        ],
    },
}



ASSISTANT_RULES = """
You are the GeoScope Earth Observation Project Assistant.

Help the user accomplish the current EO project step by step.
Use the task, AOI, date range, selected datasets, actual STAC results,
selected scene(s), and generated results supplied in the prompt.

Be concise and practical:
1. What has been completed?
2. What is the next action?
3. Why is it needed?
4. What result should be produced?
5. Can GeoScope execute it now?

Never claim unsupported processing was executed.
GeoScope currently supports real Sentinel-2 Red/NIR/NDVI GeoTIFF processing.
It does not currently implement production LST, flood-mask generation,
downscaling, classification, mosaicking, or aligned multi-date raster cubes.
Scene items are not the same as distinct acquisition dates.
""".strip()


# =============================================================================
# HELPERS
# =============================================================================

def distinct_dates(scenes: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(scene.get("date"))[:10]
            for scene in scenes or []
            if scene.get("date")
        }
    )


def scene_label(scene: dict[str, Any]) -> str:
    cloud = scene.get("cloud_cover")
    cloud_text = "n/a" if cloud is None else f"{float(cloud):.2f}%"
    return (
        f"{scene.get('date')} · {scene.get('item_id')} "
        f"· cloud {cloud_text}"
    )


def safe_preview_url(url: str | None) -> str | None:
    """
    Convert public S3 thumbnail URIs to HTTPS for Streamlit.
    Streamlit cannot display s3:// paths directly.
    """
    if not url:
        return None

    url = str(url).strip()

    if url.startswith("https://") or url.startswith("http://"):
        return url

    if url.startswith("s3://"):
        remainder = url[5:]
        if "/" not in remainder:
            return None
        bucket, key = remainder.split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"

    return None


def flatten_scenes(
    scenes_by_dataset: dict[str, list[dict[str, Any]]]
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for dataset, scenes in scenes_by_dataset.items():
        for scene in scenes:
            result.append((dataset, scene))
    return result


def best_scene_per_date(
    scenes: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        d = str(scene.get("date") or "")[:10]
        if d:
            by_date.setdefault(d, []).append(scene)

    chosen: dict[str, dict[str, Any]] = {}
    for d, items in by_date.items():
        chosen[d] = min(
            items,
            key=lambda s: (
                s.get("cloud_cover")
                if s.get("cloud_cover") is not None
                else 999
            ),
        )
    return chosen


def snapshot_state() -> dict[str, Any]:
    keys = [
        "aoi_geojson",
        "aoi_summary",
        "project_selected_datasets",
        "project_scenes_by_dataset",
        "start_date",
        "end_date",
        "max_cloud_cover",
        "project_primary_scene",
        "project_primary_dataset",
        "project_comparison_dates",
        "generated_geotiff_name",
        "generated_geotiff_summary",
        "project_result_history",
        "last_sources",
        "last_answer",
    ]
    return {
        key: st.session_state[key]
        for key in keys
        if key in st.session_state
    }


def restore_snapshot(snapshot: dict[str, Any]) -> None:
    for key, value in (snapshot or {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_project_state(
    project_id: str,
    snapshot: dict[str, Any],
    task_name: str,
    goal: str,
    chat: list[dict[str, str]],
) -> None:
    updated = dict(snapshot or {})
    updated.update(snapshot_state())
    updated["assistant_task_name"] = task_name
    updated["assistant_goal"] = goal
    updated["assistant_chat"] = chat
    save_snapshot(project_id, updated)


def build_aoi_summary(aoi_geojson: dict[str, Any]) -> str:
    geometry = shape(aoi_geojson)
    centroid = geometry.centroid
    minx, miny, maxx, maxy = geometry.bounds
    return (
        f"{aoi_geojson.get('type', 'Geometry')} AOI; "
        f"centroid {centroid.y:.4f}, {centroid.x:.4f}; "
        f"bbox [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]"
    )


def render_aoi_map(aoi: dict[str, Any] | None, key: str) -> None:
    if not aoi:
        st.info("No AOI defined.")
        return
    geometry = shape(aoi)
    centroid = geometry.centroid
    fmap = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=9,
        control_scale=True,
    )
    folium.GeoJson(aoi, name="AOI").add_to(fmap)
    st_folium(
        fmap,
        height=340,
        width=None,
        key=key,
    )


def render_geotiff_preview(
    geotiff_bytes: bytes,
    product: str,
) -> None:
    """
    Render a lightweight raster preview without matplotlib.

    NDVI uses a simple brown/yellow/green RGB ramp:
    - low values -> brown/yellow
    - medium values -> light green
    - high values -> dark green

    Other products are shown as grayscale.
    """
    with MemoryFile(geotiff_bytes) as mem:
        with mem.open() as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

    valid = np.isfinite(arr)

    if not np.any(valid):
        st.warning("The generated raster has no valid pixels to preview.")
        return

    product_upper = product.upper()

    if product_upper in {"LST_C", "LST", "LAND SURFACE TEMPERATURE"}:
        finite = arr[valid]
        p5 = float(np.nanpercentile(finite, 5))
        p95 = float(np.nanpercentile(finite, 95))
        if p95 <= p5:
            p5 = float(np.nanmin(finite))
            p95 = float(np.nanmax(finite))

        norm = (
            np.clip((arr - p5) / (p95 - p5), 0.0, 1.0)
            if p95 > p5
            else np.zeros(arr.shape, dtype="float32")
        )

        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)

        cool = norm < 0.5
        hot = ~cool

        t_cool = np.clip(norm / 0.5, 0.0, 1.0)
        rgb[..., 0][cool] = (35 + 220 * t_cool[cool]).astype(np.uint8)
        rgb[..., 1][cool] = (110 + 120 * t_cool[cool]).astype(np.uint8)
        rgb[..., 2][cool] = (220 - 120 * t_cool[cool]).astype(np.uint8)

        t_hot = np.clip((norm - 0.5) / 0.5, 0.0, 1.0)
        rgb[..., 0][hot] = 255
        rgb[..., 1][hot] = (230 - 190 * t_hot[hot]).astype(np.uint8)
        rgb[..., 2][hot] = (100 - 80 * t_hot[hot]).astype(np.uint8)

        rgb[~valid] = 255

        st.image(
            rgb,
            caption=(
                f"LST preview — cooler surfaces → warmer surfaces "
                f"(display stretch {p5:.1f} to {p95:.1f} °C)"
            ),
            use_container_width=True,
        )
        st.caption(
            "LST is surface temperature, not air temperature. "
            "Interpret with acquisition date/time, land cover, vegetation, "
            "water, emissivity, and atmospheric conditions."
        )

    elif product_upper == "NDVI":
        # Normalize the physically meaningful NDVI range [-1, 1] to [0, 1].
        norm = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)

        # Piecewise brown/yellow -> green ramp.
        low = norm < 0.5
        high = ~low

        # Low NDVI: brown to yellow.
        t_low = np.clip(norm / 0.5, 0.0, 1.0)
        rgb[..., 0][low] = (150 + 105 * t_low[low]).astype(np.uint8)
        rgb[..., 1][low] = (80 + 140 * t_low[low]).astype(np.uint8)
        rgb[..., 2][low] = (40 + 40 * t_low[low]).astype(np.uint8)

        # High NDVI: yellow-green to dark green.
        t_high = np.clip((norm - 0.5) / 0.5, 0.0, 1.0)
        rgb[..., 0][high] = (220 * (1.0 - t_high[high])).astype(np.uint8)
        rgb[..., 1][high] = (220 - 90 * t_high[high]).astype(np.uint8)
        rgb[..., 2][high] = (80 - 40 * t_high[high]).astype(np.uint8)

        rgb[~valid] = 255

        st.image(
            rgb,
            caption=(
                "NDVI preview — brown/yellow = lower vegetation signal, "
                "green = higher vegetation signal"
            ),
            use_container_width=True,
        )

        st.caption(
            "NDVI scale: -1 to +1. Interpretation depends on land cover, "
            "season, crop stage, clouds, and data quality."
        )

    else:
        finite = arr[valid]
        p2 = float(np.nanpercentile(finite, 2))
        p98 = float(np.nanpercentile(finite, 98))

        if p98 <= p2:
            p2 = float(np.nanmin(finite))
            p98 = float(np.nanmax(finite))

        if p98 <= p2:
            gray = np.zeros(arr.shape, dtype=np.uint8)
        else:
            scaled = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
            gray = (scaled * 255).astype(np.uint8)

        gray[~valid] = 255

        st.image(
            gray,
            caption=f"{product_upper} preview",
            use_container_width=True,
            clamp=True,
        )



def geotiff_overlay(
    geotiff_bytes: bytes,
    product: str,
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Convert a generated GeoTIFF into an RGBA image plus WGS84 bounds
    suitable for folium.ImageOverlay.
    """
    with MemoryFile(geotiff_bytes) as mem:
        with mem.open() as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            crs = src.crs
            src_bounds = src.bounds

            if nodata is not None:
                arr[arr == nodata] = np.nan

            if crs is None:
                raise ValueError("Generated raster has no CRS.")

            west, south, east, north = transform_bounds(
                crs,
                "EPSG:4326",
                src_bounds.left,
                src_bounds.bottom,
                src_bounds.right,
                src_bounds.top,
                densify_pts=21,
            )

    valid = np.isfinite(arr)
    if not np.any(valid):
        raise ValueError("Generated raster has no valid pixels.")

    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)

    product_upper = product.upper()

    if product_upper in {"LST_C", "LST", "LAND SURFACE TEMPERATURE"}:
        finite = arr[valid]
        p5 = float(np.nanpercentile(finite, 5))
        p95 = float(np.nanpercentile(finite, 95))
        if p95 <= p5:
            p5 = float(np.nanmin(finite))
            p95 = float(np.nanmax(finite))

        norm = (
            np.clip((arr - p5) / (p95 - p5), 0.0, 1.0)
            if p95 > p5
            else np.zeros(arr.shape, dtype="float32")
        )

        cool = norm < 0.5
        hot = ~cool
        t_cool = np.clip(norm / 0.5, 0.0, 1.0)
        rgba[..., 0][cool] = (35 + 220 * t_cool[cool]).astype(np.uint8)
        rgba[..., 1][cool] = (110 + 120 * t_cool[cool]).astype(np.uint8)
        rgba[..., 2][cool] = (220 - 120 * t_cool[cool]).astype(np.uint8)

        t_hot = np.clip((norm - 0.5) / 0.5, 0.0, 1.0)
        rgba[..., 0][hot] = 255
        rgba[..., 1][hot] = (230 - 190 * t_hot[hot]).astype(np.uint8)
        rgba[..., 2][hot] = (100 - 80 * t_hot[hot]).astype(np.uint8)

    elif product_upper == "NDVI":
        norm = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

        low = norm < 0.5
        high = ~low

        t_low = np.clip(norm / 0.5, 0.0, 1.0)
        rgba[..., 0][low] = (150 + 105 * t_low[low]).astype(np.uint8)
        rgba[..., 1][low] = (80 + 140 * t_low[low]).astype(np.uint8)
        rgba[..., 2][low] = (40 + 40 * t_low[low]).astype(np.uint8)

        t_high = np.clip((norm - 0.5) / 0.5, 0.0, 1.0)
        rgba[..., 0][high] = (220 * (1.0 - t_high[high])).astype(np.uint8)
        rgba[..., 1][high] = (220 - 90 * t_high[high]).astype(np.uint8)
        rgba[..., 2][high] = (80 - 40 * t_high[high]).astype(np.uint8)

    else:
        finite = arr[valid]
        p2 = float(np.nanpercentile(finite, 2))
        p98 = float(np.nanpercentile(finite, 98))

        if p98 <= p2:
            p2 = float(np.nanmin(finite))
            p98 = float(np.nanmax(finite))

        if p98 <= p2:
            gray = np.zeros(arr.shape, dtype=np.uint8)
        else:
            scaled = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
            gray = (scaled * 255).astype(np.uint8)

        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray

    rgba[..., 3][valid] = 190
    rgba[..., 3][~valid] = 0

    bounds = [
        [south, west],
        [north, east],
    ]

    return rgba, bounds


def render_result_map(
    aoi: dict[str, Any] | None,
    geotiff_bytes: bytes | None,
    product: str | None,
) -> None:
    """
    Render the latest derived raster on top of the project AOI.

    Because the generated GeoTIFF is already clipped to the AOI, GeoScope
    uses the AOI bounding box for the Leaflet image overlay. This avoids
    browser/map-display problems caused by projected raster bounds while
    keeping the result correctly positioned for this clipped-output MVP.
    """
    if not aoi:
        st.info("No AOI is available for the result map.")
        return

    geometry = shape(aoi)
    minx, miny, maxx, maxy = geometry.bounds
    centroid = geometry.centroid

    fmap = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=10,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    folium.GeoJson(
        aoi,
        name="AOI",
        style_function=lambda _: {
            "color": "#2474A6",
            "weight": 3,
            "fillOpacity": 0.02,
        },
    ).add_to(fmap)

    if geotiff_bytes and product:
        rgba, _raster_bounds = geotiff_overlay(
            geotiff_bytes,
            product,
        )

        # The processed raster is clipped to the AOI.  Use the AOI bbox for
        # display instead of the source projected-raster bounds; this keeps
        # the Leaflet overlay spatially aligned and prevents over-zooming.
        overlay_bounds = [
            [miny, minx],
            [maxy, maxx],
        ]

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=overlay_bounds,
            opacity=0.78,
            name=f"{product.upper()} result",
            interactive=True,
            cross_origin=False,
            zindex=5,
        ).add_to(fmap)

    # Fit tightly to the AOI when the map first opens.
    fmap.fit_bounds(
        [
            [miny, minx],
            [maxy, maxx],
        ],
        padding=(18, 18),
        max_zoom=13,
    )

    # Add a visible scientific color scale ("thermometer") for the result.
    if geotiff_bytes and product:
        product_upper = str(product).upper()

        if product_upper == "NDVI":
            legend_html = """
            <div style="
                position: fixed;
                bottom: 28px;
                right: 24px;
                z-index: 9999;
                background: rgba(255,255,255,0.95);
                border: 1px solid #999;
                border-radius: 10px;
                padding: 10px 12px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.25);
                font-size: 12px;
                min-width: 220px;">
              <div style="font-weight:700;font-size:13px;margin-bottom:6px;">
                NDVI vegetation intensity
              </div>

              <div style="
                  height:16px;
                  border-radius:6px;
                  background: linear-gradient(
                      to right,
                      #8c4b2f 0%,
                      #d9a441 25%,
                      #e6dc71 45%,
                      #9ccc65 65%,
                      #2e7d32 82%,
                      #0b5d1e 100%
                  );">
              </div>

              <div style="
                  display:flex;
                  justify-content:space-between;
                  margin-top:4px;">
                <span>-1</span>
                <span>0</span>
                <span>0.5</span>
                <span>1</span>
              </div>

              <div style="
                  display:flex;
                  justify-content:space-between;
                  margin-top:4px;
                  color:#555;">
                <span>low signal</span>
                <span>high vegetation</span>
              </div>
            </div>
            """
            fmap.get_root().html.add_child(
                folium.Element(legend_html)
            )

        elif product_upper in {
            "LST_C",
            "LST",
            "LAND SURFACE TEMPERATURE",
        }:
            legend_html = """
            <div style="
                position: fixed;
                bottom: 28px;
                right: 24px;
                z-index: 9999;
                background: rgba(255,255,255,0.95);
                border: 1px solid #999;
                border-radius: 10px;
                padding: 10px 12px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.25);
                font-size: 12px;
                min-width: 220px;">
              <div style="font-weight:700;font-size:13px;margin-bottom:6px;">
                Surface temperature intensity
              </div>

              <div style="
                  height:16px;
                  border-radius:6px;
                  background: linear-gradient(
                      to right,
                      #2376d8 0%,
                      #65b9e8 25%,
                      #f1dc73 55%,
                      #ef913c 75%,
                      #c62828 100%
                  );">
              </div>

              <div style="
                  display:flex;
                  justify-content:space-between;
                  margin-top:4px;color:#555;">
                <span>cooler</span>
                <span>hotter</span>
              </div>
            </div>
            """
            fmap.get_root().html.add_child(
                folium.Element(legend_html)
            )

    folium.LayerControl(collapsed=False).add_to(fmap)

    st_folium(
        fmap,
        width=None,
        height=560,
        key=f"page7_result_overlay_map_{product or 'aoi'}",
    )



def render_processing_education(task_name: str) -> None:
    guide = PROCESS_EDUCATION[task_name]

    st.markdown(
        f"""
        <div style="
            padding:18px 20px;
            border-radius:16px;
            border:1px solid rgba(49,130,99,0.25);
            background:rgba(49,130,99,0.06);
            margin-bottom:14px;">
            <div style="font-size:0.82rem;font-weight:700;letter-spacing:.08em;
                        text-transform:uppercase;opacity:.72;">Understand the indicator</div>
            <div style="font-size:1.35rem;font-weight:750;margin-top:4px;">
                {guide['concept_title']}
            </div>
            <div style="margin-top:8px;line-height:1.55;">{guide['concept']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🧮 Method / formula")
        st.info(guide["formula"])
    with c2:
        st.markdown("#### 👁️ What does it mean?")
        st.success(guide["meaning"])

    st.markdown("### Processing journey")
    st.caption(
        "GeoScope explains each operation before the result is produced, "
        "so the workflow remains transparent rather than becoming a black box."
    )

    for number, title, why in guide["steps"]:
        st.markdown(
            f"""
            <div style="
                display:flex;
                gap:14px;
                align-items:flex-start;
                padding:13px 15px;
                margin:8px 0;
                border-radius:14px;
                background:rgba(255,255,255,0.42);
                border:1px solid rgba(120,120,120,0.17);">
                <div style="
                    min-width:34px;height:34px;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    background:rgba(49,130,99,.14);
                    font-weight:800;">{number}</div>
                <div>
                    <div style="font-weight:750;font-size:1.02rem;">{title}</div>
                    <div style="opacity:.78;margin-top:3px;line-height:1.45;">{why}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📚 Related indicators — what else could be used?"):
        for name, explanation in guide["other_indices"]:
            st.markdown(f"**{name}** — {explanation}")


def find_landsat_surface_temperature_asset(
    scene: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Find the Landsat Collection 2 Level-2 surface-temperature asset.
    Works with common Earth Search naming patterns and STAC band metadata.
    """
    assets = scene.get("assets") or {}

    preferred = [
        "lwir11",
        "thermal",
        "st_b10",
        "ST_B10",
        "surface_temperature",
    ]

    for key in preferred:
        if key in assets:
            return key, assets[key]

    for key, metadata in assets.items():
        key_lower = str(key).lower()
        title_lower = str(metadata.get("title", "")).lower()

        if (
            "st_b10" in key_lower
            or "surface temperature" in title_lower
            or "lwir11" in key_lower
        ):
            return key, metadata

        for band in metadata.get("eo_bands", []) or []:
            common_name = str(band.get("common_name", "")).lower()
            name = str(band.get("name", "")).lower()
            if common_name in {"lwir11", "thermal"} or "st_b10" in name:
                return key, metadata

    available = ", ".join(scene.get("available_assets") or assets.keys())
    raise KeyError(
        "Could not find a Landsat Surface Temperature asset in this scene. "
        f"Available assets: {available}"
    )


def generate_landsat_lst_geotiff(
    scene: dict[str, Any],
    aoi_geometry: dict[str, Any],
) -> tuple[bytes, str, dict[str, Any]]:
    """
    Prepare the Landsat Collection 2 Level-2 surface-temperature product.

    The STAC asset metadata scale/offset are used when available by the
    existing GeoScope raster reader. If scale/offset metadata are absent,
    the Landsat Collection 2 surface-temperature fallback coefficients are
    applied: Kelvin = DN * 0.00341802 + 149.0.
    """
    asset_key, asset = find_landsat_surface_temperature_asset(scene)

    data, profile, invalid = _read_clipped_band(
        asset,
        aoi_geometry,
    )

    raster_bands = asset.get("raster_bands") or []
    metadata = raster_bands[0] if raster_bands else {}

    has_scale_metadata = (
        isinstance(metadata, dict)
        and (
            metadata.get("scale") is not None
            or metadata.get("offset") is not None
        )
    )

    if not has_scale_metadata:
        data = data * 0.00341802 + 149.0

    # Surface Temperature is now expected in Kelvin.
    celsius = data - 273.15

    # Exclude implausible values / fill artifacts from summary and display.
    invalid = (
        invalid
        | ~np.isfinite(celsius)
        | (celsius < -80.0)
        | (celsius > 80.0)
    )

    output_bytes = _to_geotiff_bytes(
        celsius,
        profile,
        invalid,
        dtype="float32",
        nodata=-9999.0,
        band_description="Landsat Level-2 Land Surface Temperature (Celsius)",
    )

    valid = celsius[(~invalid) & np.isfinite(celsius)]

    summary = {
        "product": "LST_C",
        "display_name": "Land Surface Temperature",
        "unit": "°C",
        "scene": scene.get("item_id"),
        "date": scene.get("date"),
        "asset": asset_key,
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "crs": str(profile["crs"]),
        "minimum": float(valid.min()) if valid.size else None,
        "maximum": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean()) if valid.size else None,
        "p90": float(np.percentile(valid, 90)) if valid.size else None,
    }

    filename = (
        f"{scene.get('item_id', 'landsat')}_"
        f"{scene.get('date', 'unknown-date')}_lst_celsius_clip.tif"
    )

    return output_bytes, filename, summary


def project_context(
    task_name: str,
    goal: str,
    snapshot: dict[str, Any],
) -> str:
    scenes_by_dataset = (
        st.session_state.get("project_scenes_by_dataset")
        or snapshot.get("project_scenes_by_dataset")
        or {}
    )
    selected = (
        st.session_state.get("project_selected_datasets")
        or snapshot.get("project_selected_datasets")
        or []
    )
    primary_scene = (
        st.session_state.get("project_primary_scene")
        or snapshot.get("project_primary_scene")
    )
    result_summary = (
        st.session_state.get("generated_geotiff_summary")
        or snapshot.get("generated_geotiff_summary")
    )

    availability = []
    for dataset, scenes in scenes_by_dataset.items():
        availability.append(
            f"{dataset}: {len(scenes)} scene items / "
            f"{len(distinct_dates(scenes))} distinct dates"
        )

    return "\n".join(
        [
            f"Task: {task_name}",
            f"Goal: {goal}",
            f"AOI: {st.session_state.get('aoi_summary') or snapshot.get('aoi_summary') or 'not defined'}",
            f"Dates: {st.session_state.get('start_date') or snapshot.get('start_date') or 'not set'} "
            f"to {st.session_state.get('end_date') or snapshot.get('end_date') or 'not set'}",
            f"Selected datasets: {', '.join(selected) if selected else 'none'}",
            "STAC availability:",
            *(availability or ["no STAC search yet"]),
            f"Primary scene: {primary_scene.get('item_id') if primary_scene else 'not selected'}",
            f"Generated result summary: {result_summary or 'none'}",
        ]
    )


# =============================================================================
# HEADER / PROJECT MANAGER
# =============================================================================

st.title("🧭 GeoScope Assisted EO Project")

st.markdown(
    """
### From research goal to an adaptive Earth Observation workflow

Earth Observation analysis is rarely a single fixed pipeline. The correct
workflow depends on the **task**, **sensor**, **AOI**, **date range**,
**available acquisitions**, **cloud conditions**, **processing level**,
**available bands/assets**, and sometimes even the **remote data-access
method**.

A crop-monitoring workflow based on Sentinel-2 NDVI is therefore very
different from an urban-heat workflow based on Landsat Surface Temperature,
or a flood workflow that may need Sentinel-1 radar.

The challenge is not only choosing an algorithm. It is keeping the workflow
correct while the **data context changes**.

GeoScope treats the analysis as an **AI-assisted project workflow**, not as
a rigid recipe.
"""
)

with st.expander(
    "💡 Why can Earth Observation workflows become complicated?",
    expanded=True,
):
    st.markdown(
        """
During a real EO project, the expected workflow may need to adapt because:

- the recommended dataset may change after the AOI and dates are known;
- a selected scene may not contain the required asset or band;
- several STAC items may represent the **same acquisition date**;
- cloud, quality, spatial coverage, or acquisition timing may make a scene unusable;
- different sensors require different bands, formulas, scaling rules, and preprocessing;
- some remote assets require a different access method, authentication, or provider;
- an apparently correct method may fail because the **actual data context**
  is different from the assumed context.

A simple example is urban heat:

```text
Expected workflow
Landsat
   ↓
Surface Temperature asset
   ↓
Scale / convert to Celsius
   ↓
Heat indicators
   ↓
Heat map

Actual project context
Landsat scene found
   ↓
Surface Temperature asset found
   ↓
Remote asset access method causes a failure
   ↓
Workflow must adapt
```

This is why GeoScope exposes the processing steps and explains what is
happening instead of hiding everything behind one button.
"""
    )

with st.expander(
    "🧠 What is the role of the LLM and context engineering?",
    expanded=False,
):
    st.markdown(
        """
The LLM is not used only to answer questions.

It helps the user:

- understand what each EO processing step means;
- explain **why** a step is required;
- adapt the recommended workflow to the current project state;
- identify when the available data is insufficient;
- suggest the next action;
- interpret the generated indicators;
- distinguish between **implemented**, **partially implemented**, and
  **guided/future** processing.

The quality of this assistance depends on **context engineering**. The model
may need more than the user's question:

```text
Research goal
    +
Task
    +
AOI
    +
Dates
    +
Selected datasets
    +
STAC search results
    +
Available bands/assets
    +
Previous processing results
    +
Project state
    ↓
Context supplied to the LLM
    ↓
Explanation / next action / interpretation
```

The workflow can change at any point as this context changes.
"""
    )

st.info(
    "**Implementation status:** Sentinel-2 Red/NIR/NDVI processing is "
    "executable in GeoScope. Landsat Surface Temperature is partially "
    "implemented but remote raster access may depend on the selected data "
    "provider/network. Other EO workflows are explained and tracked without "
    "claiming unsupported automation."
)

st.markdown(
    """
### One workspace: define → search → select → understand → process → compare → review → save

All assisted-project actions are performed **here on Page 7**.
Page 2 remains an independent AOI/STAC exploration page and is not required
for this assisted project workflow.
"""
)

metrics = project_metrics()
m1, m2, m3 = st.columns(3)
m1.metric("Active projects", metrics.get("ACTIVE", 0))
m2.metric("Completed", metrics.get("COMPLETED", 0))
m3.metric("Archived", metrics.get("ARCHIVED", 0))

new_tab, open_tab = st.tabs(["✨ New project", "📁 Open project"])

with new_tab:
    task_name_new = st.selectbox(
        "EO task",
        list(TASKS.keys()),
        key="new_task",
    )
    task_new = TASKS[task_name_new]
    c1, c2 = st.columns([1.3, 1])
    with c1:
        goal_new = st.text_area(
            "Research objective",
            value=task_new["goal"],
            height=90,
        )
        project_name_new = st.text_input(
            "Project name",
            value=f"{task_name_new} project",
        )
    with c2:
        st.write(f"**Recommended source(s):** {', '.join(task_new['defaults'])}")
        st.write(f"**Target indicator:** {task_new['indicator']}")
        st.write(f"**Processing:** {task_new['processing']}")

    if st.button(
        "Create project",
        type="primary",
        use_container_width=True,
    ):
        pid = create_project(
            project_name_new,
            goal_new.strip() or task_new["goal"],
        )
        save_snapshot(
            pid,
            {
                "assistant_task_name": task_name_new,
                "assistant_goal": goal_new.strip() or task_new["goal"],
                "assistant_chat": [],
            },
        )
        st.session_state["active_project_id"] = pid
        st.rerun()

with open_tab:
    projects = list_projects(include_archived=True)
    if not projects:
        st.info("No saved projects.")
    else:
        selected_pid = st.selectbox(
            "Project",
            [p["project_id"] for p in projects],
            format_func=lambda pid: next(
                (
                    f"{p['project_name']} · {p['status']}"
                    for p in projects
                    if p["project_id"] == pid
                ),
                pid,
            ),
        )
        if st.button("Open selected project", use_container_width=True):
            st.session_state["active_project_id"] = selected_pid
            st.rerun()


# =============================================================================
# ACTIVE PROJECT
# =============================================================================

project_id = st.session_state.get("active_project_id")

if project_id:
    project = get_project(project_id)
    snapshot = project.get("snapshot") or {}
    restore_snapshot(snapshot)

    task_name = snapshot.get("assistant_task_name")
    if task_name not in TASKS:
        st.warning(
            "This project uses an older Page 7 format. Create a new assisted "
            "project to use the all-in-one workflow."
        )
        st.stop()

    task = TASKS[task_name]
    goal = snapshot.get("assistant_goal") or project.get("description") or task["goal"]
    chat = snapshot.get("assistant_chat") or []

    st.divider()
    st.header(f"{task['icon']} {project['project_name']}")
    st.write(f"**Objective:** {goal}")

    step1, step2, step3, step4, step5, step6 = st.tabs(
        [
            "1️⃣ AOI + data",
            "2️⃣ Scene search",
            "3️⃣ Process",
            "4️⃣ Compare",
            "5️⃣ Results + save",
            "🗺️ Result map",
        ]
    )

    # =========================================================================
    # STEP 1 — AOI + DATA
    # =========================================================================

    with step1:
        st.subheader("Step 1 — Define AOI, dates, and source(s)")

        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown("#### Draw / update AOI")
            existing_aoi = st.session_state.get("aoi_geojson")
            if existing_aoi:
                geometry = shape(existing_aoi)
                centroid = geometry.centroid
                center = [centroid.y, centroid.x]
                zoom = 9
            else:
                center = [30.5, 30.8]
                zoom = 7

            fmap = folium.Map(
                location=center,
                zoom_start=zoom,
                control_scale=True,
            )

            if existing_aoi:
                folium.GeoJson(existing_aoi, name="Current AOI").add_to(fmap)

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

            output = st_folium(
                fmap,
                width=None,
                height=470,
                returned_objects=["last_active_drawing"],
                key="page7_aoi_draw",
            )

            drawing = output.get("last_active_drawing") if output else None
            if drawing and drawing.get("geometry"):
                aoi = drawing["geometry"]
                st.session_state["aoi_geojson"] = aoi
                st.session_state["aoi_summary"] = build_aoi_summary(aoi)

            if st.session_state.get("aoi_geojson"):
                st.success("AOI ready.")
                st.caption(st.session_state.get("aoi_summary"))

        with right:
            st.markdown("#### Analysis period")
            saved_start = st.session_state.get("start_date")
            saved_end = st.session_state.get("end_date")
            default_end = date.today()
            default_start = default_end - timedelta(days=60)

            try:
                start_value = date.fromisoformat(saved_start) if saved_start else default_start
            except Exception:
                start_value = default_start
            try:
                end_value = date.fromisoformat(saved_end) if saved_end else default_end
            except Exception:
                end_value = default_end

            start_date = st.date_input("Start date", value=start_value)
            end_date = st.date_input("End date", value=end_value)
            st.session_state["start_date"] = start_date.isoformat()
            st.session_state["end_date"] = end_date.isoformat()

            st.markdown("#### Data source(s)")
            available = [
                dataset
                for dataset in task["sources"]
                if dataset in DATASET_CONFIG
            ]
            defaults = [
                d for d in (
                    st.session_state.get("project_selected_datasets")
                    or task["defaults"]
                )
                if d in available
            ]
            selected_datasets = st.multiselect(
                "Dataset(s)",
                available,
                default=defaults,
            )
            st.session_state["project_selected_datasets"] = selected_datasets

            for dataset in selected_datasets:
                st.caption(
                    f"**{dataset}:** {DATASET_CONFIG[dataset]['description']}"
                )

            has_optical = any(
                DATASET_CONFIG[d]["cloud_filter"]
                for d in selected_datasets
            )
            if has_optical:
                max_cloud = st.slider(
                    "Maximum cloud cover (%)",
                    0,
                    100,
                    int(st.session_state.get("max_cloud_cover", 20)),
                )
            else:
                max_cloud = 100
                st.caption("Cloud filter not used for radar.")
            st.session_state["max_cloud_cover"] = max_cloud

        if st.button(
            "✅ Save Step 1 and continue",
            type="primary",
            use_container_width=True,
        ):
            if not st.session_state.get("aoi_geojson"):
                st.error("Define an AOI first.")
            elif not selected_datasets:
                st.error("Select at least one dataset.")
            else:
                save_project_state(
                    project_id, snapshot, task_name, goal, chat
                )
                st.success("Context saved. Open Step 2 — Scene search.")

    # =========================================================================
    # STEP 2 — SEARCH + SELECT
    # =========================================================================

    with step2:
        st.subheader("Step 2 — Search real satellite observations")

        aoi = st.session_state.get("aoi_geojson")
        selected_datasets = st.session_state.get("project_selected_datasets") or []

        context_cols = st.columns(4)
        context_cols[0].metric("AOI", "Ready" if aoi else "Missing")
        context_cols[1].metric("Sources", len(selected_datasets))
        context_cols[2].metric("Start", st.session_state.get("start_date", "—"))
        context_cols[3].metric("End", st.session_state.get("end_date", "—"))

        scene_limit = st.slider(
            "Maximum scene items per dataset",
            1,
            20,
            10,
            key="page7_scene_limit",
        )

        if st.button(
            "🔎 Search selected dataset(s)",
            type="primary",
            use_container_width=True,
        ):
            if not aoi:
                st.error("Complete Step 1 first: AOI is missing.")
            elif not selected_datasets:
                st.error("Complete Step 1 first: no dataset selected.")
            else:
                scenes_by_dataset: dict[str, list[dict[str, Any]]] = {}
                for dataset in selected_datasets:
                    try:
                        with st.spinner(f"Searching {dataset}..."):
                            scenes_by_dataset[dataset] = search_dataset(
                                dataset_name=dataset,
                                aoi_geometry=aoi,
                                start_date=st.session_state["start_date"],
                                end_date=st.session_state["end_date"],
                                max_cloud_cover=st.session_state.get(
                                    "max_cloud_cover", 20
                                ),
                                limit=scene_limit,
                            )
                    except Exception as exc:
                        st.warning(f"{dataset}: {exc}")
                        scenes_by_dataset[dataset] = []

                st.session_state["project_scenes_by_dataset"] = scenes_by_dataset

                if "Sentinel-2 Level-2A" in scenes_by_dataset:
                    st.session_state["stac_scenes"] = scenes_by_dataset[
                        "Sentinel-2 Level-2A"
                    ]

                save_project_state(
                    project_id, snapshot, task_name, goal, chat
                )
                st.rerun()

        scenes_by_dataset = (
            st.session_state.get("project_scenes_by_dataset") or {}
        )

        if scenes_by_dataset:
            st.markdown("### Search results")

            for dataset, scenes in scenes_by_dataset.items():
                dates = distinct_dates(scenes)
                with st.expander(
                    f"{dataset} · {len(scenes)} items · "
                    f"{len(dates)} distinct dates",
                    expanded=True,
                ):
                    if not scenes:
                        st.warning("No observations found.")
                        continue

                    rows = [
                        {
                            "date": s.get("date"),
                            "cloud_cover": s.get("cloud_cover"),
                            "item_id": s.get("item_id"),
                        }
                        for s in scenes
                    ]
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    previews = []
                    for scene in scenes:
                        preview_url = safe_preview_url(scene.get("thumbnail"))
                        if preview_url:
                            previews.append((scene, preview_url))
                        if len(previews) >= 3:
                            break

                    if previews:
                        cols = st.columns(len(previews))
                        for col, (scene, preview_url) in zip(cols, previews):
                            with col:
                                try:
                                    st.image(
                                        preview_url,
                                        caption=str(scene.get("date")),
                                        use_container_width=True,
                                    )
                                except Exception:
                                    st.caption(
                                        f"Preview unavailable for {scene.get('date')}."
                                    )
                    else:
                        st.caption(
                            "No browser-compatible preview is available for these "
                            "STAC items. Scene metadata can still be selected and used."
                        )

            st.markdown("### Choose the scene to process")

            searchable_pairs = flatten_scenes(scenes_by_dataset)
            sentinel_pairs = [
                pair for pair in searchable_pairs
                if pair[0] == "Sentinel-2 Level-2A"
            ]

            if task["supported_product"] == "NDVI" and sentinel_pairs:
                options = list(range(len(sentinel_pairs)))
                default_idx = 0
                selected_index = st.selectbox(
                    "Primary Sentinel-2 scene",
                    options,
                    format_func=lambda i: scene_label(sentinel_pairs[i][1]),
                )
                selected_dataset, selected_scene = sentinel_pairs[selected_index]
                st.session_state["project_primary_dataset"] = selected_dataset
                st.session_state["project_primary_scene"] = selected_scene
                st.success(
                    f"Selected for processing: {scene_label(selected_scene)}"
                )
            elif searchable_pairs:
                options = list(range(len(searchable_pairs)))
                selected_index = st.selectbox(
                    "Primary scene",
                    options,
                    format_func=lambda i: (
                        f"{searchable_pairs[i][0]} · "
                        f"{scene_label(searchable_pairs[i][1])}"
                    ),
                )
                selected_dataset, selected_scene = searchable_pairs[selected_index]
                st.session_state["project_primary_dataset"] = selected_dataset
                st.session_state["project_primary_scene"] = selected_scene

            if st.button(
                "✅ Save scene selection",
                use_container_width=True,
            ):
                save_project_state(
                    project_id, snapshot, task_name, goal, chat
                )
                st.success("Scene selection saved. Open Step 3 — Process.")

    # =========================================================================
    # STEP 3 — PROCESS
    # =========================================================================

    with step3:
        st.subheader(f"Step 3 — Understand, process, and interpret")

        primary_scene = st.session_state.get("project_primary_scene")
        primary_dataset = st.session_state.get("project_primary_dataset")

        if not primary_scene:
            st.warning("Search and select a scene in Step 2 first.")
        else:
            scene_top1, scene_top2 = st.columns([1.35, 0.65])

            with scene_top1:
                st.markdown(
                    f"**Selected dataset:** {primary_dataset}  \n"
                    f"**Selected scene:** {scene_label(primary_scene)}"
                )

            with scene_top2:
                st.metric(
                    "Target indicator",
                    task["indicator"],
                )

            # Change scene without leaving Page 7.
            scenes_by_dataset = (
                st.session_state.get("project_scenes_by_dataset") or {}
            )
            relevant_scenes = scenes_by_dataset.get(primary_dataset, [])

            if relevant_scenes:
                with st.expander("🔄 Change the scene used for this processing run"):
                    idx_lookup = {
                        s.get("item_id"): i
                        for i, s in enumerate(relevant_scenes)
                    }
                    current_idx = idx_lookup.get(
                        primary_scene.get("item_id"),
                        0,
                    )
                    new_idx = st.selectbox(
                        "Scene",
                        list(range(len(relevant_scenes))),
                        index=current_idx,
                        format_func=lambda i: scene_label(relevant_scenes[i]),
                        key="page7_processing_scene",
                    )
                    primary_scene = relevant_scenes[new_idx]
                    st.session_state["project_primary_scene"] = primary_scene

            st.divider()

            # -------------------------------------------------------------
            # LEARN FIRST: concept + method + processing journey.
            # -------------------------------------------------------------
            render_processing_education(task_name)

            st.divider()
            st.markdown("## ▶ Execute this processing step")

            # -------------------------------------------------------------
            # EXECUTABLE PATH 1: Sentinel-2 vegetation products
            # -------------------------------------------------------------
            if task["supported_product"] == "NDVI":
                if primary_dataset != "Sentinel-2 Level-2A":
                    st.warning(
                        "The executable vegetation processor currently uses "
                        "Sentinel-2. Select a Sentinel-2 scene in Step 2 or "
                        "change the processing scene above."
                    )
                else:
                    product = st.selectbox(
                        "Choose the product to calculate",
                        ["NDVI", "Red", "NIR"],
                        help=(
                            "NDVI is the task-level vegetation indicator. "
                            "Red and NIR are also exposed so the user can inspect "
                            "the source bands used in the calculation."
                        ),
                    )

                    p1, p2, p3, p4 = st.columns(4)
                    p1.info("**1 · Read bands**\n\nAccess the required Sentinel-2 COG assets.")
                    p2.info("**2 · Clip AOI**\n\nKeep only pixels inside the study area.")
                    p3.info("**3 · Calculate**\n\nApply the selected spectral operation.")
                    p4.info("**4 · Summarize**\n\nCreate statistics, map, and GeoTIFF.")

                    if st.button(
                        f"⚙️ Run {product} processing",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            progress = st.progress(
                                0,
                                text="Preparing satellite assets...",
                            )
                            progress.progress(
                                20,
                                text="Reading remote spectral band(s)...",
                            )

                            geotiff_bytes, filename, summary = (
                                generate_product_geotiff(
                                    scene=primary_scene,
                                    aoi_geometry=st.session_state[
                                        "aoi_geojson"
                                    ],
                                    product=product,
                                )
                            )

                            progress.progress(
                                70,
                                text="Calculating statistics and preparing GeoTIFF...",
                            )

                            st.session_state["generated_geotiff"] = geotiff_bytes
                            st.session_state["generated_geotiff_name"] = filename
                            st.session_state["generated_geotiff_summary"] = summary

                            history = (
                                st.session_state.get("project_result_history")
                                or []
                            )
                            history.append(
                                {
                                    "dataset": primary_dataset,
                                    "scene": primary_scene.get("item_id"),
                                    "date": primary_scene.get("date"),
                                    "product": product,
                                    "filename": filename,
                                    "summary": summary,
                                }
                            )
                            st.session_state["project_result_history"] = history

                            save_project_state(
                                project_id,
                                snapshot,
                                task_name,
                                goal,
                                chat,
                            )

                            progress.progress(
                                100,
                                text=f"{product} processing completed.",
                            )
                            st.success(
                                f"✅ {product} was calculated from the selected "
                                "scene and clipped to the AOI."
                            )
                        except Exception as exc:
                            st.error(str(exc))

            # -------------------------------------------------------------
            # EXECUTABLE PATH 2: Landsat Level-2 Surface Temperature
            # -------------------------------------------------------------
            elif task_name == "Urban heat island detection":
                if primary_dataset != "Landsat Collection 2 Level-2":
                    st.warning(
                        "LST requires a thermal dataset. Select a Landsat "
                        "Collection 2 Level-2 scene for this processing run."
                    )
                else:
                    lst_cols = st.columns(4)
                    lst_cols[0].info(
                        "**1 · Thermal asset**\n\nLocate the Landsat Level-2 "
                        "Surface Temperature raster."
                    )
                    lst_cols[1].info(
                        "**2 · Physical units**\n\nApply scale/offset and convert "
                        "the result to °C."
                    )
                    lst_cols[2].info(
                        "**3 · AOI statistics**\n\nCalculate mean, min, max, "
                        "and the 90th percentile."
                    )
                    lst_cols[3].info(
                        "**4 · Heat map**\n\nRender cooler-to-hotter spatial "
                        "patterns and save a GeoTIFF."
                    )

                    st.caption(
                        "Why P90? The 90th percentile is a simple descriptive "
                        "threshold for the hotter end of the AOI distribution. "
                        "It is not a universal urban-heat-island threshold."
                    )

                    if st.button(
                        "🌡️ Run Landsat Surface Temperature processing",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            progress = st.progress(
                                0,
                                text="Locating the Surface Temperature asset...",
                            )

                            progress.progress(
                                15,
                                text="Reading and clipping the Landsat thermal raster...",
                            )

                            geotiff_bytes, filename, summary = (
                                generate_landsat_lst_geotiff(
                                    scene=primary_scene,
                                    aoi_geometry=st.session_state[
                                        "aoi_geojson"
                                    ],
                                )
                            )

                            progress.progress(
                                70,
                                text="Converting to Celsius and calculating heat indicators...",
                            )

                            st.session_state["generated_geotiff"] = geotiff_bytes
                            st.session_state["generated_geotiff_name"] = filename
                            st.session_state["generated_geotiff_summary"] = summary

                            history = (
                                st.session_state.get("project_result_history")
                                or []
                            )
                            history.append(
                                {
                                    "dataset": primary_dataset,
                                    "scene": primary_scene.get("item_id"),
                                    "date": primary_scene.get("date"),
                                    "product": "LST_C",
                                    "filename": filename,
                                    "summary": summary,
                                }
                            )
                            st.session_state["project_result_history"] = history

                            save_project_state(
                                project_id,
                                snapshot,
                                task_name,
                                goal,
                                chat,
                            )

                            progress.progress(
                                100,
                                text="Surface Temperature processing completed.",
                            )
                            st.success(
                                "✅ Landsat Surface Temperature was prepared for "
                                "the AOI and converted to Celsius."
                            )
                        except Exception as exc:
                            st.error(
                                "LST processing could not be completed for this "
                                f"scene: {exc}"
                            )
                            st.caption(
                                "If the STAC item does not expose the Landsat "
                                "Surface Temperature asset, select another Level-2 "
                                "scene and retry."
                            )

            # -------------------------------------------------------------
            # GUIDED, NOT YET EXECUTABLE TASKS
            # -------------------------------------------------------------
            else:
                st.warning(
                    "The scientific processing chain is defined and explained "
                    "above, but this specific derived raster is not yet automated "
                    "in the current GeoScope MVP."
                )
                st.markdown(
                    """
**What GeoScope does now**

- keeps the selected AOI, dataset, scene, and dates;
- explains the validated processing chain;
- lets the user select comparison dates;
- uses the Project Assistant to adapt the method and expected outputs;
- clearly separates implemented computation from future/external processing.
"""
                )

            # -------------------------------------------------------------
            # SHOW ACTUAL RESULT, IF ONE EXISTS
            # -------------------------------------------------------------
            geotiff_bytes = st.session_state.get("generated_geotiff")
            summary = st.session_state.get("generated_geotiff_summary")
            filename = st.session_state.get("generated_geotiff_name")

            if geotiff_bytes and summary:
                st.divider()
                st.markdown("## ✅ Result of the processing run")

                unit = summary.get("unit", "")
                k1, k2, k3, k4 = st.columns(4)

                def _metric_value(value: Any) -> str:
                    if value is None:
                        return "n/a"
                    suffix = f" {unit}" if unit else ""
                    return f"{float(value):.2f}{suffix}"

                k1.metric("Mean", _metric_value(summary.get("mean")))
                k2.metric("Minimum", _metric_value(summary.get("minimum")))
                k3.metric("Maximum", _metric_value(summary.get("maximum")))

                if summary.get("p90") is not None:
                    k4.metric("P90", _metric_value(summary.get("p90")))
                else:
                    k4.metric(
                        "Pixels",
                        f"{summary.get('width', '—')} × "
                        f"{summary.get('height', '—')}",
                    )

                render_geotiff_preview(
                    geotiff_bytes,
                    summary.get("product", "result"),
                )

                if str(summary.get("product", "")).upper() == "NDVI":
                    st.success(
                        "**How to read this result:** stronger green tones "
                        "represent higher NDVI / vegetation signal. Do not "
                        "automatically label low NDVI as unhealthy crop: water, "
                        "soil, buildings, shadows, and crop stage also affect it."
                    )

                elif str(summary.get("product", "")).upper() == "LST_C":
                    st.success(
                        "**How to read this result:** hotter colors represent "
                        "higher land-surface temperature. This is surface "
                        "temperature, not weather-station air temperature. "
                        "Compare it with vegetation, water, built-up cover, "
                        "season, and acquisition conditions."
                    )

                st.download_button(
                    "⬇️ Download this processed GeoTIFF",
                    data=geotiff_bytes,
                    file_name=filename,
                    mime="image/tiff",
                    use_container_width=True,
                )

                st.info(
                    "➡️ **Next:** open Step 4 to select other distinct dates "
                    "for comparison, or open the Result map tab to inspect this "
                    "derived raster spatially."
                )

    # =========================================================================
    # STEP 4 — COMPARE DATES
    # =========================================================================

    with step4:
        st.subheader(f"Step 4 — {task['comparison']}")

        scenes_by_dataset = (
            st.session_state.get("project_scenes_by_dataset") or {}
        )

        if not scenes_by_dataset:
            st.warning("Run Step 2 first.")
        else:
            comparison_dataset = st.selectbox(
                "Dataset for comparison",
                list(scenes_by_dataset.keys()),
                key="comparison_dataset",
            )
            comparison_scenes = scenes_by_dataset.get(comparison_dataset, [])
            by_date = best_scene_per_date(comparison_scenes)
            dates = list(by_date.keys())

            st.metric("Distinct acquisition dates available", len(dates))

            if len(dates) < 2:
                st.warning(
                    "At least two distinct acquisition dates are required "
                    "for before/after comparison."
                )
            else:
                default_dates = (
                    st.session_state.get("project_comparison_dates")
                    or dates[: min(3, len(dates))]
                )
                default_dates = [d for d in default_dates if d in dates]

                selected_dates = st.multiselect(
                    "Select comparison dates",
                    dates,
                    default=default_dates,
                )
                st.session_state["project_comparison_dates"] = selected_dates

                if selected_dates:
                    rows = []
                    for d in selected_dates:
                        scene = by_date[d]
                        rows.append(
                            {
                                "date": d,
                                "scene": scene.get("item_id"),
                                "cloud_cover": scene.get("cloud_cover"),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                if task["supported_product"] == "NDVI":
                    st.info(
                        "GeoScope can generate NDVI one selected scene at a time. "
                        "Use Step 3 to generate and save results for different "
                        "selected dates. Automated aligned multi-date raster "
                        "comparison is not yet implemented."
                    )
                else:
                    st.info(
                        "The dates are now selected and tracked. The derived "
                        "multi-date processing for this task is not yet automated."
                    )

                if st.button(
                    "✅ Save comparison date selection",
                    use_container_width=True,
                ):
                    save_project_state(
                        project_id, snapshot, task_name, goal, chat
                    )
                    st.success("Comparison dates saved.")

    # =========================================================================
    # STEP 5 — RESULTS / ASSISTANT / SAVE
    # =========================================================================

    with step5:
        st.subheader(f"Step 5 — Results dashboard")

        scenes_by_dataset = (
            st.session_state.get("project_scenes_by_dataset") or {}
        )
        all_scenes = [
            scene
            for scenes in scenes_by_dataset.values()
            for scene in scenes
        ]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Datasets", len(scenes_by_dataset))
        r2.metric("Scene items", len(all_scenes))
        r3.metric("Distinct dates", len(distinct_dates(all_scenes)))
        r4.metric("Indicator", task["indicator"])

        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown("#### AOI")
            render_aoi_map(
                st.session_state.get("aoi_geojson"),
                key="page7_result_aoi",
            )

        with right:
            st.markdown("#### Scientific interpretation")
            st.write(task["interpretation"])

            summary = st.session_state.get("generated_geotiff_summary")
            if summary:
                st.markdown("#### Latest calculated result")
                s1, s2, s3 = st.columns(3)
                s1.metric(
                    "Mean",
                    f"{summary.get('mean'):.3f}"
                    if summary.get("mean") is not None
                    else "n/a",
                )
                s2.metric(
                    "Minimum",
                    f"{summary.get('minimum'):.3f}"
                    if summary.get("minimum") is not None
                    else "n/a",
                )
                s3.metric(
                    "Maximum",
                    f"{summary.get('maximum'):.3f}"
                    if summary.get("maximum") is not None
                    else "n/a",
                )

        geotiff_bytes = st.session_state.get("generated_geotiff")
        summary = st.session_state.get("generated_geotiff_summary")
        if geotiff_bytes and summary:
            st.markdown("### Derived product")
            render_geotiff_preview(
                geotiff_bytes,
                summary.get("product", "NDVI"),
            )

        history = st.session_state.get("project_result_history") or []
        if history:
            st.markdown("### Saved result history")
            history_rows = []
            for item in history:
                s = item.get("summary") or {}
                history_rows.append(
                    {
                        "date": item.get("date"),
                        "dataset": item.get("dataset"),
                        "product": item.get("product"),
                        "mean": s.get("mean"),
                        "minimum": s.get("minimum"),
                        "maximum": s.get("maximum"),
                        "filename": item.get("filename"),
                    }
                )
            st.dataframe(
                pd.DataFrame(history_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### 🧠 Ask what to do next")

        context = project_context(task_name, goal, snapshot)

        if st.button(
            "Generate next-step guidance",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("GeoScope is reviewing the project state..."):
                guidance = generate_text(
                    instructions=ASSISTANT_RULES,
                    prompt=(
                        f"{context}\n\n"
                        f"Target indicator: {task['indicator']}\n"
                        f"Task interpretation: {task['interpretation']}\n\n"
                        "Give the single most useful next project action."
                    ),
                    model=get_generation_model(),
                )
            st.session_state["project_guidance"] = guidance

        if st.session_state.get("project_guidance"):
            st.info(st.session_state["project_guidance"])

        for message in chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask about this EO project..."
        )
        if question:
            chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("GeoScope is reasoning about the project..."):
                    answer = generate_text(
                        instructions=ASSISTANT_RULES,
                        prompt=(
                            f"{project_context(task_name, goal, snapshot)}\n\n"
                            f"QUESTION:\n{question}"
                        ),
                        model=get_generation_model(),
                    )
                st.markdown(answer)
            chat.append({"role": "assistant", "content": answer})

        st.divider()
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button(
                "💾 Save project state + results",
                type="primary",
                use_container_width=True,
            ):
                save_project_state(
                    project_id, snapshot, task_name, goal, chat
                )
                st.success("Project saved.")

        with b2:
            if st.button(
                "✅ Mark project completed",
                use_container_width=True,
            ):
                save_project_state(
                    project_id, snapshot, task_name, goal, chat
                )
                set_project_status(project_id, "COMPLETED")
                st.rerun()

        with b3:
            if st.button(
                "Close project",
                use_container_width=True,
            ):
                st.session_state.pop("active_project_id", None)
                st.rerun()

        with st.expander("AI assistance"):
            st.write(f"**Provider:** {get_provider()}")
            st.write(f"**Model:** {get_generation_model()}")
            st.caption(
                "The LLM guides the workflow; STAC and geospatial functions "
                "provide the executable evidence and calculations."
            )

    # =========================================================================
    # STEP 6 — RESULT / INDEX MAP
    # =========================================================================

    with step6:
        current_aoi = st.session_state.get("aoi_geojson")
        current_geotiff = st.session_state.get("generated_geotiff")
        current_summary = st.session_state.get("generated_geotiff_summary") or {}
        current_product = str(current_summary.get("product") or "").upper()

        # ---------------------------------------------------------------------
        # CROP MONITORING: keep this tab focused on vegetation indices/results.
        # ---------------------------------------------------------------------
        if task_name == "Crop monitoring":
            st.subheader("🌱 Crop vegetation indices")
            st.caption(
                "This view focuses on the indicators used to understand crop "
                "and vegetation condition. Only indices actually calculated by "
                "GeoScope are shown as results. The NDVI map includes a visible "
                "color-scale legend so vegetation intensity can be read directly."
            )

            i1, i2, i3, i4 = st.columns(4)

            with i1:
                st.markdown("### 🟢 NDVI")
                st.caption("Implemented")
                st.write(
                    "Vegetation greenness / vigor signal using Red and NIR."
                )
                st.code("NDVI = (NIR - Red) / (NIR + Red)")

            with i2:
                st.markdown("### EVI")
                st.caption("Explained / future")
                st.write(
                    "Useful in dense vegetation and designed to reduce some "
                    "atmospheric and background effects."
                )

            with i3:
                st.markdown("### SAVI")
                st.caption("Explained / future")
                st.write(
                    "Vegetation index that reduces soil-background influence, "
                    "especially when crop cover is sparse."
                )

            with i4:
                st.markdown("### NDMI")
                st.caption("Explained / future")
                st.write(
                    "Uses NIR and SWIR to indicate vegetation/canopy moisture."
                )

            st.divider()

            if current_geotiff and current_product == "NDVI":
                st.markdown("## 🗺️ NDVI result")

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Mean NDVI",
                    f"{current_summary.get('mean'):.3f}"
                    if current_summary.get("mean") is not None
                    else "n/a",
                )
                m2.metric(
                    "Minimum NDVI",
                    f"{current_summary.get('minimum'):.3f}"
                    if current_summary.get("minimum") is not None
                    else "n/a",
                )
                m3.metric(
                    "Maximum NDVI",
                    f"{current_summary.get('maximum'):.3f}"
                    if current_summary.get("maximum") is not None
                    else "n/a",
                )

                render_result_map(
                    current_aoi,
                    current_geotiff,
                    "NDVI",
                )

                st.markdown(
                    """
**How to read the NDVI map**

- brown / yellow → lower vegetation signal
- light green → moderate vegetation signal
- darker green → higher vegetation signal

NDVI should be interpreted with crop stage, irrigation, soil, clouds,
shadows, and field knowledge. A low value does **not** automatically mean
an unhealthy crop.
"""
                )

            else:
                st.info(
                    "No NDVI raster has been generated for the current crop "
                    "project yet. Go to **Step 3 — Process**, select a "
                    "Sentinel-2 scene, and run **NDVI**. The resulting index "
                    "will then be displayed here on the map."
                )

            history = st.session_state.get("project_result_history") or []
            ndvi_history = [
                item
                for item in history
                if str(item.get("product", "")).upper() == "NDVI"
            ]

            if ndvi_history:
                st.markdown("### 📈 Generated NDVI observations")
                rows = []
                for item in ndvi_history:
                    summary = item.get("summary") or {}
                    rows.append(
                        {
                            "date": item.get("date"),
                            "scene": item.get("scene"),
                            "mean_ndvi": summary.get("mean"),
                            "min_ndvi": summary.get("minimum"),
                            "max_ndvi": summary.get("maximum"),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

                if len(rows) >= 2:
                    st.success(
                        "Several NDVI runs are available. These values can be "
                        "used as a simple temporal comparison, while remembering "
                        "that GeoScope does not yet build an aligned multi-date "
                        "raster cube automatically."
                    )

        # ---------------------------------------------------------------------
        # URBAN HEAT: show only LST result when available.
        # ---------------------------------------------------------------------
        elif task_name == "Urban heat island detection":
            st.subheader("🌡️ Surface temperature result")
            st.caption(
                "This view focuses on the heat indicator produced by the "
                "selected Landsat Surface Temperature observation."
            )

            if current_geotiff and current_product == "LST_C":
                map_m1, map_m2, map_m3, map_m4 = st.columns(4)
                map_m1.metric("Indicator", "LST")
                map_m2.metric(
                    "Mean",
                    f"{current_summary.get('mean'):.2f} °C"
                    if current_summary.get("mean") is not None
                    else "n/a",
                )
                map_m3.metric(
                    "Maximum",
                    f"{current_summary.get('maximum'):.2f} °C"
                    if current_summary.get("maximum") is not None
                    else "n/a",
                )
                map_m4.metric(
                    "P90",
                    f"{current_summary.get('p90'):.2f} °C"
                    if current_summary.get("p90") is not None
                    else "n/a",
                )

                render_result_map(
                    current_aoi,
                    current_geotiff,
                    "LST_C",
                )

                st.markdown(
                    """
**LST interpretation**

- cooler colors → relatively cooler surfaces
- yellow / orange → warmer surfaces
- red → hotter surfaces

LST measures **surface temperature**, not air temperature.
"""
                )
            else:
                st.info(
                    "No executable LST result is available yet. The Landsat "
                    "processing workflow is explained in Step 3, but remote "
                    "asset access may still require another provider or "
                    "authenticated access."
                )

        # ---------------------------------------------------------------------
        # OTHER TASKS: show the target indicator and expected result only.
        # ---------------------------------------------------------------------
        else:
            st.subheader(f"🗺️ {task['indicator']} result")
            st.caption(
                "This task is currently guided rather than fully automated. "
                "GeoScope explains the indicator, required processing, and "
                "expected map without fabricating a raster result."
            )

            st.info(
                f"**Target indicator:** {task['indicator']}  \n\n"
                f"**Processing objective:** {task['processing']}  \n\n"
                f"**Interpretation:** {task['interpretation']}"
            )

            if current_geotiff and current_product:
                render_result_map(
                    current_aoi,
                    current_geotiff,
                    current_product,
                )

