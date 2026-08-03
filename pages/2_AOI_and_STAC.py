from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import mapping, shape
from streamlit_folium import st_folium

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geocoding import search_places
from src.geotiff_processing import generate_product_geotiff
from src.stac_search import search_sentinel2


st.set_page_config(
    page_title="AOI, STAC and GeoTIFF",
    page_icon="🗺️",
    layout="wide",
)

apply_global_style()

st.title("🗺️ Step 2 — AOI, STAC Search and GeoTIFF")

st.markdown(
    """
This page can now execute a small remote-sensing workflow:

```text
Define AOI → search STAC → select one scene
→ clip Red/NIR or calculate NDVI → download GeoTIFF
```

The AOI can be drawn on the map or found from a place name such as
**Kom Ombo, Aswan, Egypt**.
"""
)


def geometry_signature(geometry: dict | None) -> str:
    if not geometry:
        return ""

    return json.dumps(
        geometry,
        sort_keys=True,
        separators=(",", ":"),
    )


def aoi_summary(
    geometry_dict: dict,
    label: str | None = None,
) -> str:
    geometry = shape(geometry_dict)
    centroid = geometry.centroid
    minx, miny, maxx, maxy = geometry.bounds
    prefix = f"{label}; " if label else ""

    return (
        f"{prefix}{geometry_dict.get('type', 'Geometry')} AOI; "
        f"centroid {centroid.y:.4f}, {centroid.x:.4f}; "
        f"bbox [{minx:.4f}, {miny:.4f}, "
        f"{maxx:.4f}, {maxy:.4f}]"
    )


def save_aoi(
    geometry: dict,
    *,
    label: str | None = None,
) -> None:
    normalized = dict(mapping(shape(geometry)))

    st.session_state["aoi_geojson"] = normalized
    st.session_state["aoi_summary"] = aoi_summary(
        normalized,
        label,
    )
    st.session_state["aoi_label"] = label or "Drawn AOI"

    for key in (
        "stac_scenes",
        "last_search_signature",
        "generated_geotiff",
        "generated_geotiff_name",
        "generated_geotiff_summary",
    ):
        st.session_state.pop(key, None)


if "stac_start_date" not in st.session_state:
    st.session_state["stac_start_date"] = (
        date.today() - timedelta(days=60)
    )

if "stac_end_date" not in st.session_state:
    st.session_state["stac_end_date"] = date.today()

if "stac_max_cloud_cover" not in st.session_state:
    st.session_state["stac_max_cloud_cover"] = 20

if "stac_scene_limit" not in st.session_state:
    st.session_state["stac_scene_limit"] = 10


st.subheader("1. Define the Area of Interest")

draw_tab, text_tab = st.tabs(
    [
        "Draw on the map",
        "Search by place name",
    ]
)

with draw_tab:
    stored_aoi = st.session_state.get("aoi_geojson")

    if stored_aoi:
        centroid = shape(stored_aoi).centroid
        location = [centroid.y, centroid.x]
        zoom = 10
    else:
        location = [24.5, 32.9]
        zoom = 6

    fmap = folium.Map(
        location=location,
        zoom_start=zoom,
        control_scale=True,
    )

    if stored_aoi:
        folium.GeoJson(
            stored_aoi,
            name="Current AOI",
            style_function=lambda _: {
                "color": "#4F7D68",
                "weight": 3,
                "fillOpacity": 0.12,
            },
        ).add_to(fmap)

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
        height=500,
        returned_objects=[
            "all_drawings",
            "last_active_drawing",
        ],
        key="aoi_map_v3",
    )

    drawings = (
        output.get("all_drawings", [])
        if output
        else []
    )

    candidate = None

    if drawings:
        candidate = drawings[-1].get("geometry")
    elif output and output.get("last_active_drawing"):
        candidate = output[
            "last_active_drawing"
        ].get("geometry")

    if (
        candidate
        and geometry_signature(candidate)
        != geometry_signature(stored_aoi)
    ):
        save_aoi(candidate, label="Drawn on map")
        st.success(
            "The new map AOI was saved and old STAC results were cleared."
        )

with text_tab:
    place_query = st.text_input(
        "Place name",
        value="Kom Ombo, Aswan, Egypt",
        placeholder="Kom Ombo, Aswan, Egypt",
    )

    if st.button(
        "Search place",
        use_container_width=True,
    ):
        try:
            with st.spinner("Searching OpenStreetMap Nominatim..."):
                st.session_state["place_results"] = search_places(
                    place_query,
                    limit=5,
                )
        except Exception as exc:
            st.error(str(exc))

    place_results = st.session_state.get(
        "place_results",
        [],
    )

    if place_results:
        labels = [
            result["display_name"]
            for result in place_results
        ]

        selected_label = st.selectbox(
            "Select the correct result",
            labels,
        )

        selected = place_results[
            labels.index(selected_label)
        ]

        st.write(f"**Category:** {selected['category']}")
        st.write(f"**Type:** {selected['type']}")
        st.write(
            f"**Centre:** {selected['lat']:.5f}, "
            f"{selected['lon']:.5f}"
        )

        preview_map = folium.Map(
            location=[
                selected["lat"],
                selected["lon"],
            ],
            zoom_start=10,
        )

        folium.GeoJson(
            selected["geometry"],
            style_function=lambda _: {
                "color": "#4F7D68",
                "weight": 3,
                "fillOpacity": 0.18,
            },
        ).add_to(preview_map)

        preview_map.fit_bounds(
            shape(
                selected["geometry"]
            ).bounds[1::-1]
            if False
            else [
                [
                    shape(selected["geometry"]).bounds[1],
                    shape(selected["geometry"]).bounds[0],
                ],
                [
                    shape(selected["geometry"]).bounds[3],
                    shape(selected["geometry"]).bounds[2],
                ],
            ]
        )

        st_folium(
            preview_map,
            width=None,
            height=350,
            key="place_preview_map",
        )

        if st.button(
            "Use this place as the AOI",
            type="primary",
            use_container_width=True,
        ):
            save_aoi(
                selected["geometry"],
                label=selected["display_name"],
            )
            st.success(
                "The selected place is now the active AOI."
            )
            st.rerun()


active_aoi = st.session_state.get("aoi_geojson")

if active_aoi:
    st.success(
        f"**Active AOI:** "
        f"{st.session_state.get('aoi_summary')}"
    )
else:
    st.warning("Define an AOI before searching STAC.")


st.divider()
st.subheader("2. Search Sentinel-2 scenes")

c1, c2, c3, c4 = st.columns(4)

with c1:
    start_date = st.date_input(
        "Start date",
        key="stac_start_date",
    )

with c2:
    end_date = st.date_input(
        "End date",
        key="stac_end_date",
    )

with c3:
    max_cloud = st.slider(
        "Max cloud cover (%)",
        0,
        100,
        key="stac_max_cloud_cover",
    )

with c4:
    scene_limit = st.slider(
        "Max scenes",
        1,
        20,
        key="stac_scene_limit",
    )

invalid_dates = start_date > end_date

if invalid_dates:
    st.error(
        "The start date must be before or equal to the end date."
    )

search_signature = {
    "aoi": geometry_signature(active_aoi),
    "start": start_date.isoformat(),
    "end": end_date.isoformat(),
    "cloud": int(max_cloud),
    "limit": int(scene_limit),
}

if st.button(
    "Search Sentinel-2",
    type="primary",
    use_container_width=True,
    disabled=(not active_aoi or invalid_dates),
):
    try:
        with st.spinner("Querying Earth Search..."):
            scenes = search_sentinel2(
                aoi_geometry=active_aoi,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                max_cloud_cover=max_cloud,
                limit=scene_limit,
            )

        st.session_state["stac_scenes"] = scenes
        st.session_state["last_search_signature"] = (
            search_signature
        )
        st.session_state["start_date"] = (
            start_date.isoformat()
        )
        st.session_state["end_date"] = (
            end_date.isoformat()
        )
        st.session_state["max_cloud_cover"] = max_cloud

        st.success(
            f"Found {len(scenes)} matching scene items."
        )

    except Exception as exc:
        st.error(str(exc))


scenes = st.session_state.get("stac_scenes", [])

if (
    scenes
    and st.session_state.get("last_search_signature")
    == search_signature
):
    unique_dates = sorted(
        {
            scene.get("date")
            for scene in scenes
            if scene.get("date")
        }
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Scene items", len(scenes))
    m2.metric("Distinct dates", len(unique_dates))
    m3.metric(
        "Time series",
        "Possible" if len(unique_dates) >= 2 else "Not yet",
    )

    if len(unique_dates) < 2:
        st.warning(
            "Several scene items may belong to the same acquisition date. "
            "At least two distinct dates are required for a time series."
        )

    scene_rows = [
        {
            "item_id": scene["item_id"],
            "date": scene.get("date"),
            "cloud_cover": scene.get("cloud_cover"),
            "asset_count": len(
                scene.get("assets", {})
            ),
        }
        for scene in scenes
    ]

    st.dataframe(
        pd.DataFrame(scene_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("3. Generate a clipped GeoTIFF")

    scene_labels = [
        (
            f"{scene.get('date')} | "
            f"cloud {scene.get('cloud_cover')}% | "
            f"{scene.get('item_id')}"
        )
        for scene in scenes
    ]

    selected_label = st.selectbox(
        "Select one scene",
        scene_labels,
    )

    selected_scene = scenes[
        scene_labels.index(selected_label)
    ]

    product = st.radio(
        "Output product",
        ["NDVI", "Red", "NIR"],
        horizontal=True,
    )

    st.info(
        "GeoScope reads the cloud-optimized GeoTIFF assets, clips them "
        "to the AOI, and creates the requested single-band GeoTIFF."
    )

    if st.button(
        "Generate GeoTIFF",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Reading the selected COG and processing the AOI..."
            ):
                geotiff, filename, summary = (
                    generate_product_geotiff(
                        scene=selected_scene,
                        aoi_geometry=active_aoi,
                        product=product,
                    )
                )

            st.session_state["generated_geotiff"] = (
                geotiff
            )
            st.session_state[
                "generated_geotiff_name"
            ] = filename
            st.session_state[
                "generated_geotiff_summary"
            ] = summary

            st.success("GeoTIFF generated successfully.")

        except Exception as exc:
            st.error(
                "GeoTIFF processing failed. "
                f"Technical detail: {exc}"
            )

    generated = st.session_state.get(
        "generated_geotiff"
    )

    if generated:
        summary = st.session_state[
            "generated_geotiff_summary"
        ]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Product", summary["product"])
        s2.metric(
            "Raster size",
            f"{summary['width']} × {summary['height']}",
        )
        s3.metric(
            "Mean",
            (
                f"{summary['mean']:.4f}"
                if summary["mean"] is not None
                else "N/A"
            ),
        )
        s4.metric("CRS", summary["crs"])

        st.download_button(
            "Download GeoTIFF",
            data=generated,
            file_name=st.session_state[
                "generated_geotiff_name"
            ],
            mime="image/tiff",
            type="primary",
            use_container_width=True,
        )
