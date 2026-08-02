from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import folium
import pandas as pd
import streamlit as st

from src.ui import apply_global_style
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stac_search import search_sentinel2


st.set_page_config(
    page_title="AOI and STAC",
    page_icon="🗺️",
    layout="wide",
)

apply_global_style()

st.title("🗺️ Step 2 — AOI Selection and STAC Search")

st.markdown(
    """
### What is implemented on this page?

This page connects the geographic question to actual satellite availability.

- **AOI — Area of Interest:** the polygon or rectangle that defines where
  the analysis should take place.
- **STAC — SpatioTemporal Asset Catalog:** a standard way to search
  geospatial datasets by location, date, collection, and metadata.
- **Sentinel-2:** an optical Earth-observation mission commonly used for
  vegetation, agriculture, water, and land-cover analysis.

GeoScope sends the selected AOI and date filters to the STAC API and returns
matching Sentinel-2 scene metadata. The MVP uses scene metadata and previews;
it does not download the full satellite bands.
"""
)

with st.expander("Why does GeoScope use both documents and STAC?"):
    st.markdown(
        """
The knowledge-base documents explain **how** a dataset should be used.
The STAC API checks **whether relevant scenes are actually available**
for the selected AOI and time period.

Together they support a more practical recommendation:

```text
Technical guidance + AOI + date range + available scenes
→ dataset and workflow recommendation
```
"""
    )

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("1. Draw the Area of Interest")

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

    output = st_folium(
        fmap,
        width=None,
        height=520,
        returned_objects=["last_active_drawing"],
        key="aoi_stac_map",
    )

    drawing = output.get("last_active_drawing") if output else None
    aoi_geojson = drawing.get("geometry") if drawing else None

    if aoi_geojson:
        geometry = shape(aoi_geojson)
        centroid = geometry.centroid
        minx, miny, maxx, maxy = geometry.bounds

        aoi_summary = (
            f"{aoi_geojson.get('type', 'Geometry')} AOI; "
            f"centroid latitude {centroid.y:.4f}, "
            f"longitude {centroid.x:.4f}; "
            f"bounding box [{minx:.4f}, {miny:.4f}, "
            f"{maxx:.4f}, {maxy:.4f}]"
        )

        st.session_state["aoi_geojson"] = aoi_geojson
        st.session_state["aoi_summary"] = aoi_summary

        st.success("AOI captured and stored for the next pages.")
        st.write(f"**AOI description:** {aoi_summary}")

        with st.expander("View AOI geometry"):
            st.json(aoi_geojson)

    else:
        aoi_geojson = st.session_state.get("aoi_geojson")
        aoi_summary = st.session_state.get("aoi_summary")

        if aoi_geojson:
            st.info("Using the AOI already stored in this session.")
            if aoi_summary:
                st.write(f"**AOI description:** {aoi_summary}")
        else:
            st.warning("Draw a polygon or rectangle first.")

with right:
    st.subheader("2. Search the satellite catalogue")

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
        0,
        100,
        20,
    )

    scene_limit = st.slider(
        "Maximum returned scenes",
        1,
        20,
        5,
    )

    st.caption(
        "A lower cloud threshold may improve optical-image quality, "
        "but it can also return fewer scenes."
    )

    if st.button(
        "Search Sentinel-2 scenes",
        type="primary",
        use_container_width=True,
    ):
        try:
            if not aoi_geojson:
                raise ValueError("Please draw an AOI first.")

            with st.spinner("Querying Earth Search STAC..."):
                scenes = search_sentinel2(
                    aoi_geometry=aoi_geojson,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    max_cloud_cover=max_cloud_cover,
                    limit=scene_limit,
                )

            st.session_state["stac_scenes"] = scenes
            st.session_state["start_date"] = start_date.isoformat()
            st.session_state["end_date"] = end_date.isoformat()
            st.session_state["max_cloud_cover"] = max_cloud_cover

            st.success(f"Found {len(scenes)} matching scenes.")

        except Exception as exc:
            st.error(str(exc))

scenes = st.session_state.get("stac_scenes", [])

if scenes:
    st.divider()
    st.subheader("3. Review matching scenes")

    rows = [
        {
            "item_id": scene.get("item_id"),
            "date": scene.get("date"),
            "cloud_cover": scene.get("cloud_cover"),
            "collection": scene.get("collection"),
        }
        for scene in scenes
    ]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    lowest_cloud = min(
        scenes,
        key=lambda item: (
            item.get("cloud_cover")
            if item.get("cloud_cover") is not None
            else 999
        ),
    )

    st.info(
        "Lowest-cloud scene in the returned results: "
        f"{lowest_cloud.get('item_id')} "
        f"({lowest_cloud.get('cloud_cover')}% cloud cover)."
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
                f"Cloud cover: {scene.get('cloud_cover')}%"
            ),
        )
