from __future__ import annotations

from typing import Any

from pystac_client import Client


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-c1-l2a"


def search_sentinel2(
    aoi_geometry: dict[str, Any],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search Sentinel-2 Level-2A scenes intersecting a GeoJSON AOI.

    Parameters
    ----------
    aoi_geometry:
        GeoJSON geometry returned by the Streamlit/Folium drawing tool.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.
    max_cloud_cover:
        Maximum accepted cloud-cover percentage.
    limit:
        Maximum number of scenes to return.
    """
    if not aoi_geometry:
        raise ValueError("An AOI geometry is required.")

    if start_date > end_date:
        raise ValueError("The start date must be before the end date.")

    catalog = Client.open(EARTH_SEARCH_URL)

    search = catalog.search(
        collections=[DEFAULT_COLLECTION],
        intersects=aoi_geometry,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lte": float(max_cloud_cover)
            }
        },
        max_items=int(limit),
    )

    results: list[dict[str, Any]] = []

    for item in search.items():
        thumbnail = None

        for asset_name in (
            "thumbnail",
            "rendered_preview",
            "visual",
        ):
            if asset_name in item.assets:
                thumbnail = item.assets[asset_name].href
                break

        results.append(
            {
                "item_id": item.id,
                "date": (
                    item.datetime.date().isoformat()
                    if item.datetime
                    else None
                ),
                "cloud_cover": item.properties.get(
                    "eo:cloud_cover"
                ),
                "collection": item.collection_id,
                "bbox": item.bbox,
                "geometry": item.geometry,
                "thumbnail": thumbnail,
                "available_assets": list(item.assets.keys()),
            }
        )

    return results