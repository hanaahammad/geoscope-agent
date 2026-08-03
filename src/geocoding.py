from __future__ import annotations

from typing import Any

import requests
from shapely.geometry import box, mapping, shape


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "GeoScope-Agent/1.0 educational-capstone"


def _geometry_from_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a Polygon or MultiPolygon suitable for an AOI.

    Nominatim may return a point for some place records. In that case,
    use the returned bounding box as the AOI.
    """
    geojson = result.get("geojson")

    if geojson and geojson.get("type") in {
        "Polygon",
        "MultiPolygon",
    }:
        return mapping(shape(geojson))

    bounding_box = result.get("boundingbox")

    if not bounding_box or len(bounding_box) != 4:
        raise ValueError(
            "The selected place does not provide a usable polygon "
            "or bounding box."
        )

    south, north, west, east = map(float, bounding_box)

    return mapping(
        box(
            min(west, east),
            min(south, north),
            max(west, east),
            max(south, north),
        )
    )


def search_places(
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Search OpenStreetMap Nominatim using a free-form place name.

    This function is intended for interactive, low-volume use only.
    """
    cleaned_query = query.strip()

    if len(cleaned_query) < 3:
        raise ValueError(
            "Enter at least three characters for the place search."
        )

    response = requests.get(
        NOMINATIM_SEARCH_URL,
        params={
            "q": cleaned_query,
            "format": "jsonv2",
            "polygon_geojson": 1,
            "polygon_threshold": 0.0005,
            "addressdetails": 1,
            "limit": int(limit),
        },
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en",
        },
        timeout=30,
    )
    response.raise_for_status()

    results = []

    for item in response.json():
        try:
            geometry = _geometry_from_result(item)
        except ValueError:
            continue

        results.append(
            {
                "display_name": item.get(
                    "display_name",
                    "Unnamed place",
                ),
                "place_id": item.get("place_id"),
                "category": item.get("category"),
                "type": item.get("type"),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "boundingbox": item.get("boundingbox"),
                "geometry": geometry,
            }
        )

    return results
