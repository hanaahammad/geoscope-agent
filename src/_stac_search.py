from __future__ import annotations

from datetime import date
from typing import Any

from pystac_client import Client


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-c1-l2a"


def _asset_metadata(asset: Any) -> dict[str, Any]:
    extra = asset.extra_fields or {}
    eo_bands = extra.get("eo:bands", [])
    raster_bands = extra.get("raster:bands", [])

    return {
        "href": asset.href,
        "type": asset.media_type,
        "title": asset.title,
        "roles": list(asset.roles or []),
        "eo_bands": eo_bands,
        "raster_bands": raster_bands,
        "gsd": extra.get("gsd"),
    }


def search_sentinel2(
    aoi_geometry: dict[str, Any],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search Sentinel-2 Collection 1 Level-2A scenes intersecting an AOI.
    """
    if not aoi_geometry:
        raise ValueError("An AOI geometry is required.")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError(
            "The start date must be before or equal to the end date."
        )

    catalog = Client.open(EARTH_SEARCH_URL)

    search = catalog.search(
        collections=[DEFAULT_COLLECTION],
        intersects=aoi_geometry,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={
            "eo:cloud_cover": {
                "lte": float(max_cloud_cover)
            }
        },
        max_items=int(limit),
    )

    results: list[dict[str, Any]] = []

    for item in search.items():
        item_date = (
            item.datetime.date().isoformat()
            if item.datetime
            else None
        )

        if item_date:
            parsed = date.fromisoformat(item_date)
            if not start <= parsed <= end:
                continue

        cloud_cover = item.properties.get("eo:cloud_cover")

        if (
            cloud_cover is not None
            and float(cloud_cover) > float(max_cloud_cover)
        ):
            continue

        assets = {
            name: _asset_metadata(asset)
            for name, asset in item.assets.items()
        }

        thumbnail = None

        for asset_name in (
            "thumbnail",
            "rendered_preview",
            "visual",
        ):
            if asset_name in assets:
                thumbnail = assets[asset_name]["href"]
                break

        results.append(
            {
                "item_id": item.id,
                "date": item_date,
                "cloud_cover": cloud_cover,
                "collection": item.collection_id,
                "bbox": item.bbox,
                "geometry": item.geometry,
                "thumbnail": thumbnail,
                "assets": assets,
                "available_assets": list(assets.keys()),
            }
        )

    results.sort(
        key=lambda item: (
            item.get("cloud_cover")
            if item.get("cloud_cover") is not None
            else 999,
            item.get("date") or "",
        )
    )

    return results[: int(limit)]


def find_band_asset(
    scene: dict[str, Any],
    common_name: str,
) -> tuple[str, dict[str, Any]]:
    """
    Find a STAC asset by EO common name, with common key fallbacks.
    """
    assets = scene.get("assets", {})
    common_name = common_name.lower()

    preferred_keys = {
        "red": ["red", "B04", "b04"],
        "nir": ["nir", "nir08", "B08", "b08"],
    }

    for key in preferred_keys.get(common_name, []):
        if key in assets:
            return key, assets[key]

    for key, metadata in assets.items():
        for band in metadata.get("eo_bands", []):
            if (
                str(band.get("common_name", "")).lower()
                == common_name
            ):
                return key, metadata

    raise KeyError(
        f"No {common_name} asset was found for scene "
        f"{scene.get('item_id')}."
    )
