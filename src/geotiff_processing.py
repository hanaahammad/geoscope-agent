from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import rasterio
import requests
import urllib3
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import transform_geom

from src.stac_search import find_band_asset


# ---------------------------------------------------------------------------
# SSL configuration
# ---------------------------------------------------------------------------
# Temporary demo workaround for a corporate HTTPS-inspection proxy.
#
# YES:
#   - Requests may download the remote GeoTIFF with verify=False.
#   - This is only appropriate for local testing.
#
# NO:
#   - Normal TLS verification is used.
#   - Configure the organization's root CA using REQUESTS_CA_BUNDLE
#     and/or install it in the Windows Trusted Root store.
#
ALLOW_INSECURE_SSL = (
    os.getenv("GEOSCOPE_ALLOW_INSECURE_SSL", "YES")
    .strip()
    .upper()
    in {"1", "TRUE", "YES", "ON"}
)

if ALLOW_INSECURE_SSL:
    urllib3.disable_warnings(
        urllib3.exceptions.InsecureRequestWarning
    )


def _ca_bundle() -> str | None:
    """
    Return a configured CA bundle path when available.
    """
    return (
        os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("GDAL_CURL_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )


def _requests_verify() -> bool | str:
    """
    TLS verification value passed to requests.
    """
    ca_path = _ca_bundle()

    if ca_path:
        return ca_path

    if ALLOW_INSECURE_SSL:
        return False

    return True


def _build_gdal_environment() -> dict[str, Any]:
    """
    GDAL options used for direct remote COG access.

    Direct access is attempted first. If Windows Schannel still rejects
    the corporate certificate, GeoScope falls back to downloading the
    asset with requests and opening the local temporary file.
    """
    options: dict[str, Any] = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_HTTP_TIMEOUT": "60",
        "GDAL_HTTP_CONNECTTIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "2",
        "GDAL_HTTP_RETRY_DELAY": "2",
    }

    ca_path = _ca_bundle()

    if ca_path:
        options["GDAL_CURL_CA_BUNDLE"] = ca_path
        options["CURL_CA_BUNDLE"] = ca_path
        options["SSL_CERT_FILE"] = ca_path

    if ALLOW_INSECURE_SSL:
        options["GDAL_HTTP_UNSAFESSL"] = "YES"

    return options


def _is_certificate_error(exc: Exception) -> bool:
    """
    Detect common Windows/GDAL TLS certificate failures.
    """
    message = str(exc).lower()

    indicators = (
        "cert_trust_is_untrusted_root",
        "certificate verify failed",
        "ssl certificate problem",
        "self-signed certificate",
        "untrusted root",
        "schannel",
    )

    return any(
        indicator in message
        for indicator in indicators
    )


@contextmanager
def _download_remote_asset(
    href: str,
) -> Iterator[str]:
    """
    Download a remote GeoTIFF to a temporary local file.

    This fallback avoids GDAL/libcurl Schannel certificate handling.
    The temporary file is deleted after Rasterio finishes reading it.
    """
    temp_path: str | None = None

    try:
        with requests.get(
            href,
            stream=True,
            timeout=(30, 300),
            verify=_requests_verify(),
            headers={
                "User-Agent": (
                    "GeoScope-Agent/1.0 educational-capstone"
                )
            },
        ) as response:
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                suffix=".tif",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024,
                ):
                    if chunk:
                        temp_file.write(chunk)

        if not temp_path:
            raise RuntimeError(
                "The temporary GeoTIFF file was not created."
            )

        yield temp_path

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "GeoScope could not download the remote GeoTIFF "
            f"through the fallback HTTP client: {exc}"
        ) from exc

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _scale_and_offset(
    asset: dict[str, Any],
) -> tuple[float, float]:
    """
    Read scale and offset from STAC raster metadata.
    """
    raster_bands = asset.get("raster_bands") or []

    if not raster_bands:
        return 1.0, 0.0

    metadata = raster_bands[0] or {}

    return (
        float(metadata.get("scale", 1.0)),
        float(metadata.get("offset", 0.0)),
    )


def _extract_clipped_data(
    dataset: rasterio.io.DatasetReader,
    asset: dict[str, Any],
    aoi_geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """
    Clip an already opened raster dataset to the WGS84 AOI.
    """
    if dataset.crs is None:
        raise ValueError(
            "The selected raster has no coordinate reference system."
        )

    projected_geometry = transform_geom(
        "EPSG:4326",
        dataset.crs,
        aoi_geometry,
        precision=6,
    )

    clipped, output_transform = mask(
        dataset,
        [projected_geometry],
        crop=True,
        filled=False,
        indexes=1,
    )

    if clipped.ndim == 3:
        clipped = clipped[0]

    data = clipped.astype("float32")
    invalid_mask = np.ma.getmaskarray(clipped)

    scale, offset = _scale_and_offset(asset)
    data = data * scale + offset

    profile = dataset.profile.copy()
    profile.update(
        {
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": output_transform,
            "count": 1,
            "driver": "GTiff",
            "compress": "deflate",
            "tiled": True,
        }
    )

    return data, profile, invalid_mask


def _read_clipped_band(
    asset: dict[str, Any],
    aoi_geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """
    Read and clip one Sentinel-2 band.

    Strategy:
    1. Try direct COG access through Rasterio/GDAL.
    2. On a certificate error, download with requests and process locally.
    """
    href = asset.get("href")

    if not href:
        raise ValueError(
            "The selected STAC asset has no URL."
        )

    direct_error: Exception | None = None

    try:
        with rasterio.Env(**_build_gdal_environment()):
            with rasterio.open(href) as dataset:
                return _extract_clipped_data(
                    dataset,
                    asset,
                    aoi_geometry,
                )

    except rasterio.errors.RasterioIOError as exc:
        direct_error = exc

        if not _is_certificate_error(exc):
            raise RuntimeError(
                "Rasterio could not open the selected remote "
                f"GeoTIFF: {exc}"
            ) from exc

    # Fallback for Windows Schannel / corporate proxy certificate errors.
    try:
        with _download_remote_asset(href) as local_path:
            with rasterio.open(local_path) as dataset:
                return _extract_clipped_data(
                    dataset,
                    asset,
                    aoi_geometry,
                )

    except Exception as fallback_exc:
        raise RuntimeError(
            "Direct remote GeoTIFF access failed because Windows "
            "Schannel did not trust the HTTPS certificate. GeoScope "
            "also tried the local-download fallback, but that failed. "
            f"Direct error: {direct_error}. "
            f"Fallback error: {fallback_exc}"
        ) from fallback_exc


def _to_geotiff_bytes(
    data: np.ndarray,
    profile: dict[str, Any],
    invalid_mask: np.ndarray,
    *,
    dtype: str,
    nodata: float,
    band_description: str,
) -> bytes:
    """
    Write a single-band GeoTIFF in memory.
    """
    output = data.astype(dtype, copy=True)

    combined_invalid = (
        invalid_mask
        | ~np.isfinite(output)
    )

    output[combined_invalid] = nodata

    output_profile = profile.copy()
    output_profile.update(
        {
            "dtype": dtype,
            "nodata": nodata,
            "count": 1,
            "driver": "GTiff",
            "compress": "deflate",
            "tiled": True,
        }
    )

    with MemoryFile() as memory_file:
        with memory_file.open(
            **output_profile
        ) as destination:
            destination.write(output, 1)
            destination.set_band_description(
                1,
                band_description,
            )

        return memory_file.read()


def generate_product_geotiff(
    *,
    scene: dict[str, Any],
    aoi_geometry: dict[str, Any],
    product: str,
) -> tuple[bytes, str, dict[str, Any]]:
    """
    Generate a clipped Red, NIR, or NDVI GeoTIFF.
    """
    if not scene:
        raise ValueError(
            "A STAC scene must be selected."
        )

    if not aoi_geometry:
        raise ValueError(
            "An AOI geometry is required."
        )

    normalized_product = product.strip().upper()

    if normalized_product not in {
        "RED",
        "NIR",
        "NDVI",
    }:
        raise ValueError(
            "Product must be Red, NIR, or NDVI."
        )

    item_id = scene.get(
        "item_id",
        "sentinel2",
    )
    item_date = scene.get(
        "date",
        "unknown-date",
    )

    if normalized_product == "RED":
        red_key, red_asset = find_band_asset(
            scene,
            "red",
        )

        data, profile, invalid = _read_clipped_band(
            red_asset,
            aoi_geometry,
        )

        description = (
            f"Sentinel-2 Red ({red_key})"
        )
        filename = (
            f"{item_id}_{item_date}_red_clip.tif"
        )

    elif normalized_product == "NIR":
        nir_key, nir_asset = find_band_asset(
            scene,
            "nir",
        )

        data, profile, invalid = _read_clipped_band(
            nir_asset,
            aoi_geometry,
        )

        description = (
            f"Sentinel-2 NIR ({nir_key})"
        )
        filename = (
            f"{item_id}_{item_date}_nir_clip.tif"
        )

    else:
        red_key, red_asset = find_band_asset(
            scene,
            "red",
        )
        nir_key, nir_asset = find_band_asset(
            scene,
            "nir",
        )

        red, red_profile, red_mask = (
            _read_clipped_band(
                red_asset,
                aoi_geometry,
            )
        )

        nir, nir_profile, nir_mask = (
            _read_clipped_band(
                nir_asset,
                aoi_geometry,
            )
        )

        if red.shape != nir.shape:
            raise ValueError(
                "The Red and NIR arrays do not have "
                "the same shape."
            )

        if (
            red_profile["crs"]
            != nir_profile["crs"]
        ):
            raise ValueError(
                "The Red and NIR bands use different "
                "coordinate systems."
            )

        if (
            red_profile["transform"]
            != nir_profile["transform"]
        ):
            raise ValueError(
                "The Red and NIR bands are not aligned "
                "on the same pixel grid."
            )

        denominator = nir + red

        invalid = (
            red_mask
            | nir_mask
            | ~np.isfinite(red)
            | ~np.isfinite(nir)
            | ~np.isfinite(denominator)
            | (
                np.abs(denominator)
                < 1e-10
            )
        )

        data = np.full(
            red.shape,
            np.nan,
            dtype="float32",
        )

        valid = ~invalid

        data[valid] = (
            (nir[valid] - red[valid])
            / denominator[valid]
        )

        data[valid] = np.clip(
            data[valid],
            -1.0,
            1.0,
        )

        profile = red_profile
        description = (
            "Normalized Difference Vegetation "
            f"Index using {nir_key} and {red_key}"
        )
        filename = (
            f"{item_id}_{item_date}_ndvi_clip.tif"
        )

    geotiff_bytes = _to_geotiff_bytes(
        data,
        profile,
        invalid,
        dtype="float32",
        nodata=-9999.0,
        band_description=description,
    )

    valid_values = data[
        (~invalid)
        & np.isfinite(data)
    ]

    summary = {
        "product": normalized_product,
        "scene": item_id,
        "date": item_date,
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "crs": str(profile["crs"]),
        "minimum": (
            float(valid_values.min())
            if valid_values.size
            else None
        ),
        "maximum": (
            float(valid_values.max())
            if valid_values.size
            else None
        ),
        "mean": (
            float(valid_values.mean())
            if valid_values.size
            else None
        ),
        "insecure_ssl_mode": (
            ALLOW_INSECURE_SSL
        ),
    }

    return (
        geotiff_bytes,
        filename,
        summary,
    )
