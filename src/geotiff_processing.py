from __future__ import annotations

import os
import tempfile
from typing import Any

import requests

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import transform_geom

from src.stac_search import find_band_asset


# ---------------------------------------------------------------------------
# HTTPS / certificate configuration
# ---------------------------------------------------------------------------
# Temporary demo setting for networks that insert a self-signed certificate.
#
# Keep "YES" only while testing on the current corporate network.
# For a production deployment, set this environment variable to "NO" and
# configure a trusted CA bundle instead:
#
#   set GEOSCOPE_ALLOW_INSECURE_SSL=NO
#   set GDAL_CURL_CA_BUNDLE=C:\certificates\company-root-ca.pem
#   set CURL_CA_BUNDLE=C:\certificates\company-root-ca.pem
#   set SSL_CERT_FILE=C:\certificates\company-root-ca.pem
#
ALLOW_INSECURE_SSL = (
    os.getenv("GEOSCOPE_ALLOW_INSECURE_SSL", "YES").strip().upper()
    in {"1", "TRUE", "YES", "ON"}
)


def _build_gdal_environment() -> dict[str, Any]:
    """
    Build the GDAL/Rasterio configuration used to read remote COG assets.

    Preference order:
    1. Use a trusted CA bundle when one is configured.
    2. Otherwise, allow the temporary insecure demo mode when enabled.
    """
    options: dict[str, Any] = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_HTTP_TIMEOUT": "60",
        "GDAL_HTTP_CONNECTTIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2",

        # Earth Search may return public Landsat assets as s3:// URIs.
        # These are public-read objects, so Rasterio/GDAL must access them
        # anonymously instead of looking for ~/.aws/credentials.
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    ca_bundle = (
        os.getenv("GDAL_CURL_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )

    if ca_bundle:
        options["GDAL_CURL_CA_BUNDLE"] = ca_bundle
        options["CURL_CA_BUNDLE"] = ca_bundle
        options["SSL_CERT_FILE"] = ca_bundle

    elif ALLOW_INSECURE_SSL:
        # Temporary workaround only:
        # GDAL/libcurl will skip TLS certificate verification.
        options["GDAL_HTTP_UNSAFESSL"] = "YES"

    return options


def _scale_and_offset(
    asset: dict[str, Any],
) -> tuple[float, float]:
    """
    Read scale and offset from STAC raster metadata.

    Sentinel-2 assets may expose physical scaling information in
    raster:bands. If absent, the raw values are used unchanged.
    """
    raster_bands = asset.get("raster_bands") or []

    if not raster_bands:
        return 1.0, 0.0

    metadata = raster_bands[0] or {}

    return (
        float(metadata.get("scale", 1.0)),
        float(metadata.get("offset", 0.0)),
    )



def _read_clipped_band_from_local_file(
    local_path: str,
    aoi_geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """
    Open a local GeoTIFF and clip it to the AOI.

    Used as a fallback when GDAL/libcurl cannot trust the corporate HTTPS
    certificate chain even though GeoScope insecure demo mode is enabled.
    """
    with rasterio.open(local_path) as dataset:
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


def _download_https_asset_insecurely(
    href: str,
) -> str:
    """
    Download a public HTTPS GeoTIFF with TLS verification disabled.

    This is a DEMO fallback only for corporate networks that inject an
    untrusted root certificate into HTTPS traffic.

    The returned path points to a temporary .tif file. The caller is
    responsible for deleting it.
    """
    response = requests.get(
        href,
        stream=True,
        timeout=(30, 300),
        verify=False,
        headers={"User-Agent": "GeoScope/1.0"},
    )
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".tif",
        delete=False,
    )

    try:
        with tmp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
    except Exception:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        raise

    return tmp.name


def _read_clipped_band(
    asset: dict[str, Any],
    aoi_geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """
    Read one remote Cloud-Optimized GeoTIFF band and clip it to the AOI.
    """
    href = asset.get("href")

    if not href:
        raise ValueError("The selected STAC asset has no URL.")

    # Sentinel-2 assets are commonly HTTPS, while Landsat Earth Search
    # assets may be returned as public s3:// URIs.  Convert those to HTTPS
    # so GDAL uses normal HTTP range requests and never needs AWS credentials.
    if str(href).startswith("s3://"):
        s3_path = str(href)[5:]
        if "/" in s3_path:
            bucket, key = s3_path.split("/", 1)
            href = f"https://{bucket}.s3.amazonaws.com/{key}"

    environment = _build_gdal_environment()

    try:
        with rasterio.Env(**environment):
            with rasterio.open(href) as dataset:
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

    except rasterio.errors.RasterioIOError as exc:
        message = str(exc)

        if (
            "CERT_TRUST_IS_UNTRUSTED_ROOT" in message
            or "certificate verify failed" in message.lower()
            or "ssl" in message.lower()
        ):
            if ALLOW_INSECURE_SSL and str(href).startswith("https://"):
                local_path = None
                try:
                    # Windows GDAL/libcurl can still reject an intercepted TLS
                    # certificate even when GDAL_HTTP_UNSAFESSL is enabled.
                    # For the local demo only, fall back to requests with
                    # certificate verification disabled, then let Rasterio
                    # read the downloaded file locally.
                    local_path = _download_https_asset_insecurely(str(href))

                    data, profile, invalid_mask = (
                        _read_clipped_band_from_local_file(
                            local_path,
                            aoi_geometry,
                        )
                    )

                    scale, offset = _scale_and_offset(asset)
                    data = data * scale + offset

                    return data, profile, invalid_mask

                except Exception as fallback_exc:
                    raise RuntimeError(
                        "The remote GeoTIFF could not be opened through GDAL "
                        "because the corporate HTTPS certificate is not trusted. "
                        "GeoScope also attempted its local-demo HTTPS fallback, "
                        f"but that failed: {fallback_exc}. "
                        f"Original GDAL error: {message}"
                    ) from fallback_exc

                finally:
                    if local_path:
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass

            raise RuntimeError(
                "The remote GeoTIFF could not be opened because the HTTPS "
                "certificate is not trusted. GeoScope temporary insecure SSL "
                f"mode is {'enabled' if ALLOW_INSECURE_SSL else 'disabled'}. "
                "For production, configure the organization's trusted root CA. "
                f"Original error: {message}"
            ) from exc

        raise RuntimeError(
            f"Rasterio could not open the selected GeoTIFF: {message}"
        ) from exc


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
    Write a single-band GeoTIFF to memory and return its bytes.
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
        with memory_file.open(**output_profile) as destination:
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
    Generate a clipped Red, NIR, or NDVI GeoTIFF in memory.

    Parameters
    ----------
    scene:
        Scene dictionary returned by search_sentinel2().
    aoi_geometry:
        GeoJSON Polygon or MultiPolygon in EPSG:4326.
    product:
        Red, NIR, or NDVI.

    Returns
    -------
    tuple
        GeoTIFF bytes, output filename, and summary metadata.
    """
    if not scene:
        raise ValueError("A STAC scene must be selected.")

    if not aoi_geometry:
        raise ValueError("An AOI geometry is required.")

    normalized_product = product.strip().upper()

    if normalized_product not in {"RED", "NIR", "NDVI"}:
        raise ValueError(
            "Product must be Red, NIR, or NDVI."
        )

    item_id = scene.get("item_id", "sentinel2")
    item_date = scene.get("date", "unknown-date")

    if normalized_product == "RED":
        red_key, red_asset = find_band_asset(
            scene,
            "red",
        )

        data, profile, invalid = _read_clipped_band(
            red_asset,
            aoi_geometry,
        )

        description = f"Sentinel-2 Red ({red_key})"
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

        description = f"Sentinel-2 NIR ({nir_key})"
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

        red, red_profile, red_mask = _read_clipped_band(
            red_asset,
            aoi_geometry,
        )

        nir, nir_profile, nir_mask = _read_clipped_band(
            nir_asset,
            aoi_geometry,
        )

        if red.shape != nir.shape:
            raise ValueError(
                "The Red and NIR arrays do not have the same shape. "
                "This scene requires reprojection or resampling."
            )

        if red_profile["crs"] != nir_profile["crs"]:
            raise ValueError(
                "The Red and NIR bands use different coordinate systems."
            )

        if red_profile["transform"] != nir_profile["transform"]:
            raise ValueError(
                "The Red and NIR bands are not aligned on the same pixel grid."
            )

        denominator = nir + red

        invalid = (
            red_mask
            | nir_mask
            | ~np.isfinite(red)
            | ~np.isfinite(nir)
            | ~np.isfinite(denominator)
            | (np.abs(denominator) < 1e-10)
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

        # NDVI should normally remain between -1 and 1.
        data[valid] = np.clip(
            data[valid],
            -1.0,
            1.0,
        )

        profile = red_profile
        description = (
            "Normalized Difference Vegetation Index "
            f"using {nir_key} and {red_key}"
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
        (~invalid) & np.isfinite(data)
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
        "insecure_ssl_mode": ALLOW_INSECURE_SSL,
    }

    return geotiff_bytes, filename, summary
