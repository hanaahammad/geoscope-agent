# GeoScope AOI and GeoTIFF extension

This bundle adds:

- AOI search by text through OpenStreetMap Nominatim
- Place-result validation before use
- Map preview of the place geometry
- STAC scene search by AOI and date range
- Distinct acquisition-date count
- One-scene Red, NIR, or NDVI processing
- AOI clipping
- In-memory GeoTIFF generation
- Streamlit download button

## Replace

- `pages/2_AOI_and_STAC.py`
- `src/stac_search.py`

## Add

- `src/geocoding.py`
- `src/geotiff_processing.py`

## Install

```powershell
python -m pip install rasterio numpy shapely requests pystac-client
```

Then restart:

```powershell
python -m streamlit run GeoScope.py
```

## Test

1. Search `Kom Ombo, Aswan, Egypt`.
2. Select the correct result.
3. Click `Use this place as the AOI`.
4. Choose a valid historical period.
5. Search Sentinel-2.
6. Select one scene.
7. Choose NDVI.
8. Generate and download the GeoTIFF.

## Scope

This MVP processes one STAC item at a time. It does not yet mosaic
multiple tiles or build a multi-date data cube.
