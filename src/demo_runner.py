from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable

from src.generation import generate_answer
from src.geocoding import search_places
from src.geotiff_processing import generate_product_geotiff
from src.retrieval import search_documents
from src.stac_search import search_sentinel2


ProgressCallback = Callable[[str, str], None]


@dataclass
class DemoConfig:
    place_query: str = "Kom Ombo, Aswan, Egypt"
    start_date: str = "2025-11-01"
    end_date: str = "2026-03-31"
    max_cloud_cover: int = 40
    scene_limit: int = 10
    retrieval_approach: str = "rewrite_rerank"
    top_k: int = 5
    candidate_k: int = 15
    product: str = "NDVI"
    generate_geotiff: bool = False
    use_text_geocoding: bool = True


def _report(
    callback: ProgressCallback | None,
    step: str,
    message: str,
) -> None:
    if callback:
        callback(step, message)


def _demo_fallback_aoi() -> dict[str, Any]:
    """
    Small demonstration bounding box around Kom Ombo.

    This is a fallback study-area box, not an official administrative
    boundary. Text geocoding remains the preferred mode.
    """
    west = 32.87
    south = 24.40
    east = 33.03
    north = 24.56

    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def _resolve_aoi(
    config: DemoConfig,
    callback: ProgressCallback | None,
) -> tuple[dict[str, Any], str, str]:
    _report(
        callback,
        "AOI",
        f"Resolving the demonstration area: {config.place_query}",
    )

    if config.use_text_geocoding:
        try:
            results = search_places(
                config.place_query,
                limit=5,
            )

            if results:
                preferred = results[0]
                return (
                    preferred["geometry"],
                    preferred["display_name"],
                    "Nominatim text search",
                )
        except Exception as exc:
            _report(
                callback,
                "AOI warning",
                "Text geocoding failed; using the bundled Kom Ombo "
                f"demonstration box. Detail: {exc}",
            )

    return (
        _demo_fallback_aoi(),
        "Kom Ombo demonstration AOI",
        "Bundled fallback bounding box",
    )


def _build_augmented_question(
    *,
    question: str,
    aoi_label: str,
    scenes: list[dict[str, Any]],
    unique_dates: list[str],
) -> str:
    scene_lines = [
        (
            f"- {scene.get('item_id')} | "
            f"date={scene.get('date')} | "
            f"cloud_cover={scene.get('cloud_cover')}"
        )
        for scene in scenes[:10]
    ]

    return f"""
User question:
{question}

Demonstration context:
- AOI: {aoi_label}
- Sentinel-2 scene items: {len(scenes)}
- Distinct acquisition dates: {len(unique_dates)}
- Dates: {", ".join(unique_dates) if unique_dates else "None"}
- Time-series analysis possible: {"yes" if len(unique_dates) >= 2 else "no"}

Available Sentinel-2 scene metadata:
{chr(10).join(scene_lines) if scene_lines else "No scenes found."}

Important:
Multiple scene items from one acquisition date must not be described as
a time series.
""".strip()


def run_automated_demo(
    *,
    question: str,
    config: DemoConfig,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Execute the GeoScope demonstration from AOI resolution to RAG answer
    and optional GeoTIFF generation.
    """
    if not question.strip():
        raise ValueError("The demonstration question cannot be empty.")

    start = date.fromisoformat(config.start_date)
    end = date.fromisoformat(config.end_date)

    if start > end:
        raise ValueError(
            "The demonstration start date must be before the end date."
        )

    output: dict[str, Any] = {
        "config": asdict(config),
        "question": question,
        "steps": [],
    }

    aoi, aoi_label, aoi_source = _resolve_aoi(
        config,
        callback,
    )
    output["aoi_geojson"] = aoi
    output["aoi_label"] = aoi_label
    output["aoi_source"] = aoi_source
    output["steps"].append(
        {
            "step": "AOI",
            "status": "completed",
            "detail": aoi_label,
        }
    )

    _report(
        callback,
        "STAC",
        "Searching Earth Search for Sentinel-2 scenes...",
    )

    scenes = search_sentinel2(
        aoi_geometry=aoi,
        start_date=config.start_date,
        end_date=config.end_date,
        max_cloud_cover=config.max_cloud_cover,
        limit=config.scene_limit,
    )

    unique_dates = sorted(
        {
            scene.get("date")
            for scene in scenes
            if scene.get("date")
        }
    )

    output["scenes"] = scenes
    output["scene_count"] = len(scenes)
    output["unique_dates"] = unique_dates
    output["distinct_date_count"] = len(unique_dates)
    output["time_series_available"] = (
        len(unique_dates) >= 2
    )
    output["steps"].append(
        {
            "step": "STAC",
            "status": "completed",
            "detail": (
                f"{len(scenes)} scene items, "
                f"{len(unique_dates)} distinct dates"
            ),
        }
    )

    if not scenes:
        raise RuntimeError(
            "No Sentinel-2 scenes were returned. Increase the cloud "
            "threshold or widen the date range."
        )

    augmented_question = _build_augmented_question(
        question=question,
        aoi_label=aoi_label,
        scenes=scenes,
        unique_dates=unique_dates,
    )
    output["augmented_question"] = augmented_question

    _report(
        callback,
        "Retrieval",
        "Rewriting the query, retrieving Chroma candidates, "
        "and applying FlashRank...",
    )

    sources = search_documents(
        augmented_question,
        top_k=config.top_k,
        approach=config.retrieval_approach,
        candidate_k=(
            config.candidate_k
            if config.retrieval_approach
            in {"rerank", "rewrite_rerank"}
            else None
        ),
    )

    output["sources"] = sources
    output["rewritten_query"] = (
        sources[0].get("retrieval_query")
        if sources
        else augmented_question
    )
    output["steps"].append(
        {
            "step": "Retrieval",
            "status": "completed",
            "detail": f"{len(sources)} final context chunks",
        }
    )

    _report(
        callback,
        "Generation",
        "Generating the grounded GeoAI recommendation...",
    )

    answer = generate_answer(
        question=augmented_question,
        retrieved_chunks=sources,
    )

    output["answer"] = answer
    output["steps"].append(
        {
            "step": "Generation",
            "status": "completed",
            "detail": "Grounded answer generated",
        }
    )

    best_scene = min(
        scenes,
        key=lambda item: (
            item.get("cloud_cover")
            if item.get("cloud_cover") is not None
            else 999
        ),
    )
    output["selected_scene"] = best_scene

    if config.generate_geotiff:
        _report(
            callback,
            "GeoTIFF",
            f"Generating {config.product} from the lowest-cloud scene...",
        )

        geotiff, filename, summary = generate_product_geotiff(
            scene=best_scene,
            aoi_geometry=aoi,
            product=config.product,
        )

        output["geotiff_bytes"] = geotiff
        output["geotiff_filename"] = filename
        output["geotiff_summary"] = summary
        output["steps"].append(
            {
                "step": "GeoTIFF",
                "status": "completed",
                "detail": filename,
            }
        )
    else:
        output["steps"].append(
            {
                "step": "GeoTIFF",
                "status": "skipped",
                "detail": (
                    "Disabled for this run; enable it for the full demo."
                ),
            }
        )

    _report(
        callback,
        "Completed",
        "The automated GeoScope demonstration finished successfully.",
    )

    return output
