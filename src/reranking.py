from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_CACHE_DIR = PROJECT_ROOT / "data" / "flashrank_cache"
FLASHRANK_MODEL_DIR = FLASHRANK_CACHE_DIR / FLASHRANK_MODEL_NAME

# These files are enough to identify the locally cached reranker used by GeoScope.
FLASHRANK_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "flashrank-MiniLM-L-12-v2_Q.onnx",
)


def flashrank_local_cache_available() -> bool:
    """
    Return True when the expected GeoScope FlashRank model files are
    already present in the local cache.

    This check is intentionally lightweight: it does not initialize the
    model and therefore does not trigger a download.
    """
    return all(
        (FLASHRANK_MODEL_DIR / file_name).exists()
        for file_name in FLASHRANK_REQUIRED_FILES
    )


@lru_cache(maxsize=1)
def _get_ranker():
    """
    Load the FlashRank model once per Python process.

    If the model is not cached, FlashRank may try to download it.
    Any resulting setup/network error is converted to a clear GeoScope
    RuntimeError instead of leaking a low-level traceback to the user.
    """
    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise RuntimeError(
            "FlashRank is not installed. Run: "
            "python -m pip install flashrank"
        ) from exc

    try:
        return Ranker(
            model_name=FLASHRANK_MODEL_NAME,
            cache_dir=str(FLASHRANK_CACHE_DIR),
        )
    except Exception as exc:
        raise RuntimeError(
            "GeoScope could not initialize the FlashRank reranker "
            f"`{FLASHRANK_MODEL_NAME}`. The model may be missing and "
            "the environment may not allow it to be downloaded. "
            "Use Vector search or Query rewriting + vector search, "
            "or install/cache the FlashRank model and retry. "
            f"Technical detail: {exc}"
        ) from exc


def check_flashrank_ready() -> tuple[bool, str]:
    """
    Verify that FlashRank can actually be initialized.

    When the model is not cached, this may allow FlashRank to perform its
    normal first-time download. The caller receives a status instead of
    an uncaught exception.
    """
    try:
        _get_ranker()
        return True, (
            f"FlashRank reranker `{FLASHRANK_MODEL_NAME}` is available."
        )
    except Exception as exc:
        return False, str(exc)


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Re-rank retrieved chunks with FlashRank and return the best top_k.
    """
    if not documents:
        return []

    try:
        from flashrank import RerankRequest
    except ImportError as exc:
        raise RuntimeError(
            "FlashRank is not installed. Run: "
            "python -m pip install flashrank"
        ) from exc

    passages = [
        {
            "id": str(index),
            "text": document.get("text", ""),
            "meta": {
                "original_index": index,
            },
        }
        for index, document in enumerate(documents)
    ]

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    ranked = _get_ranker().rerank(request)

    output: list[dict[str, Any]] = []

    for rank, item in enumerate(ranked[:top_k], start=1):
        original_index = int(
            item.get("meta", {}).get(
                "original_index",
                item.get("id", 0),
            )
        )
        row = dict(documents[original_index])
        row["rerank_score"] = float(item.get("score", 0.0))
        row["rerank_rank"] = rank
        output.append(row)

    return output
