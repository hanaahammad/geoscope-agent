from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _get_ranker():
    """
    Load the FlashRank model once per Python process.
    """
    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise RuntimeError(
            "FlashRank is not installed. Run: "
            "python -m pip install flashrank"
        ) from exc

    return Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir="data/flashrank_cache",
    )


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
