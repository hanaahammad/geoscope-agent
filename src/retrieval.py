from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import chromadb
import requests

from src.llm_provider import get_ollama_base_url, get_setting
from src.query_rewrite import rewrite_query
from src.reranking import rerank_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_store"

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "geoscope_documents"

RetrievalApproach = Literal[
    "vector",
    "rewrite",
    "rerank",
    "rewrite_rerank",
]

APPROACH_LABELS = {
    "vector": "Vector search",
    "rewrite": "Query rewriting + vector search",
    "rerank": "Vector search + FlashRank reranking",
    "rewrite_rerank": (
        "Query rewriting + vector search + FlashRank reranking"
    ),
}


def embed_query(query: str) -> list[float]:
    response = requests.post(
        f"{get_ollama_base_url()}/api/embed",
        json={
            "model": get_setting(
                "OLLAMA_EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            ),
            "input": query,
        },
        timeout=120,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not embeddings:
        raise RuntimeError(
            "Ollama returned no query embedding."
        )

    return embeddings[0]


def get_collection():
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )
    return client.get_collection(COLLECTION_NAME)


def _vector_search(
    query: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    collection = get_collection()

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    rows: list[dict[str, Any]] = []

    for rank, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        rows.append(
            {
                "text": document,
                "distance": float(distance),
                "vector_rank": rank,
                **metadata,
            }
        )

    return rows


def search_documents(
    query: str,
    top_k: int = 5,
    *,
    approach: RetrievalApproach = "vector",
    candidate_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search the GeoScope knowledge base using one of four approaches.

    vector
        Original query -> Chroma vector search.
    rewrite
        LLM-rewritten query -> Chroma vector search.
    rerank
        Original query -> wider vector candidate set -> FlashRank.
    rewrite_rerank
        Rewritten query -> wider vector candidate set -> FlashRank.
    """
    if approach not in APPROACH_LABELS:
        raise ValueError(
            f"Unknown retrieval approach: {approach}"
        )

    original_query = " ".join(query.split())

    if not original_query:
        raise ValueError(
            "A non-empty retrieval query is required."
        )

    use_rewrite = approach in {
        "rewrite",
        "rewrite_rerank",
    }
    use_rerank = approach in {
        "rerank",
        "rewrite_rerank",
    }

    retrieval_query = (
        rewrite_query(original_query)
        if use_rewrite
        else original_query
    )

    requested_candidates = (
        candidate_k
        if candidate_k is not None
        else max(top_k * 3, 10)
    )

    vector_top_k = (
        requested_candidates
        if use_rerank
        else top_k
    )

    rows = _vector_search(
        retrieval_query,
        top_k=vector_top_k,
    )

    if use_rerank:
        rows = rerank_documents(
            retrieval_query,
            rows,
            top_k=top_k,
        )
    else:
        rows = rows[:top_k]

    for final_rank, row in enumerate(
        rows,
        start=1,
    ):
        row["rank"] = final_rank
        row["original_query"] = original_query
        row["retrieval_query"] = retrieval_query
        row["retrieval_approach"] = approach
        row["retrieval_approach_label"] = (
            APPROACH_LABELS[approach]
        )

    return rows
