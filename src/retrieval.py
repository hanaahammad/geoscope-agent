from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_store"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "geoscope_documents"


def embed_query(query: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": query},
        timeout=120,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no query embedding.")

    return embeddings[0]


def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_collection(COLLECTION_NAME)


def search_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    collection = get_collection()

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    rows: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        rows.append(
            {
                "text": document,
                "distance": float(distance),
                **metadata,
            }
        )

    return rows
