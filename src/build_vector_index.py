from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "docs" / "processed" / "chunks.jsonl"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_store"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "geoscope_documents"
BATCH_SIZE = 20


def load_chunks() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def create_embeddings(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=300,
    )
    response.raise_for_status()
    embeddings = response.json().get("embeddings")

    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings.")
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count does not match input count.")

    return embeddings


def build_vector_index() -> dict[str, Any]:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(f"No chunks found in {CHUNKS_FILE}")

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "GeoScope remote-sensing knowledge base"},
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        documents = [item["text"] for item in batch]

        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            documents=documents,
            metadatas=[
                {
                    "document_id": item["document_id"],
                    "file_name": item["file_name"],
                    "page_number": int(item["page_number"]),
                    "chunk_number": int(item["chunk_number"]),
                    "source_type": item.get("source_type", "unknown"),
                    "title": item.get("title", item["file_name"]),
                }
                for item in batch
            ],
            embeddings=create_embeddings(documents),
        )

    return {
        "status": "success",
        "records": collection.count(),
        "path": str(VECTOR_DB_DIR),
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
    }


if __name__ == "__main__":
    print(json.dumps(build_vector_index(), indent=2))
