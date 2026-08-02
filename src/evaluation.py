from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm_provider import generate_text, get_judge_model, get_provider
from src.retrieval import search_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "evaluation_questions.csv"


def load_ground_truth() -> pd.DataFrame:
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(f"Ground-truth file was not found: {GROUND_TRUTH_FILE}")
    data = pd.read_csv(GROUND_TRUTH_FILE)
    required = {"question_id", "question", "expected_document", "domain", "difficulty"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing ground-truth columns: {sorted(missing)}")
    return data


def normalize_file_name(value: str) -> str:
    return str(value).strip().lower().replace("\\", "/")


def find_expected_rank(results: list[dict[str, Any]], expected_document: str) -> int | None:
    expected = normalize_file_name(expected_document)
    for rank, result in enumerate(results, start=1):
        retrieved = normalize_file_name(result.get("file_name", ""))
        if retrieved == expected or expected in retrieved or retrieved in expected:
            return rank
    return None


def evaluate_retrieval(ground_truth: pd.DataFrame, top_k: int = 5):
    rows = []
    for record in ground_truth.to_dict(orient="records"):
        results = search_documents(record["question"], top_k=top_k)
        rank = find_expected_rank(results, record["expected_document"])
        hit = rank is not None
        rows.append({
            "question_id": record["question_id"],
            "question": record["question"],
            "domain": record["domain"],
            "difficulty": record["difficulty"],
            "expected_document": record["expected_document"],
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": 1 / rank if rank else 0.0,
            "retrieved_documents": " | ".join(item.get("file_name", "") for item in results),
        })
    details = pd.DataFrame(rows)
    total = len(details)
    hits = int(details["hit"].sum()) if total else 0
    return {
        "questions_evaluated": total,
        "hits": hits,
        "failures": total - hits,
        "hit_rate": float(details["hit"].mean()) if total else 0.0,
        "mrr": float(details["reciprocal_rank"].mean()) if total else 0.0,
    }, details


def evaluate_generation(*, question: str, answer: str, retrieved_chunks: list[dict[str, Any]], judge_model: str | None = None, aoi_supplied: bool = False) -> dict[str, Any]:
    context = "\n\n".join(
        f"[Source {i}] {chunk.get('file_name')} (page {chunk.get('page_number')})\n{chunk.get('text', '')}"
        for i, chunk in enumerate(retrieved_chunks[:8], start=1)
    )
    instructions = (
        "You are an independent evaluator of a GeoAI RAG application. "
        "Evaluate only against the question and retrieved context. Return valid JSON only."
    )
    prompt = f'''User question:\n{question}\n\nAOI supplied: {"yes" if aoi_supplied else "no"}\n\nRetrieved context:\n{context}\n\nGenerated answer:\n{answer}\n\nReturn JSON with scores 1-5:\n{{"relevance":1,"groundedness":1,"completeness":1,"technical_correctness":1,"citation_quality":1,"geographic_relevance":1,"overall":1.0,"comment":"Brief explanation."}}'''
    raw = generate_text(
        instructions=instructions,
        prompt=prompt,
        model=judge_model or get_judge_model(),
        json_output=True,
    ).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


def active_judge_configuration() -> dict[str, str]:
    return {"provider": get_provider(), "model": get_judge_model()}
