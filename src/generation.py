from __future__ import annotations

from typing import Any

from src.llm_provider import (
    generate_text,
    get_generation_model,
    get_provider,
)


INSTRUCTIONS = """
You are GeoScope, a remote-sensing and GeoAI assistant.
Use only the supplied retrieved context and AOI/STAC information.
Give a clear practical answer, explain recommended datasets and workflow,
mention limitations, cite source filenames when possible, and do not claim
that full satellite imagery was downloaded or processed.

Important temporal rule:
- The number of STAC scene items is not the number of acquisition dates.
- Do not recommend time-series analysis unless at least two distinct
  acquisition dates are explicitly available.
- Multiple tiles or scene items from one date are a single-date
  observation, not a temporal series.
""".strip()


def build_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    sections = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        sections.append(
            f"[Source {index}]\n"
            f"File: {chunk.get('file_name', 'Unknown')}\n"
            f"Page: {chunk.get('page_number', 'Unknown')}\n"
            f"Vector rank: {chunk.get('vector_rank', 'Unknown')}\n"
            f"Final rank: {chunk.get('rank', index)}\n"
            f"Rerank score: {chunk.get('rerank_score', 'N/A')}\n"
            f"Content:\n{chunk.get('text', '')}"
        )

    return "\n\n".join(sections)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n"
        f"{build_context(retrieved_chunks)}"
    )

    return generate_text(
        instructions=INSTRUCTIONS,
        prompt=prompt,
        model=get_generation_model(),
    )


def active_generation_configuration() -> dict[str, str]:
    return {
        "provider": get_provider(),
        "model": get_generation_model(),
    }
