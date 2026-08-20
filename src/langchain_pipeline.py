from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from src.generation import generate_answer
from src.query_rewrite import rewrite_query
from src.retrieval import search_documents


def _trace(state: dict[str, Any], step: str, detail: str) -> dict[str, Any]:
    state = dict(state)
    state.setdefault("trace", []).append(
        {"step": step, "decision_by": "Fixed pipeline", "detail": detail}
    )
    return state


def _rewrite(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    rewritten = rewrite_query(state["question"])
    state["retrieval_query"] = rewritten
    return _trace(state, "Rewrite query", rewritten)


def _retrieve(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    sources = search_documents(
        state["retrieval_query"],
        top_k=state["top_k"],
        approach="rerank",
        candidate_k=state["candidate_k"],
    )
    state["sources"] = sources
    return _trace(
        state,
        "Retrieve + rerank",
        f"Selected {len(sources)} context chunks.",
    )


def _generate(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    question = state["question"]
    geo_context = state.get("geo_context", "")
    if geo_context:
        question = f"{question}\n\nCURRENT GEOGRAPHIC CONTEXT:\n{geo_context}"

    state["answer"] = generate_answer(
        question=question,
        retrieved_chunks=state["sources"],
    )
    return _trace(state, "Generate answer", "Grounded answer generated.")


def build_standard_chain():
    """A deterministic LangChain Runnable pipeline with a fixed sequence."""
    return RunnableLambda(_rewrite) | RunnableLambda(_retrieve) | RunnableLambda(_generate)


def run_standard_pipeline(
    question: str,
    *,
    geo_context: str = "",
    top_k: int = 5,
    candidate_k: int = 15,
) -> dict[str, Any]:
    initial = {
        "question": question.strip(),
        "geo_context": geo_context,
        "top_k": int(top_k),
        "candidate_k": int(candidate_k),
        "trace": [],
    }
    if not initial["question"]:
        raise ValueError("A question is required.")
    return build_standard_chain().invoke(initial)
