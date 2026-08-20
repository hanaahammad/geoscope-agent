from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.generation import generate_answer
from src.llm_provider import generate_text, get_generation_model
from src.retrieval import search_documents
from src.stac_search import search_sentinel2


class AgentState(TypedDict, total=False):
    question: str
    aoi_geojson: dict[str, Any] | None
    aoi_summary: str
    start_date: str
    end_date: str
    max_cloud_cover: float
    scenes: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    answer: str
    top_k: int
    candidate_k: int
    trace: list[dict[str, Any]]
    next_action: str
    rationale: str
    iteration: int


ALLOWED_ACTIONS = {
    "inspect_context",
    "search_stac",
    "retrieve_knowledge",
    "final_answer",
}


def _add_trace(
    state: AgentState,
    *,
    step: str,
    detail: str,
    decision_by: str = "Agent",
) -> AgentState:
    updated = dict(state)
    trace = list(updated.get("trace", []))
    trace.append(
        {
            "step": step,
            "decision_by": decision_by,
            "detail": detail,
        }
    )
    updated["trace"] = trace
    return updated


def _fallback_action(state: AgentState) -> str:
    if not state.get("trace"):
        return "inspect_context"
    if state.get("aoi_geojson") and not state.get("scenes"):
        return "search_stac"
    if not state.get("sources"):
        return "retrieve_knowledge"
    return "final_answer"


def _planner(state: AgentState) -> AgentState:
    iteration = int(state.get("iteration", 0)) + 1
    if iteration > 5:
        updated = dict(state)
        updated["next_action"] = "final_answer"
        updated["rationale"] = "Maximum planning iterations reached."
        updated["iteration"] = iteration
        return updated

    available = {
        "has_aoi": bool(state.get("aoi_geojson")),
        "scene_count": len(state.get("scenes", [])),
        "source_count": len(state.get("sources", [])),
        "has_answer": bool(state.get("answer")),
    }

    instructions = """
You are the planning component of GeoScope. Decide the next useful action.
Return JSON only with keys: action and rationale.
Allowed actions: inspect_context, search_stac, retrieve_knowledge, final_answer.
Rules:
- inspect_context summarizes what geographic context is already available.
- search_stac is useful only when an AOI exists and no STAC scenes are available.
- retrieve_knowledge should be used before final_answer unless sources already exist.
- final_answer is appropriate once enough evidence is available.
- Do not invent tools or data.
""".strip()

    prompt = f"""
Question: {state.get('question', '')}
Available state: {json.dumps(available)}
AOI summary: {state.get('aoi_summary', '')}
Previous trace: {json.dumps(state.get('trace', [])[-4:])}
Choose the next action.
""".strip()

    action = ""
    rationale = ""
    try:
        raw = generate_text(
            instructions=instructions,
            prompt=prompt,
            model=get_generation_model(),
            json_output=True,
        )
        parsed = json.loads(raw)
        action = str(parsed.get("action", "")).strip()
        rationale = str(parsed.get("rationale", "")).strip()
    except Exception as exc:
        rationale = f"Planner fallback used: {exc}"

    if action not in ALLOWED_ACTIONS:
        action = _fallback_action(state)
        if not rationale:
            rationale = "Validated fallback selected from current state."

    # Guard impossible actions while keeping the LLM as planner.
    if action == "search_stac" and not state.get("aoi_geojson"):
        action = "retrieve_knowledge" if not state.get("sources") else "final_answer"
        rationale += " STAC search skipped because no AOI is available."
    if action == "final_answer" and not state.get("sources"):
        action = "retrieve_knowledge"
        rationale += " Knowledge retrieval is required before final answer."

    updated = dict(state)
    updated["next_action"] = action
    updated["rationale"] = rationale
    updated["iteration"] = iteration
    return _add_trace(
        updated,
        step=f"Planner → {action}",
        detail=rationale or "Next action selected from current evidence.",
        decision_by="LLM planner",
    )


def _inspect_context(state: AgentState) -> AgentState:
    dates = sorted({s.get("date") for s in state.get("scenes", []) if s.get("date")})
    detail = (
        f"AOI available={bool(state.get('aoi_geojson'))}; "
        f"scene items={len(state.get('scenes', []))}; "
        f"distinct dates={len(dates)}."
    )
    return _add_trace(state, step="Tool: inspect_context", detail=detail)


def _search_stac(state: AgentState) -> AgentState:
    updated = dict(state)
    scenes = search_sentinel2(
        aoi_geometry=state["aoi_geojson"],
        start_date=state.get("start_date") or "2025-11-01",
        end_date=state.get("end_date") or "2026-03-31",
        max_cloud_cover=float(state.get("max_cloud_cover", 40.0)),
        limit=10,
    )
    updated["scenes"] = scenes
    dates = sorted({s.get("date") for s in scenes if s.get("date")})
    return _add_trace(
        updated,
        step="Tool: search_stac",
        detail=f"Found {len(scenes)} Sentinel-2 item(s) across {len(dates)} distinct date(s).",
    )


def _retrieve(state: AgentState) -> AgentState:
    updated = dict(state)
    context = (
        f"Question: {state.get('question', '')}\n"
        f"AOI: {state.get('aoi_summary', '')}\n"
        f"STAC scene count: {len(state.get('scenes', []))}"
    )
    sources = search_documents(
        context,
        top_k=int(state.get("top_k", 5)),
        approach="rewrite_rerank",
        candidate_k=int(state.get("candidate_k", 15)),
    )
    updated["sources"] = sources
    return _add_trace(
        updated,
        step="Tool: retrieve_knowledge",
        detail=f"Retrieved and reranked {len(sources)} source chunk(s).",
    )


def _final_answer(state: AgentState) -> AgentState:
    updated = dict(state)
    dates = sorted({s.get("date") for s in state.get("scenes", []) if s.get("date")})
    question = f"""
{state.get('question', '')}

CURRENT GEOSPATIAL STATE
AOI: {state.get('aoi_summary', 'No AOI')}
STAC scene items: {len(state.get('scenes', []))}
Distinct acquisition dates: {len(dates)}
Dates: {', '.join(dates) if dates else 'None'}
Time-series possible: {'yes' if len(dates) >= 2 else 'no'}
""".strip()
    updated["answer"] = generate_answer(
        question=question,
        retrieved_chunks=state.get("sources", []),
    )
    updated["next_action"] = "done"
    return _add_trace(
        updated,
        step="Tool: final_answer",
        detail="Generated a grounded answer from the evidence gathered by the agent.",
    )


def _route_after_planner(state: AgentState) -> Literal[
    "inspect_context", "search_stac", "retrieve_knowledge", "final_answer"
]:
    return state.get("next_action", "retrieve_knowledge")  # type: ignore[return-value]


def _back_to_planner(state: AgentState) -> str:
    if state.get("next_action") == "done" or state.get("answer"):
        return "end"
    return "planner"


def build_agentic_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("inspect_context", _inspect_context)
    graph.add_node("search_stac", _search_stac)
    graph.add_node("retrieve_knowledge", _retrieve)
    graph.add_node("final_answer", _final_answer)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "inspect_context": "inspect_context",
            "search_stac": "search_stac",
            "retrieve_knowledge": "retrieve_knowledge",
            "final_answer": "final_answer",
        },
    )
    for node in ("inspect_context", "search_stac", "retrieve_knowledge"):
        graph.add_edge(node, "planner")
    graph.add_edge("final_answer", END)
    return graph.compile()


def run_agentic_workflow(
    question: str,
    *,
    aoi_geojson: dict[str, Any] | None = None,
    aoi_summary: str = "",
    scenes: list[dict[str, Any]] | None = None,
    start_date: str = "",
    end_date: str = "",
    max_cloud_cover: float = 40.0,
    top_k: int = 5,
    candidate_k: int = 15,
) -> AgentState:
    if not question.strip():
        raise ValueError("A question is required.")
    initial: AgentState = {
        "question": question.strip(),
        "aoi_geojson": aoi_geojson,
        "aoi_summary": aoi_summary,
        "scenes": list(scenes or []),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "max_cloud_cover": float(max_cloud_cover),
        "top_k": int(top_k),
        "candidate_k": int(candidate_k),
        "trace": [],
        "iteration": 0,
    }
    return build_agentic_graph().invoke(initial)
