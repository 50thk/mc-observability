import json
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class ServerErrorAnalysisResult(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    probable_cause: str
    evidence: dict
    mitigation: list[str]
    limitations: list[str]


class ServerErrorAnalysisState(TypedDict, total=False):
    mode: Literal["auto", "manual"]
    analysis_id: int | None
    session_id: str
    trace_id: str | None
    time_range: dict
    user_message: str | None
    record_detail: dict
    agent_context: dict
    analysis_result: dict | None
    quality_status: Literal["SUCCEEDED", "PARTIAL"] | None
    skip_analysis: bool
    error_message: str | None


def create_server_error_analysis_graph(repo, agent, config_manager):
    graph = StateGraph(ServerErrorAnalysisState)

    async def prepare_analysis(state: ServerErrorAnalysisState):
        analysis_config = _get_analysis_config(config_manager)
        detail = _normalize_record_detail(state)
        record_result = _load_or_create_record(repo, state, detail)
        if record_result.get("skip_analysis") and record_result.get("error_message"):
            return record_result

        prepared_state = {
            **state,
            "analysis_id": record_result["analysis_id"],
            "record_detail": record_result["record_detail"],
        }
        return {
            **record_result,
            "agent_context": _build_agent_context(prepared_state, analysis_config),
        }

    async def investigate_with_agent(state: ServerErrorAnalysisState, config: RunnableConfig):
        if state.get("skip_analysis"):
            return {}

        allow_succeeded = state["mode"] == "manual"
        if not repo.mark_running(state["analysis_id"], allow_succeeded=allow_succeeded):
            return {"skip_analysis": True}

        payload = {"messages": [{"role": "user", "content": state["agent_context"]["message"]}]}
        agent_config = dict(config or {})
        recursion_limit = state["agent_context"].get("recursion_limit")
        if recursion_limit is not None:
            agent_config["recursion_limit"] = recursion_limit

        try:
            result = await agent.ainvoke(payload, config=agent_config)
        except Exception as exc:
            return {"analysis_result": None, "error_message": str(exc)}

        structured = _extract_structured_response(result)
        if structured is None:
            return {
                "analysis_result": None,
                "error_message": "Agent returned no structured result",
            }

        return {
            "analysis_result": structured,
            "quality_status": _classify_result_quality(structured, _get_analysis_config(config_manager)),
            "error_message": None,
        }

    async def finalize_result(state: ServerErrorAnalysisState):
        if state.get("skip_analysis"):
            return {}

        error_message = state.get("error_message")
        result = state.get("analysis_result")
        if error_message or not result:
            repo.save_failed(
                state["analysis_id"],
                error_message or "Agent returned no structured result",
                state.get("record_detail"),
            )
            return {}

        detail = dict(state.get("record_detail") or {})
        detail.update(result)
        detail["quality_status"] = state.get("quality_status") or "SUCCEEDED"
        summary = result.get("summary", "")
        if detail["quality_status"] == "PARTIAL" and hasattr(repo, "save_partial"):
            repo.save_partial(state["analysis_id"], summary=summary, detail=detail)
        else:
            repo.save_success(state["analysis_id"], summary=summary, detail=detail)
        return {"record_detail": detail}

    graph.add_node("prepare_analysis", prepare_analysis)
    graph.add_node("investigate_with_agent", investigate_with_agent)
    graph.add_node("finalize_result", finalize_result)

    graph.add_edge(START, "prepare_analysis")
    graph.add_edge("prepare_analysis", "investigate_with_agent")
    graph.add_edge("investigate_with_agent", "finalize_result")
    graph.add_edge("finalize_result", END)

    return graph.compile()


def _normalize_record_detail(state: ServerErrorAnalysisState) -> dict:
    detail = dict(state.get("record_detail") or {})
    detail.setdefault("dedup_basis", "trace_id")
    detail["analysis_mode"] = state["mode"]
    detail["trace_id"] = state.get("trace_id")
    detail["time_range"] = state.get("time_range") or {}

    if state.get("trace_id") is None:
        detail.setdefault("no_trace_context", {"time_range": state.get("time_range") or {}})

    return detail


def _load_or_create_record(repo, state: ServerErrorAnalysisState, detail: dict) -> dict:
    if state.get("analysis_id"):
        record = repo.get_by_id(state["analysis_id"])
        created = False
        if record is None:
            return {
                "skip_analysis": True,
                "error_message": f"Analysis record not found: {state['analysis_id']}",
            }
    else:
        record, created = repo.load_or_create(
            trace_id=state.get("trace_id"),
            session_id=state["session_id"],
            detail=detail,
        )

    skip = bool(state["mode"] == "auto" and record.STATUS == "SUCCEEDED" and not created)
    return {
        "analysis_id": record.ID,
        "record_detail": record.DETAIL_JSON or detail,
        "skip_analysis": skip,
    }


def _build_agent_context(state: ServerErrorAnalysisState, analysis_config: dict[str, Any]) -> dict:
    mode = state["mode"]
    recursion_limit = analysis_config.get("auto_recursion_limit" if mode == "auto" else "manual_recursion_limit")
    loki_query_limit = analysis_config.get("loki_query_limit")
    max_evidence_tokens = analysis_config.get("max_evidence_tokens")

    return {
        "message": _build_investigation_brief(
            state=state,
            loki_query_limit=loki_query_limit,
            max_evidence_tokens=max_evidence_tokens,
        ),
        "mode": mode,
        "trace_id": state.get("trace_id"),
        "time_range": state.get("time_range") or {},
        "recursion_limit": recursion_limit,
        "loki_query_limit": loki_query_limit,
        "max_evidence_tokens": max_evidence_tokens,
    }


def _build_investigation_brief(
    state: ServerErrorAnalysisState,
    loki_query_limit: int | None,
    max_evidence_tokens: int | None,
) -> str:
    user_message = state.get("user_message")
    detail = state.get("record_detail") or {}
    trace_id = state.get("trace_id")
    time_range = state.get("time_range") or {}
    baseline_evidence = _format_baseline_evidence(detail, max_evidence_tokens)

    if user_message:
        request = user_message
    else:
        request = "Analyze the HTTP 5xx server error and identify the most probable cause."

    return "\n".join(
        [
            "HTTP 5xx server error investigation request",
            "",
            f"Mode: {state['mode']}",
            f"Trace ID: {trace_id or 'not provided'}",
            f"Time range: {time_range or 'not provided'}",
            f"Loki query limit: {loki_query_limit or 'not configured'}",
            f"Max evidence tokens: {max_evidence_tokens or 'not configured'}",
            "",
            "Baseline evidence from the detection pipeline:",
            baseline_evidence,
            "",
            "Investigation instructions:",
            "1. Prefer trace-specific evidence when a trace ID is available.",
            "2. Use logs, traces, and metrics tools only when they can strengthen or reject a concrete hypothesis.",
            "3. Ground probable_cause, risk_level, confidence, mitigation, and limitations in observed evidence.",
            "4. If evidence is missing or weak, say so in limitations instead of overstating certainty.",
            "5. Return the schema-valid structured response requested by the system prompt.",
            "",
            "User request:",
            request,
        ]
    )


def _format_baseline_evidence(detail: dict, max_evidence_tokens: int | None) -> str:
    evidence = json.dumps(detail, ensure_ascii=False, default=str, sort_keys=True, indent=2)
    if not max_evidence_tokens:
        return evidence

    max_chars = max(max_evidence_tokens * 4, 200)
    if len(evidence) <= max_chars:
        return evidence
    return f"{evidence[:max_chars]}\n... [truncated to max_evidence_tokens budget]"


def _get_analysis_config(config_manager) -> dict[str, Any]:
    if config_manager and hasattr(config_manager, "get_server_error_analysis_config"):
        return config_manager.get_server_error_analysis_config()
    return {}


def _extract_structured_response(result):
    if isinstance(result, BaseModel):
        return result.model_dump()
    if not isinstance(result, dict):
        return None

    structured = result.get("structured_response")
    if isinstance(structured, BaseModel):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    return None


def _classify_result_quality(result: dict, analysis_config: dict[str, Any]) -> Literal["SUCCEEDED", "PARTIAL"]:
    evidence = result.get("evidence")
    confidence = result.get("confidence")
    threshold = analysis_config.get("partial_confidence_threshold", 0.4)

    if not evidence:
        return "PARTIAL"
    if isinstance(confidence, int | float) and confidence < threshold:
        return "PARTIAL"
    return "SUCCEEDED"
