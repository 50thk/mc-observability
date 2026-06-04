import json
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class ServerErrorEvidenceItem(BaseModel):
    source: Literal["trace", "log", "metric", "baseline"]
    signal: str
    observation: str
    supports_cause: bool


class ServerErrorHypothesis(BaseModel):
    cause: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ServerErrorAnalysisResult(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    probable_cause: str
    evidence: list[ServerErrorEvidenceItem] | dict[str, Any] = Field(default_factory=list)
    mitigation: list[str]
    limitations: list[str]
    affected_service: str | None = None
    affected_endpoint: str | None = None
    hypotheses: list[ServerErrorHypothesis] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)


class ServerErrorAnalysisState(TypedDict, total=False):
    mode: Literal["auto", "manual"]
    analysis_id: int | None
    session_id: str
    trace_id: str | None
    time_range: dict
    user_message: str | None
    record_detail: dict
    incident_context: dict
    agent_context: dict
    analysis_result: dict | None
    quality_status: Literal["SUCCEEDED", "PARTIAL"] | None
    quality_findings: dict
    skip_analysis: bool
    error_message: str | None


def create_server_error_analysis_graph(repo, agent, config_manager):
    graph = StateGraph(ServerErrorAnalysisState)

    async def prepare_analysis(state: ServerErrorAnalysisState):
        detail = _normalize_record_detail(state)
        record_result = _load_or_create_record(repo, state, detail)
        if record_result.get("skip_analysis") and record_result.get("error_message"):
            return record_result

        return record_result

    async def build_incident_context(state: ServerErrorAnalysisState):
        if state.get("skip_analysis"):
            return {}

        analysis_config = _get_analysis_config(config_manager)
        return {"incident_context": _build_incident_context(state, analysis_config)}

    async def supervisor_route(state: ServerErrorAnalysisState):
        if state.get("skip_analysis"):
            return {}

        analysis_config = _get_analysis_config(config_manager)
        incident_context = dict(state.get("incident_context") or {})
        incident_context["supervisor"] = _build_supervisor_decision(
            incident_context=incident_context,
            mode=state["mode"],
        )
        return {
            "incident_context": incident_context,
            "agent_context": _build_agent_context(
                state={**state, "incident_context": incident_context},
                analysis_config=analysis_config,
            ),
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
            "error_message": None,
        }

    async def validate_quality(state: ServerErrorAnalysisState):
        if state.get("skip_analysis") or state.get("error_message") or not state.get("analysis_result"):
            return {}

        quality_status, quality_findings = _evaluate_result_quality(
            state["analysis_result"],
            _get_analysis_config(config_manager),
        )
        return {
            "quality_status": quality_status,
            "quality_findings": quality_findings,
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
        detail["incident_context"] = state.get("incident_context") or {}
        detail["quality_findings"] = state.get("quality_findings") or {}
        detail["quality_status"] = state.get("quality_status") or "SUCCEEDED"
        summary = result.get("summary", "")
        if detail["quality_status"] == "PARTIAL" and hasattr(repo, "save_partial"):
            repo.save_partial(state["analysis_id"], summary=summary, detail=detail)
        else:
            repo.save_success(state["analysis_id"], summary=summary, detail=detail)
        return {"record_detail": detail}

    graph.add_node("prepare_analysis", prepare_analysis)
    graph.add_node("build_incident_context", build_incident_context)
    graph.add_node("supervisor_route", supervisor_route)
    graph.add_node("investigate_with_agent", investigate_with_agent)
    graph.add_node("validate_quality", validate_quality)
    graph.add_node("finalize_result", finalize_result)

    graph.add_edge(START, "prepare_analysis")
    graph.add_edge("prepare_analysis", "build_incident_context")
    graph.add_edge("build_incident_context", "supervisor_route")
    graph.add_edge("supervisor_route", "investigate_with_agent")
    graph.add_edge("investigate_with_agent", "validate_quality")
    graph.add_edge("validate_quality", "finalize_result")
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


def _build_incident_context(
    state: ServerErrorAnalysisState,
    analysis_config: dict[str, Any],
) -> dict[str, Any]:
    detail = state.get("record_detail") or {}
    scope = _build_scope(state, detail)
    return {
        "scope": scope,
        "baseline": _build_baseline(detail),
        "limits": {
            "loki_query_limit": analysis_config.get("loki_query_limit"),
            "max_evidence_tokens": analysis_config.get("max_evidence_tokens"),
        },
    }


def _build_scope(state: ServerErrorAnalysisState, detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": state.get("trace_id") or detail.get("trace_id") or detail.get("traceID"),
        "service_name": _first_present(detail, "service_name", "service", "app"),
        "status_code": _first_present(detail, "status_code", "status", "http_status_code"),
        "endpoint": _first_present(detail, "endpoint", "path", "route", "http_target", "url_path"),
        "time_range": state.get("time_range") or detail.get("time_range") or {},
    }


def _build_baseline(detail: dict[str, Any]) -> dict[str, Any]:
    evidence = detail.get("evidence") if isinstance(detail.get("evidence"), dict) else {}
    return {
        "dedup_basis": detail.get("dedup_basis"),
        "log_summary": evidence.get("log_summary") or _first_present(detail, "log_summary", "message", "line"),
        "no_trace_context": detail.get("no_trace_context"),
        "raw": detail,
    }


def _build_supervisor_decision(
    incident_context: dict[str, Any],
    mode: Literal["auto", "manual"],
) -> dict[str, Any]:
    scope = incident_context.get("scope") or {}
    trace_id = scope.get("trace_id")
    service_name = scope.get("service_name")
    time_range = scope.get("time_range") or {}
    required_sources = ["log"]
    skipped_sources = []

    if trace_id:
        strategy = "trace_first"
        required_sources.insert(0, "trace")
    else:
        strategy = "log_first"
        skipped_sources.append(
            {
                "source": "trace",
                "reason": "trace_id not provided; trace evidence cannot be scoped reliably.",
            }
        )

    if mode == "manual" or service_name or time_range:
        required_sources.append("metric")

    return {
        "strategy": strategy,
        "required_evidence_sources": required_sources,
        "skipped_evidence_sources": skipped_sources,
    }


def _first_present(source: dict, *keys: str):
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return value
    return None


def _build_agent_context(state: ServerErrorAnalysisState, analysis_config: dict[str, Any]) -> dict:
    mode = state["mode"]
    recursion_limit = analysis_config.get("auto_recursion_limit" if mode == "auto" else "manual_recursion_limit")
    return {
        "message": _build_investigation_brief(state),
        "mode": mode,
        "trace_id": state.get("trace_id"),
        "time_range": state.get("time_range") or {},
        "recursion_limit": recursion_limit,
        "loki_query_limit": analysis_config.get("loki_query_limit"),
        "max_evidence_tokens": analysis_config.get("max_evidence_tokens"),
    }


def _build_investigation_brief(state: ServerErrorAnalysisState) -> str:
    user_message = state.get("user_message") or "Analyze the HTTP 5xx server error and identify the most probable cause."
    incident_context = json.dumps(state.get("incident_context") or {}, ensure_ascii=False, default=str, sort_keys=True, indent=2)
    return "\n".join(
        [
            "Task: Analyze one HTTP 5xx incident.",
            "",
            "Incident context:",
            incident_context,
            "",
            "User request:",
            user_message,
        ]
    )


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


def _evaluate_result_quality(
    result: dict,
    analysis_config: dict[str, Any],
) -> tuple[Literal["SUCCEEDED", "PARTIAL"], dict[str, Any]]:
    evidence = result.get("evidence")
    confidence = result.get("confidence")
    threshold = analysis_config.get("partial_confidence_threshold", 0.4)
    reasons = []

    if not evidence:
        reasons.append("missing evidence")
    if isinstance(confidence, int | float) and confidence < threshold:
        reasons.append("confidence below threshold")
    if confidence is None:
        reasons.append("missing confidence")

    status = "PARTIAL" if reasons else "SUCCEEDED"
    return status, {
        "status": status,
        "reasons": reasons,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "evidence_count": _count_evidence_items(evidence),
    }


def _classify_result_quality(result: dict, analysis_config: dict[str, Any]) -> Literal["SUCCEEDED", "PARTIAL"]:
    status, _ = _evaluate_result_quality(result, analysis_config)
    return status


def _count_evidence_items(evidence) -> int:
    if isinstance(evidence, list):
        return len(evidence)
    if isinstance(evidence, dict):
        return len(evidence)
    return 0
