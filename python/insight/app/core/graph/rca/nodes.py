import json
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from app.core.graph.utils.token_counter import count_tokens

from .collectors import collect_evidence as collect_evidence_task
from .models import (
    EVIDENCE_SOURCES,
    DraftEvidencePlan,
    EvidenceResult,
    EvidenceTask,
    IncidentScope,
    RcaAnalysisState,
    RcaResult,
    RcaRunContext,
)
from .specs import CAPABILITY_SPECS

_DEFAULT_QUERY = "Analyze the incident and identify the most probable evidence-backed cause."
_MAX_INVESTIGATION_ROUNDS = 1
_TRIM_NOTE_TOKENS = 64
_PARTIAL_PLAN_REASONS = {
    "capability_unavailable",
    "time_range_missing",
    "fallback_unavailable",
}


class RcaGraphNodes:
    async def plan_evidence(
        self,
        state: RcaAnalysisState,
        runtime: Runtime[RcaRunContext],
    ) -> dict[str, Any]:
        scope = IncidentScope.model_validate(state.get("scope") or {})
        scope = _scope_with_discovered_trace(scope, state.get("merged_evidence"))
        filters = state.get("filters") or {}
        reinvestigating = bool(state.get("evidence_gaps"))
        draft = await _draft_plan(runtime.context.llm, state, scope)
        available = {
            capability: bool(
                runtime.context.collector_factory and runtime.context.collector_factory(capability)
            )
            for capability in CAPABILITY_SPECS
        }
        collected_capabilities = {
            item.get("capability")
            for item in state.get("evidence", [])
            if item.get("capability")
        }
        plan = validate_plan(
            draft,
            scope,
            filters,
            available,
            excluded_capabilities=collected_capabilities,
        )
        hypotheses = list(
            dict.fromkeys([*state.get("hypotheses", []), *draft.hypotheses])
        )[:5]
        return {
            "query": state.get("query") or _DEFAULT_QUERY,
            "available_capabilities": [
                capability for capability, is_available in available.items() if is_available
            ],
            "hypotheses": hypotheses,
            "evidence_gaps": [],
            "investigation_round": (
                int(state.get("investigation_round", 0)) + 1 if reinvestigating else 0
            ),
            "scope": scope.model_dump(mode="json"),
            "filters": filters,
            "evidence_plan": plan,
        }

    async def collect_evidence(
        self,
        state: RcaAnalysisState,
        runtime: Runtime[RcaRunContext],
    ) -> dict[str, Any]:
        evidence = await collect_evidence_task(
            state["task"],
            state["scope"],
            runtime.context.collector_factory,
            filters=state.get("filters") or {},
            plan_size=int(state.get("plan_size") or 1),
        )
        return {"evidence": [evidence.model_dump(mode="json")]}

    async def merge_evidence(
        self,
        state: RcaAnalysisState,
        runtime: Runtime[RcaRunContext],
    ) -> dict[str, Any]:
        items = [
            EvidenceResult.model_validate(item).model_dump(mode="json")
            for item in state.get("evidence", [])
        ]
        skipped = (state.get("evidence_plan") or {}).get("skipped", [])
        merged = _build_merged_evidence(items, skipped)
        config = runtime.context.analysis_config
        return {
            "merged_evidence": _fit_synthesis_evidence(
                merged,
                max_tokens=config.get("synthesis_evidence_max_tokens", 26_214),
                model_name=config.get("model_name", "gpt-4"),
            )
        }

    async def synthesize(
        self,
        state: RcaAnalysisState,
        runtime: Runtime[RcaRunContext],
    ) -> dict[str, Any]:
        if reason := (state.get("merged_evidence") or {}).get(
            "synthesis_blocked_reason"
        ):
            return {"analysis_result": None, "error_message": reason}
        if not (state.get("merged_evidence") or {}).get("evidence_catalog"):
            return {
                "analysis_result": None,
                "error_message": "No usable evidence was collected.",
            }
        system_prompt = runtime.context.analysis_config.get(
            "synthesis_system_prompt",
            "Return a grounded RCA result from the supplied evidence only.",
        ).format(current_time=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        system_prompt = (
            f"{system_prompt}\n\n"
            "Set conclusion_strength to CONFIRMED only when at least two independent raw evidence records "
            "support the causal claim without contradiction; use LIKELY for one grounded causal path and "
            "INCONCLUSIVE when evidence is sparse, indirect, or conflicting. For an INCONCLUSIVE result, "
            "put only concrete, collectable evidence gaps in next_checks."
        )
        for _ in range(2):
            try:
                result = await runtime.context.llm.with_structured_output(RcaResult).ainvoke(
                    [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "query": state.get("query"),
                                    "scope": state.get("scope"),
                                    "candidate_hypotheses": state.get("hypotheses") or [],
                                    "investigation_round": state.get("investigation_round", 0),
                                    "merged_evidence": _synthesis_evidence(
                                        state.get("merged_evidence")
                                    ),
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    ]
                )
                result = RcaResult.model_validate(result)
                scope = IncidentScope.model_validate(state.get("scope") or {})
                scope_defaults = {
                    "affected_service": scope.service_name,
                    "affected_endpoint": scope.endpoint,
                }
                result = result.model_copy(
                    update={
                        field: value
                        for field, value in scope_defaults.items()
                        if getattr(result, field) is None and value is not None
                    }
                )
                return {"analysis_result": result.model_dump(mode="json")}
            except Exception as exc:
                last_error = str(exc)
        return {"analysis_result": None, "error_message": last_error}

    async def validate_result(
        self,
        state: RcaAnalysisState,
        runtime: Runtime[RcaRunContext],
    ) -> dict[str, Any]:
        return _validate_result(
            state.get("analysis_result"),
            state.get("merged_evidence"),
            runtime.context.analysis_config,
        )


def _deduplicate(items: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for item in items:
        fingerprint = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(item)
    return unique


def _build_merged_evidence(items: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    limitations = [
        {"source": item["source"], "reason": reason} for item in items for reason in item.get("limitations", [])
    ]
    limitations.extend(skipped)
    return {
        "items": items,
        "evidence_catalog": _build_evidence_catalog(items),
        "sources": _source_statuses(items),
        "capabilities": {item["capability"]: item["status"] for item in items},
        "limitations": _deduplicate(limitations),
        "discovered_trace_ids": sorted(
            {trace_id for item in items for trace_id in item.get("discovered_trace_ids", [])}
        ),
    }


def _fit_synthesis_evidence(
    merged: dict[str, Any],
    *,
    max_tokens: int,
    model_name: str,
) -> dict[str, Any]:
    """Shrink the citable catalog until the synthesis payload fits.

    Blocking on overflow means collecting *more* evidence can make an analysis fail
    outright — the LLM then sees none of it. Dropping the tail of the noisiest
    capability costs a few records and keeps the conclusion. Only a payload that
    cannot hold a single record still blocks.
    """
    if _synthesis_tokens(merged, model_name) <= max_tokens:
        return merged

    # The trim limitation is appended after fitting, so leave room for it.
    max_tokens = max(max_tokens - _TRIM_NOTE_TOKENS, 0)
    catalog = merged.get("evidence_catalog") or []
    # Price each record once against the empty-catalog payload instead of re-counting
    # the whole payload per record; the exact total is verified below.
    used = _synthesis_tokens({**merged, "evidence_catalog": []}, model_name)
    kept: list[dict[str, Any]] = []
    for record in _interleave_by_capability(catalog):
        size = count_tokens(_compact_json(record), model_name)
        if used + size > max_tokens:
            continue
        kept.append(record)
        used += size
    while kept and _synthesis_tokens({**merged, "evidence_catalog": kept}, model_name) > max_tokens:
        kept.pop()

    dropped = len(catalog) - len(kept)
    if not kept:
        merged["synthesis_blocked_reason"] = "synthesis_evidence_budget_exceeded"
        return merged

    # Preserve catalog order so evidence still reads trace -> log -> metric.
    kept_ids = {record.get("evidence_id") for record in kept}
    merged["evidence_catalog"] = [
        record for record in catalog if record.get("evidence_id") in kept_ids
    ]
    merged["limitations"] = [
        *merged.get("limitations", []),
        {"source": "synthesis", "reason": f"evidence_trimmed_to_context:{dropped}"},
    ]
    return merged


def _interleave_by_capability(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin over capabilities so a chatty one cannot crowd out the others."""
    by_capability: dict[str, list[dict[str, Any]]] = {}
    for record in catalog:
        by_capability.setdefault(record.get("capability") or record.get("source") or "", []).append(record)
    ordered: list[dict[str, Any]] = []
    while any(by_capability.values()):
        for records in by_capability.values():
            if records:
                ordered.append(records.pop(0))
    return ordered


def _synthesis_tokens(merged: dict[str, Any], model_name: str) -> int:
    return count_tokens(_compact_json(_synthesis_evidence(merged)), model_name)


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _synthesis_evidence(merged: dict[str, Any] | None) -> dict[str, Any]:
    merged = merged or {}
    return {
        key: merged.get(key, [] if key in {"evidence_catalog", "limitations"} else {})
        for key in ("evidence_catalog", "sources", "capabilities", "limitations")
    }


def _build_evidence_catalog(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    order = {source: index for index, source in enumerate(EVIDENCE_SOURCES)}
    for item in sorted(items, key=lambda value: order.get(value["source"], len(order))):
        if item.get("status") not in {"OK", "PARTIAL"}:
            continue
        catalog.extend(item.get("records", []))
    return catalog


def _source_statuses(items: list[dict[str, Any]]) -> dict[str, str]:
    rank = {"OK": 0, "NO_DATA": 1, "PARTIAL": 1, "SKIPPED": 2, "FAILED": 3}
    statuses = {}
    for item in items:
        source = item["source"]
        status = item["status"]
        if source not in statuses or rank.get(status, 3) > rank.get(statuses[source], 3):
            statuses[source] = status
    return statuses


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


async def _draft_plan(
    llm,
    state: RcaAnalysisState,
    scope: IncidentScope,
) -> DraftEvidencePlan:
    if llm is None:
        return DraftEvidencePlan()
    try:
        draft = await llm.with_structured_output(DraftEvidencePlan).ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Draft bounded RCA evidence tasks as explicit telemetry capabilities. Start with at least "
                        "two plausible hypotheses when the request is ambiguous. Select only capabilities from "
                        "the supplied catalog and use each task to test or disprove a hypothesis. During a "
                        "follow-up round, select only new capabilities that can close a listed evidence gap."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": state.get("query"),
                            "capability_catalog": {
                                capability: {
                                    "source": spec["source"],
                                    "when_to_use": spec["when_to_use"],
                                }
                                for capability, spec in CAPABILITY_SPECS.items()
                            },
                            "scope": scope.model_dump(mode="json"),
                            "filters": state.get("filters") or {},
                            "previous_capabilities": sorted(
                                {
                                    item.get("capability")
                                    for item in state.get("evidence", [])
                                    if item.get("capability")
                                }
                            ),
                            "previous_hypotheses": state.get("hypotheses") or [],
                            "evidence_gaps": state.get("evidence_gaps") or [],
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ]
        )
        return DraftEvidencePlan.model_validate(draft)
    except Exception:
        return DraftEvidencePlan()


def build_rca_graph(*, checkpointer=None):
    nodes = RcaGraphNodes()
    graph = StateGraph(RcaAnalysisState, context_schema=RcaRunContext)
    graph.add_node("plan_evidence", nodes.plan_evidence)
    graph.add_node("collect_evidence", nodes.collect_evidence)
    graph.add_node("merge_evidence", nodes.merge_evidence)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("validate_result", nodes.validate_result)
    graph.add_edge(START, "plan_evidence")
    graph.add_conditional_edges("plan_evidence", fan_out, ["collect_evidence", "merge_evidence"])
    graph.add_edge("collect_evidence", "merge_evidence")
    graph.add_edge("merge_evidence", "synthesize")
    graph.add_edge("synthesize", "validate_result")
    graph.add_conditional_edges(
        "validate_result",
        route_after_validation,
        ["plan_evidence", END],
    )
    return graph.compile(checkpointer=checkpointer)


def _validate_result(result: dict | None, merged_evidence: dict | None, analysis_config: dict) -> dict[str, Any]:
    merged_evidence = merged_evidence or {}
    evidence = merged_evidence.get("items", [])
    usable_sources = {item["source"] for item in evidence if item.get("status") in {"OK", "PARTIAL"}}
    catalog = {
        item["evidence_id"]: item
        for item in merged_evidence.get("evidence_catalog", [])
        if item.get("source") in usable_sources
    }
    threshold = analysis_config.get("partial_confidence_threshold", 0.4)
    confidence = result.get("confidence") if isinstance(result, dict) else None
    execution_reasons = _execution_reasons(
        result,
        evidence,
        usable_sources,
        merged_evidence.get("limitations", []),
        confidence,
    )
    # An empty catalog because every source ran and found nothing is a finding, not a
    # broken analysis. Only a source that actually failed makes the run unusable.
    no_telemetry = bool(evidence) and all(
        item.get("status") in {"NO_DATA", "SKIPPED"} for item in evidence
    )
    if not catalog:
        _append_unique(
            execution_reasons,
            "no telemetry data in the requested window" if no_telemetry else "no usable evidence",
        )
        result = None
    conclusion_reasons: list[str] = []
    evidence_gaps: list[str] = []
    if isinstance(result, dict):
        result, supporting, contradiction_count = _ground_result_references(
            result,
            catalog,
            execution_reasons,
        )
        result, conclusion_reasons, evidence_gaps = _assess_conclusion(
            result,
            catalog,
            supporting,
            contradiction_count,
            threshold,
        )

    execution_reasons = list(dict.fromkeys(execution_reasons))
    conclusion_reasons = list(dict.fromkeys(conclusion_reasons))
    if catalog:
        status = "PARTIAL" if execution_reasons else "SUCCEEDED"
    else:
        status = "PARTIAL" if no_telemetry else "FAILED"
    return {
        "analysis_result": result,
        "evidence_gaps": evidence_gaps,
        "result_validation": {
            "status": status,
            "no_telemetry": not catalog and no_telemetry,
            "reasons": execution_reasons,
            "conclusion_reasons": conclusion_reasons,
            "confidence": result.get("confidence") if isinstance(result, dict) else confidence,
            "confidence_threshold": threshold,
            "evidence_count": len(evidence),
            "usable_evidence_count": len(catalog),
            "conclusion_strength": (
                result.get("conclusion_strength") if isinstance(result, dict) else "INCONCLUSIVE"
            ),
        },
    }


def _execution_reasons(
    result: dict | None,
    evidence: list[dict[str, Any]],
    usable_sources: set[str],
    limitations: list[Any],
    confidence: Any,
) -> list[str]:
    reasons = []
    if not result:
        reasons.append("missing result")
    if not evidence:
        reasons.append("missing evidence")
    elif not usable_sources:
        reasons.append("no usable evidence")
    reasons.extend(
        f"incomplete evidence: {item.get('source')}"
        for item in evidence
        # NO_DATA ran to completion — an empty window is a finding, not an execution gap.
        if item.get("status") not in {"OK", "NO_DATA"}
    )
    templates = {
        "capability_unavailable": "planned evidence unavailable: {source}",
        "time_range_missing": "planned evidence time range missing: {source}",
        "fallback_unavailable": "fallback evidence unavailable: {source}",
    }
    for limitation in limitations:
        if isinstance(limitation, dict) and limitation.get("reason") in _PARTIAL_PLAN_REASONS:
            reasons.append(
                templates[limitation["reason"]].format(
                    source=limitation.get("source") or "unknown"
                )
            )
    if confidence is None:
        reasons.append("missing confidence")
    return reasons


def _ground_result_references(
    result: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    execution_reasons: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    result = {**result}
    result["evidence"] = [
        grounded
        for item in result.get("evidence", [])
        if (grounded := _ground_reference(item, catalog, execution_reasons, "evidence"))
        is not None
    ]
    grounded_hypotheses = []
    for hypothesis in result.get("hypotheses", []):
        grounded_hypothesis = {**hypothesis}
        for key in ("supporting_evidence", "contradicting_evidence"):
            grounded_hypothesis[key] = [
                grounded
                for evidence_id in hypothesis.get(key, [])
                if (
                    grounded := _ground_hypothesis_reference(
                        evidence_id,
                        catalog,
                        execution_reasons,
                    )
                )
                is not None
            ]
        grounded_hypotheses.append(grounded_hypothesis)
    result["hypotheses"] = grounded_hypotheses
    if result.get("probable_cause") and grounded_hypotheses:
        result["probable_cause"] = grounded_hypotheses[0]["cause"]
    supporting = [item for item in result["evidence"] if item.get("supports_cause")]
    contradiction_count = sum(
        len(hypothesis["contradicting_evidence"]) for hypothesis in grounded_hypotheses
    )
    return result, supporting, contradiction_count


def _assess_conclusion(
    result: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    supporting: list[dict[str, Any]],
    contradiction_count: int,
    threshold: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    reasons = []
    confidence = result.get("confidence")
    if confidence is not None and confidence < threshold:
        reasons.append("confidence below threshold")
    if result.get("probable_cause") and not supporting:
        reasons.append("probable cause has no grounded supporting evidence")

    strength = result.get("conclusion_strength") or "INCONCLUSIVE"
    independent_support = {
        (
            catalog[item["evidence_id"]]["source"],
            catalog[item["evidence_id"]].get("capability")
            or catalog[item["evidence_id"]]["source"],
        )
        for item in supporting
        if item.get("evidence_id") in catalog
    }
    if strength == "CONFIRMED" and (
        len(independent_support) < 2 or contradiction_count
    ):
        reasons.append(
            "confirmed conclusion requires two independent supports without contradiction"
        )
        strength = "LIKELY" if supporting and not contradiction_count else "INCONCLUSIVE"
    if strength == "LIKELY" and not supporting:
        reasons.append("likely conclusion requires grounded causal support")
        strength = "INCONCLUSIVE"
    if not supporting and result.get("probable_cause"):
        strength = "INCONCLUSIVE"

    # With no causal claim, confidence rates the health assessment instead, so neither
    # the causal-strength cap nor a re-investigation round applies to it.
    claims_cause = bool(result.get("probable_cause"))
    settled_healthy = not claims_cause and confidence is not None and confidence >= threshold

    evidence_gaps = []
    if strength == "INCONCLUSIVE":
        reasons.append("conclusion inconclusive")
        if not settled_healthy:
            evidence_gaps = [
                check.strip()
                for check in result.get("next_checks", [])
                if isinstance(check, str) and check.strip()
            ][:3]
    confidence_out = confidence
    if confidence is not None and claims_cause:
        strength_cap = {"CONFIRMED": 1.0, "LIKELY": 0.84, "INCONCLUSIVE": threshold}[strength]
        confidence_out = min(confidence, strength_cap)
    return (
        {
            **result,
            "conclusion_strength": strength,
            "confidence": confidence_out,
        },
        reasons,
        evidence_gaps,
    )


def route_after_validation(state: RcaAnalysisState) -> str:
    if int(state.get("investigation_round", 0)) >= _MAX_INVESTIGATION_ROUNDS:
        return END
    if not state.get("evidence_gaps"):
        return END
    available_capabilities = set(state.get("available_capabilities") or [])
    collected_capabilities = {
        item.get("capability")
        for item in state.get("evidence", [])
        if item.get("capability")
    }
    scope = IncidentScope.model_validate(state.get("scope") or {})
    scope = _scope_with_discovered_trace(scope, state.get("merged_evidence"))
    filters = state.get("filters") or {}
    available = {
        capability: capability in available_capabilities
        for capability in CAPABILITY_SPECS
    }
    for capability in available_capabilities - collected_capabilities:
        if _skip_reason(capability, scope, filters, available, set()) is None:
            return "plan_evidence"
    return END


def _scope_with_discovered_trace(
    scope: IncidentScope,
    merged_evidence: dict[str, Any] | None,
) -> IncidentScope:
    discovered = (merged_evidence or {}).get("discovered_trace_ids") or []
    if scope.trace_id or not discovered:
        return scope
    return scope.model_copy(update={"trace_id": str(discovered[0])})


def _ground_reference(
    reference: dict[str, Any],
    catalog: dict[str, dict[str, str]],
    reasons: list[str],
    kind: str,
) -> dict[str, Any] | None:
    evidence_id = reference.get("evidence_id")
    catalog_item = catalog.get(evidence_id)
    if catalog_item is None:
        reasons.append(f"ungrounded {kind}_id: {evidence_id}")
        return None
    if reference.get("source") != catalog_item["source"]:
        reasons.append(f"{kind} source mismatch: {evidence_id}")
        return None
    return {**reference}


def _ground_hypothesis_reference(
    evidence_id: str,
    catalog: dict[str, dict[str, Any]],
    reasons: list[str],
) -> str | None:
    if evidence_id not in catalog:
        reasons.append(f"ungrounded hypothesis evidence_id: {evidence_id}")
        return None
    return evidence_id


async def fan_out(state: RcaAnalysisState) -> list[Send]:
    tasks = (state.get("evidence_plan") or {}).get("tasks", [])
    if not tasks:
        return [Send("merge_evidence", dict(state))]
    return [
        Send(
            "collect_evidence",
            {
                "task": task,
                "scope": state["scope"],
                "filters": state.get("filters") or {},
                "plan_size": len(tasks),
            },
        )
        for task in tasks
    ]


def validate_plan(
    draft: DraftEvidencePlan | dict[str, Any] | None,
    scope: IncidentScope,
    filters: dict[str, Any] | None,
    available: dict[str, bool] | None,
    *,
    excluded_capabilities: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(draft, DraftEvidencePlan):
        try:
            draft = DraftEvidencePlan.model_validate(draft or {})
        except Exception:
            draft = DraftEvidencePlan()

    available = available or {}
    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    excluded_capabilities = excluded_capabilities or set()

    for task in draft.tasks:
        capability = task.capability
        spec = CAPABILITY_SPECS.get(capability)
        source = spec["source"] if spec else "unknown"
        reason = _skip_reason(
            capability,
            scope,
            filters or {},
            available,
            seen,
            excluded_capabilities,
        )
        if reason:
            skipped.append({"capability": capability, "source": source, "reason": reason})
            continue
        tasks.append(_dump_task(task, scope, filters or {}))
        seen.add(capability)

    if not tasks:
        fallback_capability = next(
            (
                capability
                for capability in (
                    "traces.get",
                    "logs.search",
                    "traces.search",
                    "metrics.infrastructure",
                    "logs.volume",
                )
                if _skip_reason(
                    capability,
                    scope,
                    filters or {},
                    available,
                    set(),
                    excluded_capabilities,
                )
                is None
            ),
            None,
        )
        if fallback_capability:
            tasks.append(
                _dump_task(
                    EvidenceTask(
                        capability=fallback_capability,
                        focus="Collect the most relevant evidence for the reported incident.",
                    ),
                    scope,
                    filters or {},
                )
            )
        else:
            skipped.append(
                {"capability": "logs.search", "source": "log", "reason": "fallback_unavailable"}
            )

    return {"tasks": tasks, "skipped": skipped}


def _skip_reason(
    capability: str,
    scope: IncidentScope,
    filters: dict[str, Any],
    available: dict[str, bool],
    seen: set[str],
    excluded_capabilities: set[str] | None = None,
) -> str | None:
    if capability not in CAPABILITY_SPECS:
        return "unknown_capability"
    if capability in seen:
        return "duplicate_capability"
    if capability in (excluded_capabilities or set()):
        return "already_collected"
    if not available.get(capability):
        return "capability_unavailable"
    if capability == "traces.get" and not scope.trace_id:
        return "trace_id_missing"
    if capability != "traces.get" and not _has_time_range(scope):
        return "time_range_missing"
    # A thin scope is not a reason to skip a source. Collectors already discover their own
    # database, measurement and attributes, so refusing to plan them here only guaranteed
    # that a broad question ("what caused these errors?") never looked past the logs.
    # Breadth is recorded on the task instead — see _task_breadth.
    return None


def _dump_task(task: EvidenceTask, scope: IncidentScope, filters: dict[str, Any]) -> dict[str, Any]:
    spec = CAPABILITY_SPECS[task.capability]
    return {
        **task.model_dump(mode="json"),
        "source": spec["source"],
        "breadth": _task_breadth(task.capability, scope, filters or {}),
    }


_METRIC_SCOPE_KEYS = {"ns_id", "infra_id", "node_id", "measurement", "database_name"}


def _task_breadth(capability: str, scope: IncidentScope, filters: dict[str, Any]) -> str:
    """Whether the request already names a target, or the collector must find one.

    A broad task is expected to start from discovery and return a wider sample; the time
    window, row limits and query budget still bound it.
    """
    if capability == "metrics.infrastructure":
        narrowed = _METRIC_SCOPE_KEYS.intersection(scope.attributes) or _METRIC_SCOPE_KEYS.intersection(filters)
    else:
        narrowed = bool(
            scope.trace_id
            or scope.service_name
            or scope.endpoint
            or scope.status_code
            or scope.attributes
            or filters
        )
    return "scoped" if narrowed else "broad"


def _has_time_range(scope: IncidentScope) -> bool:
    time_range = scope.time_range
    return bool(getattr(time_range, "start", None) and getattr(time_range, "end", None))
