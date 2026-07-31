import asyncio
import inspect
import json
import re
import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.core.llm.structured_agent import wrap_with_structured_fallback

from .evidence_store import EvidenceStore
from .models import (
    EvidenceRecord,
    EvidenceResult,
    EvidenceTask,
    IncidentScope,
    SourceCollector,
    ToolTraceEntry,
    ValidatedSourceFilters,
)
from .specs import CAPABILITY_SPECS


class _ToolInvocationError(RuntimeError):
    def __init__(self, message: str, traces: list[ToolTraceEntry]):
        super().__init__(message)
        self.traces = list(traces)


_COLLECTOR_SYSTEM_PROMPT = (
    "Collect telemetry for one bounded RCA capability. Use only supplied safe tools; "
    "do not make final root-cause claims."
)


def build_collector_runner(llm, query_tools, *, middleware, instructions: str = ""):
    """Build the per-capability collector agent.

    ``instructions`` carries the capability's own usage notes. Every collector used to
    share the same two generic sentences, which left the model to guess which parameters
    a source expects and what an empty answer means.
    """
    system_prompt = _COLLECTOR_SYSTEM_PROMPT
    if instructions.strip():
        system_prompt = f"{system_prompt}\n\n{instructions.strip()}"
    agent = create_agent(
        model=llm,
        tools=query_tools,
        checkpointer=None,
        system_prompt=system_prompt,
        response_format=EvidenceResult,
        middleware=middleware,
    )
    return wrap_with_structured_fallback(agent, llm, EvidenceResult)


async def collect_evidence(
    task: EvidenceTask | dict[str, Any],
    scope: IncidentScope | dict[str, Any],
    collector_factory,
    *,
    filters: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    plan_size: int = 1,
) -> EvidenceResult:
    task = EvidenceTask.model_validate(task)
    scope = IncidentScope.model_validate(scope)
    spec = CAPABILITY_SPECS[task.capability]
    source = spec["source"]
    collector = collector_factory(task.capability) if collector_factory else None
    if collector is None:
        return _stub(source, "SKIPPED", "collector_unavailable", task.capability)
    if not isinstance(collector, SourceCollector):
        return _stub(source, "FAILED", "invalid_collector", task.capability)
    if collector.evidence_store is None:
        collector.evidence_store = EvidenceStore()
    # Claim this capability's share before any await, so the split is decided by the
    # plan rather than by which collector happens to return first.
    collector.evidence_store.reserve(task.capability, plan_size)

    timeout = timeout_seconds or spec["timeout_seconds"]
    try:
        collection = (
            _collect_trace(task, scope, collector, timeout)
            if task.capability == "traces.get"
            else _collect_discovered_source(
                task,
                scope,
                filters or {},
                collector,
            )
        )
        evidence = await asyncio.wait_for(collection, timeout=timeout)

        evidence.capability = task.capability
        return evidence
    except TimeoutError:
        return _stub(source, "FAILED", "collector_timeout", task.capability)
    except _ToolInvocationError as exc:
        evidence = _stub(source, "FAILED", str(exc), task.capability)
        evidence.tool_trace = exc.traces
        return evidence
    except Exception as exc:
        return _stub(source, "FAILED", str(exc), task.capability)


async def _collect_trace(
    task: EvidenceTask,
    scope: IncidentScope,
    collector,
    timeout: float,
) -> EvidenceResult:
    if not scope.trace_id:
        return _stub("trace", "SKIPPED", "trace_id_missing")

    args = {"trace_id": scope.trace_id}
    traces: list[ToolTraceEntry] = []
    tool = collector.tools.get("get-trace")
    if tool is None:
        return _stub("trace", "FAILED", "collector_tool_missing:get-trace")
    raw = await asyncio.wait_for(
        _invoke_traced(collector, tool, "get-trace", args, traces),
        timeout=timeout,
    )
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="json")

    if not traces:
        traces.append(_tool_trace("get-trace", args, raw))
    payload, summary = _summarize_trace(raw, scope.trace_id)
    return await _direct_evidence_result(
        task,
        scope,
        collector,
        source="trace",
        tool="get-trace",
        args=args,
        raw=payload,
        traces=traces,
        summary=summary,
    )


# OTLP attribute values are single-key wrappers; these are the ones worth keeping.
_OTLP_VALUE_KEYS = ("stringValue", "intValue", "doubleValue", "boolValue")
_SPAN_ATTRIBUTE_KEYS = (
    "http.method",
    "http.route",
    "http.target",
    "http.status_code",
    "db.system",
    "db.statement",
    "rpc.method",
    "error",
    "exception.type",
    "exception.message",
)


def _summarize_trace(raw: Any, trace_id: str | None) -> tuple[Any, str]:
    """Turn an OTLP trace into a bounded span table the model can read directly.

    Tempo answers with ``{"trace": {"resourceSpans": [...]}}``. Handing that nested
    blob to the store spills it, and the agent then has to guess JSON Pointers into a
    structure it cannot see — which is how a perfectly retrieved trace ended up as a
    FAILED capability. The shape is fixed, so code flattens it instead: every error
    span plus the slowest ones, which is what a latency question actually needs.
    """
    spans = _otlp_spans(raw)
    if not spans:
        # Unknown payload shape (non-OTLP Tempo build): keep the previous behaviour.
        return raw, f"Trace {trace_id} retrieved."

    errors = [span for span in spans if span["error"]]
    limit = CAPABILITY_SPECS["traces.get"]["max_rows"]
    selected = sorted(errors, key=lambda span: -span["duration_ms"])[:limit]
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(
            sorted(
                (span for span in spans if not span["error"]),
                key=lambda span: -span["duration_ms"],
            )[:remaining]
        )
    services = sorted({span["service"] for span in spans if span["service"]})
    total_ms = max((span["duration_ms"] for span in spans), default=0.0)
    payload = {
        "trace_id": trace_id,
        "span_count": len(spans),
        "error_span_count": len(errors),
        "services": services,
        "total_duration_ms": total_ms,
        "truncated_spans": max(len(spans) - len(selected), 0),
        "spans": selected,
    }
    summary = (
        f"Trace {trace_id}: {len(spans)} span(s) across {len(services)} service(s), "
        f"{len(errors)} error span(s), {total_ms:.1f} ms end to end."
    )
    return payload, summary


def _otlp_spans(raw: Any) -> list[dict[str, Any]]:
    """Flatten resourceSpans -> scopeSpans -> spans into comparable rows."""
    if not isinstance(raw, dict):
        return []
    resource_spans = ((raw.get("trace") or {}).get("resourceSpans")) or []
    if not isinstance(resource_spans, list):
        return []
    rows: list[dict[str, Any]] = []
    for resource_span in resource_spans:
        if not isinstance(resource_span, dict):
            continue
        resource = _otlp_attributes((resource_span.get("resource") or {}).get("attributes"))
        for scope_span in resource_span.get("scopeSpans") or []:
            for span in (scope_span or {}).get("spans") or []:
                if not isinstance(span, dict):
                    continue
                attributes = _otlp_attributes(span.get("attributes"))
                status = span.get("status") or {}
                status_code = str(status.get("code") or "")
                rows.append(
                    {
                        "service": str(resource.get("service.name") or ""),
                        "name": str(span.get("name") or ""),
                        "span_id": str(span.get("spanId") or ""),
                        "parent_span_id": str(span.get("parentSpanId") or ""),
                        "kind": str(span.get("kind") or ""),
                        "duration_ms": _otlp_duration_ms(span),
                        "status": status_code,
                        "status_message": str(status.get("message") or "")[:200],
                        "error": "ERROR" in status_code.upper(),
                        "attributes": {
                            key: attributes[key]
                            for key in _SPAN_ATTRIBUTE_KEYS
                            if key in attributes
                        },
                    }
                )
    return rows


def _otlp_attributes(items: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key, value = item.get("key"), item.get("value")
        if not key or not isinstance(value, dict):
            continue
        for value_key in _OTLP_VALUE_KEYS:
            if value_key in value:
                found = value[value_key]
                values[key] = found[:200] if isinstance(found, str) else found
                break
    return values


def _otlp_duration_ms(span: dict[str, Any]) -> float:
    try:
        start = int(span.get("startTimeUnixNano") or 0)
        end = int(span.get("endTimeUnixNano") or 0)
    except (TypeError, ValueError):
        return 0.0
    return round(max(end - start, 0) / 1_000_000, 3)


async def _collect_discovered_source(
    task: EvidenceTask,
    scope: IncidentScope,
    filters: dict[str, Any],
    collector: SourceCollector,
) -> EvidenceResult:
    if task.capability.startswith("logs."):
        return await _collect_log(task, scope, filters, collector)
    if task.capability == "metrics.infrastructure":
        return await _collect_metric(task, scope, filters, collector)
    if task.capability == "traces.search":
        return await _collect_trace_search(task, scope, filters, collector)
    return _stub(CAPABILITY_SPECS[task.capability]["source"], "FAILED", "unsupported_capability")


async def _collect_log(
    task: EvidenceTask,
    scope: IncidentScope,
    filters: dict[str, Any],
    collector: SourceCollector,
) -> EvidenceResult:
    traces: list[ToolTraceEntry] = []
    datasource_tool = _required_tool(collector, "list_datasources")
    label_names_tool = _required_tool(collector, "list_loki_label_names")
    label_values_tool = _required_tool(collector, "list_loki_label_values")
    query_name = {
        "logs.search": "query_loki_logs",
        "logs.volume": "query_loki_stats",
    }[task.capability]
    query_tool = _required_tool(collector, query_name)

    datasources = await _invoke_traced(
        collector,
        datasource_tool,
        "list_datasources",
        {"type": "loki", "limit": 50},
        traces,
    )
    datasource_uid = _find_loki_datasource_uid(datasources)
    if not datasource_uid:
        return _evidence_with_traces("log", "SKIPPED", "loki_datasource_not_found", traces)

    window = _time_window(scope)
    discovery_args = {"datasourceUid": datasource_uid, **window}
    label_names = await _invoke_traced(
        collector,
        label_names_tool,
        "list_loki_label_names",
        discovery_args,
        traces,
    )
    labels = [value for value in _tabular_values(label_names) if _is_identifier(value)]
    if not labels:
        return _evidence_with_traces("log", "SKIPPED", "loki_labels_not_found", traces)
    validated = await _validate_loki_filters(
        scope,
        filters,
        datasource_uid,
        labels,
        window,
        collector,
        label_values_tool,
        traces,
    )
    selector = _build_log_selector(labels, validated.verified_filters)

    if task.capability == "logs.volume":
        args = {
            "datasourceUid": datasource_uid,
            "logql": selector,
            "startRfc3339": window["startRfc3339"],
            "endRfc3339": window["endRfc3339"],
        }
        raw = await _invoke_traced(collector, query_tool, query_name, args, traces)
        evidence = await _direct_evidence_result(
            task,
            scope,
            collector,
            source="log",
            tool=query_name,
            args=args,
            raw=raw,
            traces=traces,
            validated=validated,
        )
        evidence.discovered_trace_ids = _trace_ids_from_records(evidence.records)
        _add_ignored_limitations(evidence, validated)
        return evidence

    if collector.runner_factory is None:
        return _evidence_with_traces("log", "FAILED", "collector_runner_unavailable", traces)

    budget = _QueryBudget(
        "query_verified_loki_logs",
        CAPABILITY_SPECS[task.capability].get("max_queries", 1),
        traces,
    )
    allowed_refs: set[str] = set()

    async def query_verified_logs(terms: list[str] | None = None) -> Any:
        safe_terms = _safe_log_terms(terms)
        logql = selector
        if safe_terms:
            encoded_terms = [_escape_string(re.escape(term)) for term in safe_terms]
            logql = f'{selector} |~ "(?i)({"|".join(encoded_terms)})"'
        args = {
            "datasourceUid": datasource_uid,
            "logql": logql,
            "startRfc3339": window["startRfc3339"],
            "endRfc3339": window["endRfc3339"],
            "limit": CAPABILITY_SPECS["logs.search"]["max_rows"],
        }
        key = _query_key("query_loki_logs", args)
        if rejected := budget.blocked(key, {"terms": safe_terms}):
            return rejected
        raw = await _invoke_traced(collector, query_tool, "query_loki_logs", args, traces)
        budget.record(key)
        return _capture_tool_result(
            collector,
            "log",
            task.capability,
            "query_loki_logs",
            args,
            raw,
            allowed_refs,
        )

    safe_tool = StructuredTool.from_function(
        coroutine=query_verified_logs,
        name="query_verified_loki_logs",
        description=(
            "Query incident logs in the authoritative window. Supply at most eight literal search terms, "
            "or omit them for a bounded representative sample. Labels, time range, escaping, and result "
            "limits are enforced by code."
        ),
    )
    evidence = await _run_source_agent(
        task,
        scope,
        collector,
        [safe_tool, _inspection_tool(collector, allowed_refs)],
        validated,
        allowed_refs,
    )
    _finalize_agent_evidence(evidence, task.capability, collector.evidence_store)
    evidence.discovered_trace_ids = _trace_ids_from_records(evidence.records)
    evidence.tool_trace = [*traces, *evidence.tool_trace]
    _add_ignored_limitations(evidence, validated)
    return evidence


async def _collect_trace_search(
    task: EvidenceTask,
    scope: IncidentScope,
    filters: dict[str, Any],
    collector: SourceCollector,
) -> EvidenceResult:
    if collector.runner_factory is None:
        return _stub("trace", "FAILED", "collector_runner_unavailable", task.capability)
    query_tool = _required_tool(collector, "traceql-search")
    traces: list[ToolTraceEntry] = []
    allowed_refs: set[str] = set()
    budget = _QueryBudget(
        "search_verified_traces",
        CAPABILITY_SPECS[task.capability].get("max_queries", 1),
        traces,
    )
    window = _tempo_window(scope)
    validated = await _validate_tempo_attributes(scope, filters, collector, traces)
    verified_attributes = dict(validated.verified_filters)
    # A broad task named no service or endpoint on purpose: search the window itself rather
    # than refusing. The time window, row limit and query budget still bound the search.
    if task.breadth != "broad" and not (
        scope.service_name or scope.endpoint or validated.verified_filters
    ):
        evidence = _evidence_with_traces("trace", "SKIPPED", "trace_scope_unverified", traces)
        _add_ignored_limitations(evidence, validated)
        return evidence

    async def search_verified_traces(error_only: bool = False, min_duration_ms: int | None = None) -> Any:
        if min_duration_ms is not None and not 1 <= min_duration_ms <= 300_000:
            return {"error": "invalid_min_duration_ms"}
        query = _build_traceql_filter(
            scope,
            attributes=verified_attributes,
            error_only=error_only,
            min_duration_ms=min_duration_ms,
        )
        args = {"query": query, **window}
        if _tool_accepts_argument(query_tool, "limit"):
            args["limit"] = CAPABILITY_SPECS["traces.search"]["max_rows"]
        key = _query_key("traceql-search", args)
        if rejected := budget.blocked(
            key,
            {"error_only": error_only, "min_duration_ms": min_duration_ms},
        ):
            return rejected
        raw = await _invoke_traced(collector, query_tool, "traceql-search", args, traces)
        budget.record(key)
        return _capture_tool_result(
            collector,
            "trace",
            task.capability,
            "traceql-search",
            args,
            raw,
            allowed_refs,
        )

    safe_tool = StructuredTool.from_function(
        coroutine=search_verified_traces,
        name="search_verified_traces",
        description=(
            "Search only the authoritative service/endpoint and time window. Choose whether to require errors "
            "and an optional minimum duration in milliseconds."
        ),
    )
    validated.verified_filters = {
        **({"service_name": scope.service_name} if scope.service_name else {}),
        **({"endpoint": scope.endpoint} if scope.endpoint else {}),
        **validated.verified_filters,
    }
    validated.discovery["allowed_predicates"] = ["error_only", "min_duration_ms"]
    evidence = await _run_source_agent(
        task,
        scope,
        collector,
        [safe_tool, _inspection_tool(collector, allowed_refs)],
        validated,
        allowed_refs,
    )
    _finalize_agent_evidence(evidence, task.capability, collector.evidence_store)
    evidence.discovered_trace_ids = _trace_ids_from_records(evidence.records)
    evidence.tool_trace = [*traces, *evidence.tool_trace]
    _add_ignored_limitations(evidence, validated)
    return evidence


def _tempo_scope_constraints(
    scope: IncidentScope,
) -> tuple[list[tuple[str, str, Any, bool]], set[str]]:
    candidates: list[tuple[str, str, Any, bool]] = []
    authoritative_keys: set[str] = set()
    if scope.status_code is not None:
        candidates.extend(
            ("status_code", name, scope.status_code, True)
            for name in (
                "span.http.response.status_code",
                "span.http.status_code",
            )
        )
        authoritative_keys.update(
            {
                "status_code",
                "http.status_code",
                "http.response.status_code",
                "span.http.status_code",
                "span.http.response.status_code",
            }
        )
    if scope.service_name is not None:
        authoritative_keys.update({"service_name", "resource.service.name"})
    if scope.endpoint is not None:
        authoritative_keys.update({"endpoint", "span.http.route"})
    return candidates, authoritative_keys


async def _validate_tempo_attributes(
    scope: IncidentScope,
    filters: dict[str, Any],
    collector: SourceCollector,
    traces: list[ToolTraceEntry],
) -> ValidatedSourceFilters:
    ignored: dict[str, str] = {}
    candidates, authoritative_keys = _tempo_scope_constraints(scope)
    scope_attribute_keys = set(scope.attributes)
    for key, value, from_scope in [
        *((key, value, True) for key, value in scope.attributes.items()),
        *((key, value, False) for key, value in filters.items()),
    ]:
        if key in authoritative_keys or (not from_scope and key in scope_attribute_keys):
            ignored[key] = "conflicts_with_scope"
            continue
        name = _traceql_attribute_name(key)
        if name is None or not isinstance(value, (str, int, float, bool)):
            ignored[key] = "invalid_attribute"
            continue
        candidates.append((key, name, value, from_scope))
    if not candidates:
        return ValidatedSourceFilters(source="trace", ignored_filters=ignored)
    values_tool = collector.tools.get("get-attribute-values")
    if values_tool is None:
        ignored.update({key: "attribute_tool_unavailable" for key, _, _, _ in candidates})
        return ValidatedSourceFilters(
            source="trace",
            ignored_filters=ignored,
        )

    verified: dict[str, Any] = {}
    verified_origins: set[str] = set()
    scope_attributes: set[str] = set()
    base_filter = _build_traceql_filter(scope, attributes={})
    for index, (key, name, value, from_scope) in enumerate(candidates):
        if key in verified_origins:
            continue
        if index >= 4:
            ignored[key] = "attribute_budget_exceeded"
            continue
        if not from_scope and name in scope_attributes:
            ignored[key] = "conflicts_with_scope"
            continue
        if from_scope:
            scope_attributes.add(name)
        raw_values = await _invoke_optional(
            collector,
            values_tool,
            "get-attribute-values",
            {"name": name, "filter-query": base_filter},
            traces,
        )
        if raw_values is None:
            ignored[key] = "attribute_lookup_failed"
            continue
        if str(value) in _tabular_values(raw_values):
            verified[name] = value
            verified_origins.add(key)
            ignored.pop(key, None)
        else:
            ignored[key] = "value_not_found"
    return ValidatedSourceFilters(
        source="trace",
        verified_filters=verified,
        ignored_filters=ignored,
        discovery={"attributes": list(verified)},
    )


async def _validate_loki_filters(
    scope: IncidentScope,
    filters: dict[str, Any],
    datasource_uid: str,
    labels: list[str],
    window: dict[str, str],
    collector: SourceCollector,
    label_values_tool,
    traces: list[ToolTraceEntry],
) -> ValidatedSourceFilters:
    scope_label_candidates = {
        # `component` is the service label emitted by this platform's log pipeline.
        "service_name": ("service_name", "component"),
        "endpoint": ("endpoint",),
        "status_code": ("status_code",),
        "trace_id": ("trace_id",),
    }
    candidates: list[tuple[str, Any, tuple[str, ...], bool]] = []
    for key in ("service_name", "endpoint", "status_code", "trace_id"):
        value = getattr(scope, key)
        if value is not None:
            candidates.append((key, value, scope_label_candidates[key], True))
    for key, value in filters.items():
        if value is not None:
            candidates.append((key, value, (key,), False))

    by_lower = {label.lower(): label for label in labels}
    verified: dict[str, Any] = {}
    ignored: dict[str, str] = {}
    checked: set[tuple[str, str]] = set()
    scope_labels: set[str] = set()
    for origin, value, candidate_labels, from_scope in candidates:
        label = next((by_lower[item.lower()] for item in candidate_labels if item.lower() in by_lower), None)
        if not label:
            ignored[origin] = "label_not_found"
            continue
        if not from_scope and label in scope_labels:
            ignored[origin] = "conflicts_with_scope"
            continue
        if from_scope:
            scope_labels.add(label)
        candidate_key = (label, str(value))
        if candidate_key in checked:
            continue
        checked.add(candidate_key)
        args = {
            "datasourceUid": datasource_uid,
            "labelName": label,
            **window,
        }
        values = await _invoke_optional(
            collector,
            label_values_tool,
            "list_loki_label_values",
            args,
            traces,
        )
        if values is None:
            ignored[origin] = "label_lookup_failed"
            continue
        if str(value) in _tabular_values(values):
            verified[label] = value
        else:
            ignored[origin] = "value_not_found"

    return ValidatedSourceFilters(
        source="log",
        verified_filters=verified,
        ignored_filters=ignored,
        discovery={"datasource_uid": datasource_uid, "labels": labels},
    )


async def _collect_metric(
    task: EvidenceTask,
    scope: IncidentScope,
    filters: dict[str, Any],
    collector: SourceCollector,
) -> EvidenceResult:
    traces: list[ToolTraceEntry] = []
    databases_tool = _required_tool(collector, "list_influxdb_databases")
    measurements_tool = _required_tool(collector, "list_measurements")
    schema_tool = _required_tool(collector, "get_measurement_schema")
    tag_values_tool = _required_tool(collector, "get_tag_values")
    query_tool = _required_tool(collector, "execute_influxql")

    raw_databases = await _invoke_traced(
        collector,
        databases_tool,
        "list_influxdb_databases",
        {},
        traces,
    )
    databases = _tabular_values(raw_databases)
    scope_database = scope.attributes.get("database_name")
    filter_database = filters.get("database_name")
    requested_database = scope_database if scope_database is not None else filter_database
    if scope_database is not None and str(scope_database) not in databases:
        return _evidence_with_traces("metric", "SKIPPED", "influx_database_not_found", traces)
    database_name = _choose_discovered(requested_database, databases, preferred="mc-observability")
    if not database_name:
        return _evidence_with_traces("metric", "SKIPPED", "influx_database_not_found", traces)

    raw_measurements = await _invoke_traced(
        collector,
        measurements_tool,
        "list_measurements",
        {"database_name": database_name},
        traces,
    )
    scope_measurement = scope.attributes.get("measurement")
    filter_measurement = filters.get("measurement")
    requested_measurement = scope_measurement if scope_measurement is not None else filter_measurement
    measurements = _bounded_measurements(raw_measurements, requested_measurement)
    if not measurements:
        return _evidence_with_traces("metric", "SKIPPED", "metric_measurement_not_found", traces)

    schemas, validations = await _discover_metric_schemas(
        measurements,
        scope,
        filters,
        database_name,
        collector,
        schema_tool,
        tag_values_tool,
        traces,
    )
    if not schemas:
        return _evidence_with_traces("metric", "SKIPPED", "metric_fields_not_found", traces)

    if scope_measurement is not None and str(scope_measurement) not in schemas:
        return _evidence_with_traces("metric", "SKIPPED", "metric_measurement_not_found", traces)
    requested_measurement_found = requested_measurement is not None and str(requested_measurement) in schemas
    requested_database_found = requested_database is not None and str(requested_database) == database_name
    eligible = _eligible_metric_schemas(
        schemas,
        validations,
        scope_measurement,
        requested_measurement,
        requested_measurement_found,
        requested_database_found,
    )
    if not eligible:
        evidence = _evidence_with_traces(
            "metric",
            "SKIPPED",
            "metric_scope_unverified",
            traces,
        )
        for validation in validations.values():
            _add_ignored_limitations(evidence, validation)
        return evidence

    validated = _metric_agent_context(
        database_name,
        eligible,
        validations,
        scope_database,
        filter_database,
        requested_database,
        requested_database_found,
        scope_measurement,
        filter_measurement,
        requested_measurement,
        requested_measurement_found,
    )
    budget = _QueryBudget(
        "query_verified_metric",
        CAPABILITY_SPECS[task.capability].get("max_queries", 1),
        traces,
    )
    allowed_refs: set[str] = set()

    async def query_verified_metric(
        measurement_name: str,
        field_name: str,
        aggregation: str = "mean",
    ) -> Any:
        if measurement_name not in eligible or field_name not in eligible[measurement_name][0]:
            return {"error": "unverified_metric_schema"}
        aggregation = aggregation.lower()
        if aggregation not in {"mean", "max", "min", "last"}:
            return {"error": "unsupported_aggregation"}
        key = _query_key(
            "query_verified_metric",
            {
                "measurement_name": measurement_name,
                "field_name": field_name,
                "aggregation": aggregation,
            },
        )
        if rejected := budget.blocked(
            key,
            {
                "measurement_name": measurement_name,
                "field_name": field_name,
                "aggregation": aggregation,
            },
        ):
            return rejected
        captured = []
        for window in _comparison_windows(scope).values():
            query = _build_influxql(
                measurement_name,
                field_name,
                validations[measurement_name].verified_filters,
                window["start"],
                window["end"],
                aggregation,
            )
            args = {"influxql_query": query}
            if _tool_accepts_argument(query_tool, "database_name"):
                args["database_name"] = database_name
            raw = await _invoke_traced(collector, query_tool, "execute_influxql", args, traces)
            captured.append(
                _capture_tool_result(
                    collector,
                    "metric",
                    "metrics.infrastructure",
                    "execute_influxql",
                    args,
                    raw,
                    allowed_refs,
                )
            )
        budget.record(key)
        result = collector.evidence_store.capture_discovery(
            source="metric",
            capability="metrics.infrastructure",
            value={"results": captured},
        )
        if isinstance(result, dict) and (
            reference := result.get("evidence_ref")
        ):
            allowed_refs.add(reference)
        return result

    safe_tool = StructuredTool.from_function(
        coroutine=query_verified_metric,
        name="query_verified_metric",
        description=(
            "Compare one discovered infrastructure field in the incident window against the immediately "
            "preceding baseline. Choose an exact measurement and field from discovery; aggregation must be "
            "mean, max, min, or last."
        ),
    )
    evidence = await _run_source_agent(
        task,
        scope,
        collector,
        [safe_tool, _inspection_tool(collector, allowed_refs)],
        validated,
        allowed_refs,
    )
    _finalize_agent_evidence(evidence, task.capability, collector.evidence_store)
    evidence.tool_trace = [*traces, *evidence.tool_trace]
    _add_ignored_limitations(evidence, validated)
    return evidence


def _bounded_measurements(raw: Any, requested: Any) -> list[str]:
    measurements = [
        measurement for measurement in _tabular_values(raw) if _is_identifier(measurement)
    ]
    if requested is not None and str(requested) in measurements:
        measurements.remove(str(requested))
        measurements.insert(0, str(requested))
    return measurements[:10]


async def _discover_metric_schemas(
    measurements: list[str],
    scope: IncidentScope,
    filters: dict[str, Any],
    database_name: str,
    collector: SourceCollector,
    schema_tool,
    tag_values_tool,
    traces: list[ToolTraceEntry],
) -> tuple[
    dict[str, tuple[list[str], list[str]]],
    dict[str, ValidatedSourceFilters],
]:
    schemas = {}
    validations = {}
    for measurement in measurements:
        # One unreadable measurement must not cost us the other nine.
        raw_schema = await _invoke_optional(
            collector,
            schema_tool,
            "get_measurement_schema",
            {"measurement_name": measurement, "database_name": database_name},
            traces,
        )
        fields, tags = _measurement_schema(raw_schema)
        if not fields:
            continue
        schemas[measurement] = (fields, tags)
        validations[measurement] = await _validate_metric_filters(
            scope,
            filters,
            database_name,
            measurement,
            tags,
            collector,
            tag_values_tool,
            traces,
        )
    return schemas, validations


def _eligible_metric_schemas(
    schemas: dict[str, tuple[list[str], list[str]]],
    validations: dict[str, ValidatedSourceFilters],
    scope_measurement: Any,
    requested_measurement: Any,
    requested_measurement_found: bool,
    requested_database_found: bool,
) -> dict[str, tuple[list[str], list[str]]]:
    if scope_measurement is not None:
        return {str(scope_measurement): schemas[str(scope_measurement)]}
    return {
        measurement: schema
        for measurement, schema in schemas.items()
        if validations[measurement].verified_filters
        or (requested_measurement_found and str(requested_measurement) == measurement)
        or requested_database_found
    }


def _metric_agent_context(
    database_name: str,
    eligible: dict[str, tuple[list[str], list[str]]],
    validations: dict[str, ValidatedSourceFilters],
    scope_database: Any,
    filter_database: Any,
    requested_database: Any,
    requested_database_found: bool,
    scope_measurement: Any,
    filter_measurement: Any,
    requested_measurement: Any,
    requested_measurement_found: bool,
) -> ValidatedSourceFilters:
    ignored_filters = {
        key: value
        for validation in validations.values()
        for key, value in validation.ignored_filters.items()
    }
    if (
        scope_database is not None
        and filter_database is not None
        and str(scope_database) != str(filter_database)
    ):
        ignored_filters["database_name"] = "conflicts_with_scope"
    if (
        scope_measurement is not None
        and filter_measurement is not None
        and str(scope_measurement) != str(filter_measurement)
    ):
        ignored_filters["measurement"] = "conflicts_with_scope"
    if requested_database and not requested_database_found:
        ignored_filters["database_name"] = "database_not_found"
    if requested_measurement and not requested_measurement_found:
        ignored_filters["measurement"] = "measurement_not_found"
    return ValidatedSourceFilters(
        source="metric",
        ignored_filters=ignored_filters,
        discovery={
            "database_name": database_name,
            "measurements": {
                measurement: {
                    "fields": fields,
                    "tags": tags,
                    "verified_filters": validations[measurement].verified_filters,
                }
                for measurement, (fields, tags) in eligible.items()
            },
            "aggregations": ["mean", "max", "min", "last"],
        },
    )


async def _validate_metric_filters(
    scope: IncidentScope,
    filters: dict[str, Any],
    database_name: str,
    measurement: str,
    tags: list[str],
    collector: SourceCollector,
    tag_values_tool,
    traces: list[ToolTraceEntry],
) -> ValidatedSourceFilters:
    candidates = [
        (key, value, True)
        for key, value in scope.attributes.items()
        if key not in {"database_name", "measurement"}
    ]
    if scope.service_name is not None and "service_name" not in scope.attributes:
        candidates.append(("service_name", scope.service_name, True))
    candidates.extend(
        (key, value, False)
        for key, value in filters.items()
        if key not in {"database_name", "measurement"}
    )

    by_lower = {tag.lower(): tag for tag in tags if _is_identifier(tag)}
    verified: dict[str, Any] = {}
    ignored: dict[str, str] = {}
    scope_tags: set[str] = set()
    for key, value, from_scope in candidates:
        tag = by_lower.get(key.lower())
        if not tag:
            ignored[key] = "tag_not_found"
            continue
        if not from_scope and tag in scope_tags:
            ignored[key] = "conflicts_with_scope"
            continue
        if from_scope:
            scope_tags.add(tag)
        raw_values = await _invoke_optional(
            collector,
            tag_values_tool,
            "get_tag_values",
            {
                "database_name": database_name,
                "measurement_name": measurement,
                "tag_key": tag,
            },
            traces,
        )
        if raw_values is None:
            ignored[key] = "tag_lookup_failed"
            continue
        if str(value) in _tabular_values(raw_values):
            verified[tag] = value
        else:
            ignored[key] = "value_not_found"

    return ValidatedSourceFilters(
        source="metric",
        verified_filters=verified,
        ignored_filters=ignored,
    )


def _is_empty_payload(value: Any) -> bool:
    """True when a tool answered successfully but carried no rows.

    Telemetry backends report an empty window with a success envelope, not an empty
    body: InfluxDB returns ``{"results": [{"statement_id": 0}]}`` with no ``series``
    and Loki returns ``{"data": {"result": []}}``. Both are non-empty dicts, so a
    plain falsiness check records the envelope itself as evidence — the synthesis LLM
    then receives a citable record containing nothing but a status field.
    """
    if value is None or value == "" or value == [] or value == {}:
        return True
    if isinstance(value, list):
        return all(_is_empty_payload(item) for item in value)
    if not isinstance(value, dict):
        return False
    # InfluxDB: a statement without "series" matched no points.
    if isinstance(results := value.get("results"), list):
        return not any(isinstance(item, dict) and item.get("series") for item in results)
    # Loki / Tempo: rows hang off one well-known key; recurse until we find them.
    for key in ("data", "result", "traces", "values"):
        if key in value:
            return _is_empty_payload(value[key])
    return False


def _capture_tool_result(
    collector: SourceCollector,
    source: str,
    capability: str,
    tool: str,
    args: dict[str, Any],
    raw: Any,
    allowed_refs: set[str],
) -> dict[str, Any]:
    if _is_empty_payload(raw):
        # Tell the agent the query ran and the window was empty, so it stops retrying
        # the same lookup expecting a different answer.
        return {"records": [], "truncated": False, "no_data": True}
    result = collector.evidence_store.capture(
        source=source,
        capability=capability,
        tool=tool,
        query=args,
        value=raw,
    )
    if reference := result.get("evidence_ref"):
        allowed_refs.add(reference)
    return result


def _inspection_tool(
    collector: SourceCollector,
    allowed_refs: set[str],
) -> StructuredTool:
    async def inspect_evidence(
        evidence_ref: str,
        path: str = "",
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        if evidence_ref not in allowed_refs:
            return {"error": "evidence_ref_not_available"}
        if len(path) > 2_048:
            return {"error": "invalid_path"}
        return collector.evidence_store.inspect(
            evidence_ref,
            path=path,
            offset=offset,
            limit=limit,
        )

    return StructuredTool.from_function(
        coroutine=inspect_evidence,
        name="inspect_evidence",
        description=(
            "Inspect a spilled query result by evidence_ref and RFC 6901 JSON Pointer path. "
            "Use offset and limit for bounded object, array, or string views."
        ),
    )


async def _run_source_agent(
    task: EvidenceTask,
    scope: IncidentScope,
    collector: SourceCollector,
    query_tools: list[StructuredTool],
    validated: ValidatedSourceFilters,
    allowed_refs: set[str],
) -> EvidenceResult:
    runner = collector.runner_factory(query_tools)
    if inspect.isawaitable(runner):
        runner = await runner
    payload = {
        "messages": [
            {
                "role": "user",
                "content": _validated_task_prompt(
                    task,
                    scope,
                    validated,
                    [tool.name for tool in query_tools],
                    collector.evidence_store,
                    allowed_refs,
                ),
            }
        ]
    }
    raw = await _invoke(runner, payload)
    return _extract_evidence_result(raw, CAPABILITY_SPECS[task.capability]["source"])


async def _invoke(runner, payload: dict[str, Any]) -> Any:
    if hasattr(runner, "ainvoke"):
        return await runner.ainvoke(payload)
    result = runner(payload)
    if inspect.isawaitable(result):
        return await result
    return result


async def _invoke_traced(
    collector: SourceCollector,
    tool,
    name: str,
    args: dict[str, Any],
    traces: list[ToolTraceEntry],
) -> Any:
    started = time.perf_counter()
    try:
        output = _unwrap_mcp_output(await _invoke(tool, args))
        traces.append(
            ToolTraceEntry(
                tool=name,
                args=args,
                output_preview=_preview(output),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        )
        return output
    except Exception as exc:
        message = str(exc)
        traces.append(
            ToolTraceEntry(
                tool=name,
                args=args,
                error=message,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        )
        raise _ToolInvocationError(message, traces) from exc


async def _invoke_optional(
    collector: SourceCollector,
    tool,
    name: str,
    args: dict[str, Any],
    traces: list[ToolTraceEntry],
) -> Any:
    """Invoke a discovery/validation lookup that must not abort the capability.

    These calls only decide whether a requested filter is real. Letting one of them
    raise would kill a collection whose main query was still perfectly runnable, so
    the failure degrades to "unverified" instead. The error is already recorded in
    ``traces`` by _invoke_traced, so nothing is hidden.
    """
    try:
        return await _invoke_traced(collector, tool, name, args, traces)
    except _ToolInvocationError:
        return None


class _QueryBudget:
    """Repeat/volume guard for the agent-facing query tools.

    The agent gets one safe query tool per capability; without this it will happily
    re-issue the same query or keep widening until the tool-call limit trips. Errors
    are returned to the model (not raised) so it can redirect itself.
    """

    def __init__(self, tool_name: str, limit: int, traces: list[ToolTraceEntry]):
        self._tool_name = tool_name
        self._limit = limit
        self._traces = traces
        self._seen: set[str] = set()

    def blocked(self, key: str, args: dict[str, Any]) -> dict[str, str] | None:
        if key in self._seen:
            return self._reject("duplicate_tool_call_blocked", args)
        if len(self._seen) >= self._limit:
            return self._reject("query_budget_exhausted", args)
        return None

    def record(self, key: str) -> None:
        self._seen.add(key)

    def _reject(self, reason: str, args: dict[str, Any]) -> dict[str, str]:
        self._traces.append(ToolTraceEntry(tool=self._tool_name, args=args, error=reason))
        return {"error": reason}


def _extract_evidence_result(result: Any, source: str) -> EvidenceResult:
    evidence = _parse_evidence_result(result, source)
    if evidence is None:
        return _stub(source, "FAILED", "structured_evidence_missing")
    # The capability decides the source, never the model. A wrong literal here
    # detaches the result from its own records downstream — the merged view keeps
    # the model's source while the catalog keeps the real one, and the validator
    # then drops every record as unusable even though collection succeeded.
    if evidence.source != source:
        evidence = evidence.model_copy(update={"source": source})
    return evidence


def _parse_evidence_result(result: Any, source: str) -> EvidenceResult | None:
    if isinstance(result, EvidenceResult):
        return result
    if isinstance(result, BaseModel):
        result = result.model_dump()
    if isinstance(result, dict):
        structured = result.get("structured_response", result)
        if isinstance(structured, EvidenceResult):
            return structured
        if isinstance(structured, BaseModel):
            structured = structured.model_dump()
        if isinstance(structured, dict):
            # source is a fallback for models that omit it; the caller pins the real one.
            return EvidenceResult.model_validate({"source": source, **structured})
    return None


def _validated_task_prompt(
    task: EvidenceTask,
    scope: IncidentScope,
    validated: ValidatedSourceFilters,
    tool_names: list[str],
    evidence_store: EvidenceStore,
    allowed_refs: set[str],
) -> str:
    discovery = evidence_store.capture_discovery(
        source=validated.source,
        capability=task.capability,
        value=validated.discovery,
    )
    if isinstance(discovery, dict) and (
        reference := discovery.get("evidence_ref")
    ):
        allowed_refs.add(reference)
    body = {
        "focus": task.focus,
        "authoritative_time_range": scope.time_range.model_dump(mode="json"),
        "verified_filters": validated.verified_filters,
        "discovery": discovery,
        "available_tools": tool_names,
    }
    return (
        "Use only the supplied tools and do not invent labels, measurements, fields, or time ranges. "
        "When a query returns inspection_required, inspect only the referenced paths needed for this task. "
        f"\n\nTask:\n{json.dumps(body, ensure_ascii=False, default=str)}"
    )


def _build_log_selector(labels: list[str], verified: dict[str, Any]) -> str:
    if verified:
        terms = [f'{label}="{_escape_string(value)}"' for label, value in sorted(verified.items())]
        return "{" + ",".join(terms) + "}"
    if labels:
        return "{" + labels[0] + '=~".+"}'
    return "{}"


def _build_influxql(
    measurement: str,
    field: str,
    verified: dict[str, Any],
    start: str,
    end: str,
    aggregation: str,
) -> str:
    clauses = [f"time >= '{start}'", f"time <= '{end}'"]
    clauses.extend(
        f"\"{_escape_identifier(key)}\" = '{_escape_influx_value(value)}'" for key, value in sorted(verified.items())
    )
    return (
        f'SELECT {aggregation.upper()}("{_escape_identifier(field)}") '
        f'FROM "{_escape_identifier(measurement)}" '
        f"WHERE {' AND '.join(clauses)} LIMIT {CAPABILITY_SPECS['metrics.infrastructure']['max_rows']}"
    )


def _safe_log_terms(terms: list[str] | None) -> list[str]:
    safe = []
    for term in (terms or [])[:8]:
        normalized = str(term).strip()
        if normalized and len(normalized) <= 80 and normalized not in safe:
            safe.append(normalized)
    return safe


async def _direct_evidence_result(
    task: EvidenceTask,
    scope: IncidentScope,
    collector: SourceCollector,
    *,
    source: str,
    tool: str,
    args: dict[str, Any],
    raw: Any,
    traces: list[ToolTraceEntry],
    summary: str | None = None,
    validated: ValidatedSourceFilters | None = None,
) -> EvidenceResult:
    allowed_refs: set[str] = set()
    captured = _capture_tool_result(
        collector,
        source,
        task.capability,
        tool,
        args,
        raw,
        allowed_refs,
    )
    evidence = EvidenceResult(
        source=source,
        capability=task.capability,
        status="NO_DATA" if _is_empty_payload(raw) else "OK",
        summary=summary or "Raw evidence collected.",
    )
    if captured.get("inspection_required") and collector.runner_factory is not None:
        context = (
            validated.model_copy(deep=True)
            if validated is not None
            else ValidatedSourceFilters(source=source)
        )
        context.discovery["spilled_evidence"] = captured
        evidence = await _run_source_agent(
            task,
            scope,
            collector,
            [_inspection_tool(collector, allowed_refs)],
            context,
            allowed_refs,
        )
    _finalize_agent_evidence(
        evidence,
        task.capability,
        collector.evidence_store,
    )
    evidence.tool_trace = [*traces, *evidence.tool_trace]
    return evidence


def _finalize_agent_evidence(
    evidence: EvidenceResult,
    capability: str,
    store: EvidenceStore,
) -> None:
    evidence.capability = capability
    evidence.records = store.records_for(capability)
    uninspected = store.has_uninspected(capability)
    budget_exhausted = store.budget_exhausted(capability)
    unavailable = store.has_unavailable(capability)
    if uninspected:
        _add_limitation(evidence, "spilled_evidence_not_inspected")
    if budget_exhausted:
        evidence.truncated = True
        _add_limitation(evidence, "evidence_budget_exhausted")
    if unavailable:
        evidence.truncated = True
        _add_limitation(evidence, "evidence_unavailable")
    if evidence.records:
        evidence.status = (
            "PARTIAL"
            if uninspected or budget_exhausted or unavailable
            else "OK"
        )
        return
    if budget_exhausted or unavailable:
        evidence.status = "FAILED"
        return
    if uninspected:
        evidence.status = "PARTIAL"
        return
    if evidence.status == "NO_DATA":
        # The query ran and the window was empty. That is an observation, not a gap.
        return
    evidence.status = "PARTIAL" if evidence.status == "OK" else evidence.status
    _add_limitation(evidence, "raw_evidence_missing")


def _tempo_window(scope: IncidentScope) -> dict[str, str]:
    window = _time_window(scope)
    return {"start": window["startRfc3339"], "end": window["endRfc3339"]}


def _comparison_windows(scope: IncidentScope) -> dict[str, dict[str, str]]:
    start = scope.time_range.start
    end = scope.time_range.end
    if start is None or end is None:
        raise ValueError("time_range_missing")
    baseline_start = start - (end - start)
    return {
        "baseline": {"start": _rfc3339(baseline_start), "end": _rfc3339(start)},
        "incident": {"start": _rfc3339(start), "end": _rfc3339(end)},
    }


def _build_traceql_filter(
    scope: IncidentScope,
    *,
    attributes: dict[str, Any] | None = None,
    error_only: bool = False,
    min_duration_ms: int | None = None,
) -> str:
    clauses = []
    if scope.service_name:
        clauses.append(f'resource.service.name = "{_escape_traceql(scope.service_name)}"')
    if scope.endpoint:
        clauses.append(f'span.http.route = "{_escape_traceql(scope.endpoint)}"')
    for key, value in sorted((attributes or {}).items()):
        attribute = _traceql_attribute_name(key)
        if attribute is not None:
            clauses.append(f'{attribute} = "{_escape_traceql(value)}"')
    if error_only:
        clauses.append("status = error")
    if min_duration_ms is not None:
        if not 1 <= min_duration_ms <= 300_000:
            raise ValueError("invalid_min_duration_ms")
        clauses.append(f"duration >= {min_duration_ms}ms")
    return "{ " + " && ".join(clauses or ["true"]) + " }"


def _escape_traceql(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _traceql_attribute_name(value: Any) -> str | None:
    name = str(value)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
        return None
    return name if name.startswith(("resource.", "span.")) else f"span.{name}"


def _trace_ids_from_records(records: list[EvidenceRecord]) -> list[str]:
    trace_ids = []
    for record in records:
        try:
            value = json.loads(record.observation)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in _walk_dicts(value):
            trace_id = item.get("traceId") or item.get("trace_id")
            if trace_id is not None:
                trace_ids.append(str(trace_id))
    return list(dict.fromkeys(trace_ids))


def _tool_accepts_argument(tool: Any, argument: str) -> bool:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return True
    fields = getattr(args_schema, "model_fields", None)
    if fields is not None:
        return argument in fields
    schema = args_schema.model_json_schema() if hasattr(args_schema, "model_json_schema") else {}
    return argument in schema.get("properties", {})


def _time_window(scope: IncidentScope) -> dict[str, str]:
    if scope.time_range.start is None or scope.time_range.end is None:
        raise ValueError("time_range_missing")
    return {
        "startRfc3339": _rfc3339(scope.time_range.start),
        "endRfc3339": _rfc3339(scope.time_range.end),
    }


def _rfc3339(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _find_loki_datasource_uid(value: Any) -> str | None:
    fallback = None
    for item in _walk_dicts(value):
        uid = item.get("uid") or item.get("datasourceUid")
        if not uid:
            continue
        fallback = fallback or str(uid)
        if "loki" in str(item.get("type", "")).lower():
            return str(uid)
    return fallback


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def _tabular_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("tag_values", "values", "series", "results", "data"):
            if key in value:
                found = _tabular_values(value[key])
                if found:
                    return found
        return []
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return [str(item) for item in value]
        if all(isinstance(item, list) and item for item in value):
            return [str(item[0]) for item in value]
        for item in value:
            found = _tabular_values(item)
            if found:
                return found
    return []


def _measurement_schema(value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, dict):
        return [], []
    fields = [str(item[0]) for item in value.get("fields", []) if isinstance(item, list) and item]
    tags = [str(item[0]) for item in value.get("tags", []) if isinstance(item, list) and item]
    return [item for item in fields if _is_identifier(item)], [item for item in tags if _is_identifier(item)]


def _choose_discovered(requested: Any, discovered: list[str], preferred: str | None = None) -> str | None:
    if requested is not None and str(requested) in discovered:
        return str(requested)
    if preferred and preferred in discovered:
        return preferred
    return discovered[0] if discovered else None


def _unwrap_mcp_output(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, dict) and "content" in value:
        return _unwrap_mcp_output(value["content"])
    if isinstance(value, list) and value and all(isinstance(item, dict) and "text" in item for item in value):
        parsed = [_unwrap_mcp_output(item["text"]) for item in value]
        return parsed[0] if len(parsed) == 1 else parsed
    if isinstance(value, str):
        try:
            return _unwrap_mcp_output(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _required_tool(collector: SourceCollector, name: str):
    tool = collector.tools.get(name)
    if tool is None:
        raise RuntimeError(f"collector_tool_missing:{name}")
    return tool


def _query_key(tool: str, args: dict[str, Any]) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"


def _is_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value))


def _escape_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)[1:-1]


def _escape_identifier(value: Any) -> str:
    return str(value).replace('"', '\\"')


def _escape_influx_value(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _add_ignored_limitations(evidence: EvidenceResult, validated: ValidatedSourceFilters) -> None:
    for key in validated.ignored_filters:
        _add_limitation(evidence, f"ignored_unverified_filter:{key}")


def _evidence_with_traces(
    source: str,
    status: str,
    reason: str,
    traces: list[ToolTraceEntry],
) -> EvidenceResult:
    evidence = _stub(source, status, reason)
    evidence.tool_trace = traces
    return evidence


def _tool_trace(tool: str, args: dict[str, Any], output: Any) -> ToolTraceEntry:
    return ToolTraceEntry(tool=tool, args=args, output_preview=_preview(output))


def _preview(value: Any, limit: int = 1000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit]


def _stub(source: str, status: str, reason: str, capability: str = "") -> EvidenceResult:
    return EvidenceResult(source=source, capability=capability, status=status, limitations=[reason])


def _add_limitation(evidence: EvidenceResult, reason: str) -> None:
    if reason not in evidence.limitations:
        evidence.limitations.append(reason)
