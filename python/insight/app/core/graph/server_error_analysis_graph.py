"""ReAct evidence-collection graph for observability RCA (server-error-analysis).

Topology (request-plan gated ReAct):

    request_plan -> finalize | direct_answer | clarification | agent <-> tools -> synthesize -> validate -> finalize

This file intentionally owns the server-error-analysis graph end to end. The bulky MCP/query
collector plumbing lives in ``app.core.graph.utils.collector``.
"""

import json
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

# --- Schemas ---------------------------------------------------------

EvidenceSource = Literal["loki", "tempo", "influxdb"]
EvidenceStatus = Literal["OK", "NO_DATA", "PARTIAL", "FAILED", "SKIPPED"]
RcaConfidence = Literal["critical", "high", "medium", "low"]
RequestMode = Literal["direct_answer", "needs_clarification", "evidence_rca"]


class IncidentContext(BaseModel):
    """Normalized incident scope shared by planning, collection, and synthesis."""

    session_id: str
    analysis_id: int | None = None
    query: str | None = None
    trace_id: str | None = None
    service_name: str | None = None
    node_id: str | None = None
    infra_id: str | None = None
    level: str | None = None
    message_filter: str | None = None
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    max_evidence_per_source: int | None = None


class SourceSpec(BaseModel):
    """Planner-selected collection intent for one observability source."""

    enabled: bool = False
    keywords: list[str] = Field(default_factory=list)
    profiles: list[str] = Field(default_factory=list)


class EvidenceQueryPlan(BaseModel):
    """Source-level evidence collection plan produced by the request planner."""

    loki: SourceSpec = Field(default_factory=SourceSpec)
    tempo: SourceSpec = Field(default_factory=SourceSpec)
    influxdb: SourceSpec = Field(default_factory=SourceSpec)


class RequestPlan(BaseModel):
    """Top-level request route: answer directly, ask for scope, or collect evidence."""

    mode: RequestMode
    reason: str = ""
    answer_hint: str | None = None
    evidence: EvidenceQueryPlan | None = None


class EvidencePlan(BaseModel):
    """Resolved list of sources that must reach a terminal evidence state."""

    required_sources: list[EvidenceSource] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    """Stable reference id used to cite collected evidence in the RCA result."""

    id: str
    source: EvidenceSource
    summary: str
    tool_name: str | None = None
    query: str | None = None


class SourceEvidence(BaseModel):
    """Normalized evidence returned by one source collector."""

    source: EvidenceSource
    status: EvidenceStatus
    summary: str | None = None
    executed_query: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    refs: list[EvidenceRef] = Field(default_factory=list)
    raw_evidence: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    error: str | None = None


class RcaResult(BaseModel):
    """LLM-produced RCA report after evidence collection."""

    status: Literal["COMPLETED", "INCONCLUSIVE", "NEEDS_CLARIFICATION", "FAILED"] = "INCONCLUSIVE"
    summary: str
    root_cause: str | None = None
    confidence: RcaConfidence = "low"
    confirmed_facts: list[str] = Field(default_factory=list)
    ruled_out: list[str] = Field(default_factory=list)
    unverified_hypotheses: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Deterministic validation outcome for evidence grounding."""

    status: Literal["PASSED", "PARTIAL", "FAILED"] = "PASSED"
    findings: list[str] = Field(default_factory=list)


class RequestPlanningResult(BaseModel):
    """Structured wrapper returned by the request-planning LLM."""

    request_plan: RequestPlan


# --- Detail ----------------------------------------------------------

DETAIL_SCHEMA_VERSION = "server_error_analysis"


def build_rca_detail(
    *,
    request: dict[str, Any],
    incident: IncidentContext,
    evidence_plan,
    evidence: dict[str, SourceEvidence],
    result: RcaResult,
    validation: ValidationResult,
    request_plan: RequestPlan | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the persisted detail envelope for API responses and record storage."""

    error_list = list(errors or [])
    if result.status == "FAILED" and not error_list:
        error_list.append({"type": "analysis_failed", "message": "; ".join(validation.findings) or "RCA failed"})
    detail = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "request": request,
        "evidence_plan": evidence_plan.model_dump(mode="json"),
        "request_plan": request_plan.model_dump(mode="json") if request_plan else {},
        "log_evidence": source_evidence_detail(evidence.get("loki"), "loki", incident),
        "trace_evidence": source_evidence_detail(evidence.get("tempo"), "tempo", incident),
        "metric_evidence": source_evidence_detail(evidence.get("influxdb"), "influxdb", incident),
        "evidence_bundle": {
            "refs": [ref.model_dump(mode="json") for item in evidence.values() for ref in item.refs],
            "limitations": [limitation for item in evidence.values() for limitation in item.limitations],
        },
        "result": result.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "errors": error_list,
    }
    return json.loads(json.dumps(detail, default=str))


def normalize_server_error_detail(detail: dict | None, trace_id: str | None = None) -> dict[str, Any]:
    """Normalize stored detail JSON into the current public response shape."""

    detail = detail or {}
    out = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        **{
            key: detail[key]
            for key in (
                "request",
                "evidence_plan",
                "request_plan",
                "log_evidence",
                "trace_evidence",
                "metric_evidence",
                "evidence_bundle",
                "result",
                "validation",
                "errors",
                "error",
                "trajectory",
            )
            if key in detail
        },
    }

    result = dict(out.get("result") or {})
    result.setdefault("status", "INCONCLUSIVE")
    result.setdefault("confidence", "low")
    result.setdefault("root_cause", None)
    result.setdefault("next_checks", [])
    result.setdefault("limitations", [])
    result.setdefault("confirmed_facts", [])
    result.setdefault("ruled_out", [])
    result.setdefault("unverified_hypotheses", [])
    result.setdefault("evidence_refs", [])
    out["result"] = result

    if "error" not in out:
        errors = out.get("errors") if isinstance(out.get("errors"), list) else []
        out["error"] = errors[0] if errors else None
    return out


def source_evidence_detail(
    evidence: SourceEvidence | None,
    source: str,
    incident: IncidentContext,
) -> dict[str, Any]:
    """Render one source evidence object into the persisted detail format."""

    if evidence is None:
        evidence = SourceEvidence(source=source, status="SKIPPED")
    return {
        "source": evidence.source,
        "status": evidence.status,
        "time_range": {
            "start": incident.time_range_start.isoformat() if incident.time_range_start else None,
            "end": incident.time_range_end.isoformat() if incident.time_range_end else None,
        },
        "query_summary": evidence.summary,
        "executed_query": evidence.executed_query,
        "filters_applied": evidence.filters_applied,
        "observations": [
            item if isinstance(item, dict) else {"message": str(item), "attributes": {}}
            for item in evidence.observations
        ],
        "refs": [ref.model_dump(mode="json") for ref in evidence.refs],
        "limitations": evidence.limitations,
        "error": evidence.error,
    }


# --- Request planning -----------------------------------------------

REQUEST_PLANNER_PROMPT = """You create a structured observability RCA request plan.
Return only the requested schema.

Modes:
- needs_clarification: the user asks for RCA, but the request lacks enough incident scope to start safe evidence collection.
  Use this when there is no affected service, node, infra, trace id, time range, error message, metric, status code, or concrete symptom.
- direct_answer: the user asks how the feature/workflow works and does not ask for incident RCA.
- evidence_rca: any production symptom, error, latency, resource, dependency, availability, trace, metric, or log investigation.

Evidence rules:
- Enable loki for evidence_rca.
- Enable tempo when trace_id exists, traces are requested, logs may reveal trace_id, or a clear service/node/infra scope can be searched.
- Enable influxdb for latency, resource, saturation, error-rate, trend, impact, or blast-radius questions.
- Treat service_name, node_id, infra_id, level, and message as filters only when structured filters or the user request clearly provide them.
- Do not infer arbitrary words as service names or filter values; leave unclear fields empty.
- Keywords are short observable terms, not filler words.
- Profiles must be chosen from: latency, error_rate, cpu, memory.

Broad health-check rule:
- A broad health-check request asks whether anything is wrong without a concrete service_name, trace_id, node_id, infra_id, error message, status code, metric, or technical symptom.
- Examples: "문제가 있는지 봐줘", "장애가 있는지 확인해줘", "이상 있는지 확인해줘", "서비스에 문제 없는지 봐줘".
- If a broad health-check request has no concrete scope, return needs_clarification.
- If it includes a concrete service/node/infra/trace/time/error scope, return evidence_rca.
- For scoped broad health-check requests, enable loki and set loki.keywords to ["error", "exception", "failed", "timeout"].
- Do not use generic user words as Loki keywords, including "문제", "장애", "이상", "확인", "봐줘", "분석", "problem", "issue", "check", "analyze".
- For scoped broad health-check requests, enable tempo only when trace_id or a clear searchable service/node/infra scope exists.
- For scoped broad health-check requests, enable influxdb only when the user asks for metrics, latency, resource saturation, trend, impact, or error rate.

Never write LogQL, TraceQL, or InfluxQL."""


class LlmRequestPlanner:
    """LLM-backed planner that chooses request mode and evidence sources."""

    def __init__(self, chat_model):
        """Wrap the chat model with the request-planning output schema."""

        self.model = chat_model.with_structured_output(RequestPlanningResult)

    async def plan(self, incident: IncidentContext, options: dict[str, Any] | None = None) -> RequestPlanningResult:
        """Ask the LLM for a structured request plan."""

        result = await self.model.ainvoke(
            [
                {"role": "system", "content": REQUEST_PLANNER_PROMPT},
                {"role": "user", "content": _planner_payload(incident, options)},
            ]
        )
        if isinstance(result, RequestPlanningResult):
            return result
        return RequestPlanningResult.model_validate(result)


def _planner_payload(incident: IncidentContext, options: dict[str, Any] | None) -> str:
    """Build the JSON payload sent to the request planner."""

    payload = {
        "incident": incident.model_dump(mode="json"),
        "options": options or {},
        "examples": [
            {"query": "server error analysis 구조 설명", "mode": "direct_answer"},
            {
                "query": "문제가 있는지 봐줘",
                "mode": "needs_clarification",
            },
            {
                "query": "payment-api 문제가 있는지 봐줘",
                "mode": "evidence_rca",
                "service_name": "payment-api",
                "loki_keywords": ["error", "exception", "failed", "timeout"],
                "tempo_enabled": True,
                "influxdb_enabled": False,
            },
            {"query": "payment-api 500 원인 분석", "mode": "evidence_rca", "profiles": ["error_rate"]},
            {"query": "checkout timeout 느림", "mode": "evidence_rca", "profiles": ["latency"]},
            {"query": "CPU 높고 OOM 발생", "mode": "evidence_rca", "profiles": ["cpu", "memory"]},
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


# --- Agent tools -----------------------------------------------------

# Cap the LLM-chosen evidence window so a runaway range can't trigger a full-history scan.
_MAX_WINDOW = timedelta(hours=48)


def _parse_dt(value: str | None) -> datetime | None:
    """Parse ISO datetime strings used by tool-provided time overrides."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _incident_with_overrides(
    incident_data: Any,
    *,
    start: str | None = None,
    end: str | None = None,
    max_evidence_per_source: int | None = None,
) -> IncidentContext:
    """Return the base incident with only safe runtime bounds applied."""
    base = (
        incident_data
        if isinstance(incident_data, IncidentContext)
        else IncidentContext.model_validate(incident_data or {})
    )
    updates: dict[str, Any] = {}
    if max_evidence_per_source is not None:
        updates["max_evidence_per_source"] = max_evidence_per_source
    if (parsed := _parse_dt(start)) is not None:
        updates["time_range_start"] = parsed
    if (parsed := _parse_dt(end)) is not None:
        updates["time_range_end"] = parsed
    start_dt = updates.get("time_range_start", base.time_range_start)
    end_dt = updates.get("time_range_end", base.time_range_end)
    if start_dt and end_dt and (end_dt - start_dt) > _MAX_WINDOW:
        updates["time_range_start"] = end_dt - _MAX_WINDOW
    return base.model_copy(update=updates)


def _max_evidence(state: dict | None) -> int | None:
    """Read the request-level evidence limit from graph state."""

    value = ((state or {}).get("options") or {}).get("max_evidence_per_source")
    return value if isinstance(value, int) else None


def evidence_view(evidence: SourceEvidence) -> str:
    """Compact human/LLM-readable rendering of one source's evidence."""
    summary = (evidence.summary or "").strip()
    if len(summary) > 1500:
        summary = summary[:1500] + "…(truncated)"
    parts = [f"source={evidence.source}", f"status={evidence.status}", f"refs={len(evidence.refs)}"]
    if evidence.executed_query:
        parts.append(f"query={evidence.executed_query}")
    if evidence.limitations:
        parts.append("limitations=" + "; ".join(evidence.limitations))
    head = " ".join(parts)
    return f"{head}\n{summary}" if summary else head


async def safe_collect(
    collectors: dict[str, Any] | None,
    source: str,
    incident: IncidentContext,
    **kwargs: Any,
) -> SourceEvidence:
    """Run a collector, never raising: missing collector or any error becomes FAILED evidence."""
    collector = (collectors or {}).get(source)
    if collector is None:
        return SourceEvidence(source=source, status="FAILED", limitations=[f"{source} collector not configured"])
    try:
        return await collector.collect(incident, **kwargs)
    except Exception as exc:
        return SourceEvidence(
            source=source, status="FAILED", limitations=[f"{source} collector error: {exc}"], error=str(exc)
        )


@tool
async def collect_logs(
    logql: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> str:
    """Declare the Loki LogQL tool schema; the graph tools node performs execution.

    The query must start with a non-empty stream selector.
    Use explicit filters only; for broad checks prefer {service=~".+"} with OR keywords."""
    raise RuntimeError("collect_logs is executed by the graph tools node")


@tool
async def collect_traces(
    trace_id: str | None = None,
    traceql: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> str:
    """Declare the Tempo lookup tool schema; the graph tools node performs execution.

    Use trace_id for get-trace, or traceql for traceql-search.
    Provide only one of trace_id or traceql."""
    raise RuntimeError("collect_traces is executed by the graph tools node")


@tool
async def collect_metrics(
    influxql: str,
) -> str:
    """Declare the InfluxQL tool schema; the graph tools node performs execution."""

    raise RuntimeError("collect_metrics is executed by the graph tools node")


AGENT_TOOLS = [collect_logs, collect_traces, collect_metrics]


# --- Synthesis -------------------------------------------------------

SYSTEM_PROMPT = """You are an observability RCA analyst for MC-Observability.
Use only the supplied evidence bundle.
Do not invent a root cause or cite evidence that is not present.
Do not treat observed HTTP errors or status codes as a root cause; they are only symptoms.
Temporal correlation is not causation.
Metric evidence can support impact or blast radius, but should not be the only high-confidence cause.
Return root_cause only when specific log, trace, or metric evidence supports it.
Use confidence levels as evidence strength, not business impact: critical only when multiple independent evidence sources directly support the same causal mechanism and timing; high for strong cited evidence; medium for partial but coherent evidence; low for weak, missing, failed, or uncited evidence.
For broad health-check requests, return COMPLETED when collected evidence answers whether anomalies exist; root_cause may be null.
If evidence is missing, weak, failed, or only NO_DATA, return INCONCLUSIVE with clear limitations and next_checks."""


class LangChainStructuredLLM:
    """Adapter that asks LangChain for a structured RCA result."""

    def __init__(self, chat_model):
        """Wrap the chat model with the RCA result schema."""

        self.structured_model = chat_model.with_structured_output(RcaResult)

    async def generate_report(self, system_prompt: str, user_prompt: str) -> RcaResult:
        """Generate one structured RCA report from prompts."""

        result = await self.structured_model.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if isinstance(result, RcaResult):
            return result
        return RcaResult.model_validate(result)


class RcaSynthesizer:
    """Converts collected evidence into the final RCA report."""

    def __init__(self, llm, *, system_prompt: str = SYSTEM_PROMPT):
        """Store the structured LLM adapter and synthesis policy prompt."""

        self.llm = llm
        self.system_prompt = system_prompt

    async def synthesize(
        self, incident: IncidentContext, plan: EvidencePlan, evidence: dict[str, SourceEvidence]
    ) -> RcaResult:
        """Run RCA synthesis against the normalized incident and evidence bundle."""

        return await self.llm.generate_report(self.system_prompt, self._build_user_prompt(incident, plan, evidence))

    @staticmethod
    def _build_user_prompt(incident: IncidentContext, plan: EvidencePlan, evidence: dict[str, SourceEvidence]) -> str:
        """Build the synthesis prompt payload with evidence and citation rules."""

        refs = [ref for item in evidence.values() for ref in item.refs]
        limitations = [limitation for item in evidence.values() for limitation in item.limitations]
        payload = {
            "incident": incident.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "evidence": {
                "trace": evidence["tempo"].model_dump(mode="json") if "tempo" in evidence else None,
                "log": evidence["loki"].model_dump(mode="json") if "loki" in evidence else None,
                "metric": evidence["influxdb"].model_dump(mode="json") if "influxdb" in evidence else None,
                "refs": [ref.model_dump(mode="json") for ref in refs],
                "limitations": limitations,
            },
            "instructions": [
                "Return status as COMPLETED, INCONCLUSIVE, NEEDS_CLARIFICATION, or FAILED.",
                "Set confidence to one of critical, high, medium, or low using the confidence levels from the system prompt.",
                "Keep summary short: what was checked, what was found, and confidence.",
                "Set root_cause to null unless evidence_refs directly support it.",
                "Cite only existing evidence ref ids in evidence_refs.",
                "Put facts in confirmed_facts, guesses in unverified_hypotheses, gaps in limitations, actions in next_checks.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


# --- Validation ------------------------------------------------------


class RcaValidator:
    """Deterministic evidence-integrity check. Confidence is owned by the LLM (synthesizer);
    this validator never scores or caps confidence. It only enforces grounding: a root cause
    must cite real collected evidence, all-failed collection is a failure, and NO_DATA cannot
    rule causes out. When it must strip an ungrounded root cause it sets confidence to low,
    because a result with no cause cannot be confident."""

    def validate(
        self,
        incident: IncidentContext,
        plan: EvidencePlan,
        evidence: dict[str, SourceEvidence],
        result: RcaResult,
    ) -> ValidationResult:
        """Check that the report only claims causes supported by collected evidence."""

        required = [(item.status if (item := evidence.get(source)) else "SKIPPED") for source in plan.required_sources]
        if required and all(status == "FAILED" for status in required):
            finding = "all required evidence sources failed"
            result.status = "FAILED"
            result.root_cause = None
            result.confidence = "low"
            if finding not in result.limitations:
                result.limitations.append(finding)
            return ValidationResult(status="FAILED", findings=[finding])

        findings: list[str] = []
        valid_ref_ids = {ref.id for item in evidence.values() for ref in item.refs}
        unknown_refs = [ref_id for ref_id in result.evidence_refs if ref_id not in valid_ref_ids]
        if result.root_cause and not result.evidence_refs:
            findings.append("root_cause lacks evidence_refs")
            result.status = "INCONCLUSIVE"
            result.root_cause = None
            result.confidence = "low"
            result.limitations.extend(finding for finding in findings if finding not in result.limitations)
            return ValidationResult(status="PARTIAL", findings=findings)
        if result.root_cause and unknown_refs:
            findings.append(f"root_cause cites unknown evidence_refs: {', '.join(unknown_refs)}")
            result.evidence_refs = [ref_id for ref_id in result.evidence_refs if ref_id in valid_ref_ids]
            result.status = "INCONCLUSIVE"
            result.root_cause = None
            result.confidence = "low"
            result.limitations.extend(finding for finding in findings if finding not in result.limitations)
            return ValidationResult(status="PARTIAL", findings=findings)

        statuses = [item.status for item in evidence.values() if item.status != "SKIPPED"]
        if statuses and all(status == "NO_DATA" for status in statuses) and result.ruled_out:
            findings.append("NO_DATA evidence cannot rule out causes")
            result.ruled_out = []

        # Grounding is intact: keep the LLM's confidence verbatim.
        status = "PARTIAL" if result.status in {"INCONCLUSIVE", "NEEDS_CLARIFICATION"} else "PASSED"
        result.limitations.extend(finding for finding in findings if finding not in result.limitations)
        return ValidationResult(status=status, findings=findings)


# --- Orchestrator ----------------------------------------------------


class ServerErrorRcaOrchestrator:
    """Runtime dependencies for server-error RCA graph execution."""

    def __init__(
        self,
        *,
        collectors: dict[str, Any] | None = None,
        synthesizer: RcaSynthesizer | None = None,
        validator: RcaValidator | None = None,
        request_planner: Any = None,
        chat_model: Any = None,
    ):
        """Store collectors, planner, synthesizer, validator, and chat model."""

        self.collectors = collectors or {}
        self.synthesizer = synthesizer
        self.validator = validator or RcaValidator()
        self.request_planner = request_planner
        self.chat_model = chat_model

    @classmethod
    def from_mcp(
        cls,
        mcp_manager,
        synthesizer: RcaSynthesizer,
        *,
        config: dict[str, Any] | None = None,
        chat_model=None,
    ):
        """Create an orchestrator wired to MCP-backed observability collectors."""

        config = config or {}
        if chat_model is None:
            raise ValueError("chat_model is required for the RCA agent loop")
        from app.core.graph.utils.collector import (
            LogEvidenceCollector,
            McpToolRunner,
            MetricEvidenceCollector,
            TraceEvidenceCollector,
        )

        runner = McpToolRunner(
            mcp_manager,
            timeout_seconds=config.get("collector_timeout_seconds", 20),
            max_preview_chars=config.get("max_tool_result_preview_chars", 8000),
        )
        schema_config = config.get("observability_schema") or {}
        return cls(
            collectors={
                "loki": LogEvidenceCollector(
                    runner,
                    limit=config.get("max_evidence_per_source", 20),
                    schema_config=schema_config.get("loki"),
                ),
                "tempo": TraceEvidenceCollector(runner),
                "influxdb": MetricEvidenceCollector(
                    runner,
                    database=config.get("metric_database", "insight"),
                    schema_config=schema_config.get("influxdb"),
                ),
            },
            synthesizer=synthesizer,
            request_planner=LlmRequestPlanner(chat_model),
            chat_model=chat_model,
        )

    @staticmethod
    def db_status(result: RcaResult) -> str:
        """Map RCA report status to the analysis record status."""

        if result.status == "FAILED":
            return "FAILED"
        if result.status in {"INCONCLUSIVE", "NEEDS_CLARIFICATION"}:
            return "PARTIAL"
        return "SUCCEEDED"


# --- Graph -----------------------------------------------------------

EVIDENCE_SOURCES = ("loki", "tempo", "influxdb")
VALID_EVIDENCE_STATUSES = {"OK", "NO_DATA", "PARTIAL", "FAILED", "SKIPPED"}
COLLECT_TOOL_SOURCES = {
    "collect_logs": "loki",
    "collect_traces": "tempo",
    "collect_metrics": "influxdb",
}
# Supersteps: agent/tools rounds + synthesize + validate + finalize, with headroom.
GRAPH_RECURSION_LIMIT = 28


@dataclass(slots=True)
class ServerErrorRunContext:
    """LangGraph runtime context containing persistence and orchestration services."""

    repo: Any
    orchestrator: Any


AGENT_SYSTEM_PROMPT = (
    "You are an SRE root-cause-analysis agent.\n"
    "Use request_plan as the source strategy.\n"
    "You write LogQL, TraceQL, and InfluxQL yourself.\n"
    "Collector tools execute queries only; they do not create queries for you.\n\n"
    "Tools:\n"
    "- collect_logs(logql, start?, end?, limit?) executes Loki LogQL.\n"
    "- collect_traces(trace_id? OR traceql?, start?, end?, limit?) executes Tempo get-trace or traceql-search.\n"
    "- collect_metrics(influxql) executes read-only InfluxQL.\n"
    "- manage_plan is not available. Do not call it.\n\n"
    "Filter rule:\n"
    "- Use service_name, node_id, infra_id, trace_id, level, message, status code, or metric filters only when explicit in the request or structured context.\n"
    "- Do not infer arbitrary Korean or English words as service names or filter values.\n"
    "- Generic request words such as 문제, 장애, 이상, 확인, 봐줘, 분석, problem, issue, check, analyze are intent, not literal log filters.\n\n"
    "Retry rule:\n"
    "- If a collect tool returns syntax, parse, invalid query, unsupported label/field, or query-shape error, fix the query and call the same source once more.\n"
    "- Do not retry on NO_DATA, timeout, auth failure, datasource unavailable, missing scope, or broad scope.\n"
    "- Do not call the same source more than twice total.\n\n"
    "Stop condition:\n"
    "- Stop calling tools once enabled sources have terminal evidence.\n"
    "- If useful evidence is insufficient, stop and let synthesis produce an inconclusive RCA with limitations.\n"
    "- Do not fabricate findings."
)


def _merge_evidence(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Merge evidence updates from parallel graph state writes."""

    return {**(old or {}), **(new or {})}


def _sanitize_request_plan(plan: RequestPlan, incident: IncidentContext) -> RequestPlan:
    """Normalize planner output so graph execution has safe defaults."""

    if plan.mode == "needs_clarification":
        return plan.model_copy(update={"evidence": None})

    if plan.mode == "direct_answer":
        return plan.model_copy(update={"evidence": None})

    evidence = (plan.evidence or EvidenceQueryPlan()).model_copy(deep=True)
    if not any((evidence.loki.enabled, evidence.tempo.enabled, evidence.influxdb.enabled)):
        evidence.loki.enabled = True
    if incident.trace_id or incident.service_name or incident.node_id or incident.infra_id:
        evidence.tempo.enabled = True
    return plan.model_copy(update={"evidence": evidence})


def _query_authoring_context(state: dict[str, Any], runtime: Runtime[ServerErrorRunContext]) -> str:
    """Render source schema guidance for the ReAct query-authoring step."""

    plan = state.get("request_plan")
    incident = state.get("incident")
    evidence = getattr(plan, "evidence", None)
    collectors = getattr(runtime.context.orchestrator, "collectors", {})
    enabled = [source for source in ("loki", "tempo", "influxdb") if evidence and getattr(evidence, source).enabled]
    lines = [
        "Query authoring context.",
        "Follow these guides exactly for enabled sources only.",
        "Runtime will inject the incident, existing evidence, and tool results; you must write the source query strings.",
        f"Incident: {incident.model_dump(mode='json') if isinstance(incident, IncidentContext) else incident}",
        f"Enabled sources: {enabled or ['loki']}",
    ]
    if not enabled or "loki" in enabled:
        schema = getattr(collectors.get("loki"), "schema", None)
        labels = getattr(
            schema,
            "labels",
            {
                "ns_id": "NS_ID",
                "infra_id": "INFRA_ID",
                "node_id": "NODE_ID",
                "service_name": "service",
                "level": "level",
            },
        )
        parsed = getattr(schema, "parsed_fields", {"trace_id": "trace_id", "level": "level"})
        keywords = getattr(getattr(evidence, "loki", None), "keywords", []) if evidence else []
        lines += [
            "Loki LogQL guide:",
            f"- Stream labels: {labels}. Parsed fields after `| json`: {parsed}. Planner keywords: {keywords}.",
            "- Query must start with a non-empty selector. Never use `{}`.",
            '- If no explicit service/node/infra/trace scope exists, use `{service=~".+"}`.',
            '- Broad health-check query uses OR, not AND: `{service=~".+"} |~ "(?i)(error|exception|failed|timeout)"`.',
            "- Use parsed fields only after `| json`.",
        ]
    if "tempo" in enabled:
        lines += [
            "Tempo guide:",
            "- Trace by id: `collect_traces(trace_id=...)` maps to get-trace.",
            "- TraceQL search: `collect_traces(traceql=...)` maps to traceql-search.",
            "- Do not broad-search Tempo without clear service/node/infra scope.",
            '- Prefer simple TraceQL: `{ resource.service.name = "payment-api" }`, `{ resource.service.name = "payment-api" && span.http.status_code >= 500 }`, `{ resource.service.name = "payment-api" && status = error }`.',
        ]
    if "influxdb" in enabled:
        config = getattr(collectors.get("influxdb"), "config", None)
        measurements = getattr(
            config, "measurements", ["cpu", "disk", "diskio", "mem", "net", "processes", "procstat", "swap", "system"]
        )
        fields = getattr(config, "fields", ["usage_idle", "used_percent", "load1", "value"])
        tags = sorted(getattr(config, "tags", {"ns_id", "infra_id", "node_id"}))
        profiles = getattr(getattr(evidence, "influxdb", None), "profiles", []) if evidence else []
        db = getattr(config, "database", "insight")
        rp = getattr(config, "retention_policy", "autogen")
        lines += [
            "InfluxQL guide:",
            f"- Database: {db}. Retention policy: {rp}. Measurements: {measurements}. Fields: {fields}. Tags: {tags}. Planner profiles: {profiles}.",
            "- SELECT only. Include time predicates.",
            "- Use ns_id/infra_id/node_id tag filters only when explicit.",
            "- Prefer bounded aggregates such as mean(), max(), count().",
        ]
    return "\n".join(lines)


def _planning_context_message(state: dict[str, Any]) -> str:
    """Render current request plan and evidence for the ReAct agent."""

    request_plan = state.get("request_plan")
    payload = {
        "request_plan": request_plan.model_dump(mode="json") if request_plan else {},
        "existing_evidence": {
            source: item.model_dump(mode="json")
            for source, item in (state.get("evidence") or {}).items()
            if isinstance(item, SourceEvidence)
        },
    }
    return "Structured planning context:\n" + json.dumps(payload, ensure_ascii=False)


class RcaAgentState(MessagesState, total=False):
    """State carried through the server-error RCA LangGraph execution."""

    analysis_id: int | None
    trace_id: str | None
    query: str | None
    time_range: dict[str, Any]
    filters: dict[str, Any]
    options: dict[str, Any]
    incident: IncidentContext
    request_plan: RequestPlan
    planner_error: str
    evidence: Annotated[dict[str, SourceEvidence], _merge_evidence]
    trajectory: Annotated[list[dict[str, Any]], operator.add]
    report: RcaResult
    validation: ValidationResult
    rca_status: Literal["SUCCEEDED", "PARTIAL", "FAILED"] | None
    rca_summary: str | None
    detail: dict[str, Any]


class ServerErrorAnalysisGraphNodes:
    """Node implementations for the ReAct RCA graph."""

    async def request_plan(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """Structured request routing and evidence strategy before ReAct."""
        incident = state["incident"]
        options = state.get("options") or {}
        planner = getattr(runtime.context.orchestrator, "request_planner", None)
        if planner is None:
            reason = "LLM request planner is not configured"
            return self._request_planning_failed(state, reason)

        try:
            result = await planner.plan(incident, options)
        except Exception as exc:
            reason = f"LLM request planner failed: {exc}"
            return self._request_planning_failed(state, reason)

        request_plan = _sanitize_request_plan(result.request_plan, incident)
        return {"request_plan": request_plan}

    def _request_planning_failed(self, state: RcaAgentState, reason: str) -> dict[str, Any]:
        """Convert planner failure into a terminal failed RCA update."""

        entry = {"node": "request_plan", "status": "failed", "reason": reason}
        failed_state = {
            **state,
            "planner_error": reason,
            "trajectory": [*(state.get("trajectory") or []), entry],
        }
        report = RcaResult(
            status="FAILED",
            summary="Request planning failed",
            confidence="low",
            limitations=[reason],
            next_checks=["Retry after the LLM request planner is available."],
        )
        validation = ValidationResult(status="FAILED", findings=[reason])
        return {
            **self._terminal_update_from_report(failed_state, report, validation),
            "planner_error": reason,
            "trajectory": [entry],
        }

    async def direct_answer(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """Finish non-RCA questions without collecting observability evidence."""

        plan = state["request_plan"]
        summary = plan.answer_hint or "This request does not require observability evidence collection."
        report = RcaResult(
            status="COMPLETED",
            summary=summary,
            confidence="low",
            limitations=["observability evidence collection was skipped for a direct-answer request"],
        )
        validation = ValidationResult(status="PASSED")
        return self._terminal_update_from_report(state, report, validation)

    async def clarification(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """Finish broad RCA requests with a clarification result."""

        plan = state["request_plan"]
        reason = plan.reason or "query is too broad to start RCA collection"
        report = RcaResult(
            status="NEEDS_CLARIFICATION",
            summary="Additional context is required before RCA can start",
            confidence="low",
            limitations=[reason],
            next_checks=["Provide a query, time range, affected service, node, infra, or trace id."],
        )
        validation = ValidationResult(status="PARTIAL", findings=[reason])
        return self._terminal_update_from_report(state, report, validation)

    async def agent(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """ReAct step: the LLM writes source queries, calls collect_* tools, or stops."""
        chat_model = runtime.context.orchestrator.chat_model
        if chat_model is None:
            raise RuntimeError("chat_model is required for the RCA agent loop")
        bound = chat_model.bind_tools(AGENT_TOOLS)
        response = await bound.ainvoke(
            [
                SystemMessage(AGENT_SYSTEM_PROMPT),
                SystemMessage(_query_authoring_context(state, runtime)),
                HumanMessage(_planning_context_message(state)),
                *state.get("messages", []),
            ]
        )
        tool_calls = getattr(response, "tool_calls", None)
        goto = "tools" if tool_calls and _has_allowed_collect_source(state, tool_calls) else "synthesize"
        return Command(update={"messages": [response]}, goto=goto)

    async def tools(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> Command:
        """Execute allowed collect_* tool calls from the last agent message."""
        messages = state.get("messages") or []
        tool_calls = getattr(messages[-1], "tool_calls", None) if messages else None
        evidence_update: dict[str, SourceEvidence] = {}
        trajectory: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []

        for tool_call in tool_calls or []:
            tool_name = _tool_call_name(tool_call)
            source = COLLECT_TOOL_SOURCES.get(tool_name or "")
            if not tool_name or not source or not _can_collect_source(state, source):
                continue
            args = _tool_call_args(tool_call)
            source, incident, collect_kwargs, clean_args = _collector_request(tool_name, args, state)
            evidence = await safe_collect(runtime.context.orchestrator.collectors, source, incident, **collect_kwargs)
            evidence_update[source] = evidence
            trajectory.append({"tool": tool_name, "args": clean_args, "source": source, "status": evidence.status})
            tool_messages.append(ToolMessage(evidence_view(evidence), tool_call_id=_tool_call_id(tool_call)))

        update = {"evidence": evidence_update, "trajectory": trajectory, "messages": tool_messages}
        next_state = {
            **state,
            "evidence": {**(state.get("evidence") or {}), **evidence_update},
            "trajectory": [*(state.get("trajectory") or []), *trajectory],
            "messages": [*(state.get("messages") or []), *tool_messages],
        }
        goto = (
            "synthesize"
            if not tool_messages or _collection_complete(next_state) or _tool_budget_exhausted(next_state)
            else "agent"
        )
        return Command(update=update, goto=goto)

    async def synthesize(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """Deterministic single-call RCA synthesis from collected evidence."""
        incident = state["incident"]
        evidence = state.get("evidence") or {}
        evidence_plan = _evidence_plan_from_state(state)
        synthesizer = runtime.context.orchestrator.synthesizer
        if synthesizer is None:
            raise ValueError("synthesizer is required for RCA synthesis")
        report = await synthesizer.synthesize(incident, evidence_plan, evidence)
        return {"report": report}

    async def validate(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """Deterministic evidence-integrity validation (confidence is LLM-owned)."""
        orchestrator = runtime.context.orchestrator
        incident = state["incident"]
        evidence = state.get("evidence") or {}
        evidence_plan = _evidence_plan_from_state(state)
        report = state.get("report") or _inconclusive_report(evidence)
        validation = orchestrator.validator.validate(incident, evidence_plan, evidence, report)
        return self._terminal_update_from_report(state, report, validation)

    async def finalize(self, state: RcaAgentState, runtime: Runtime[ServerErrorRunContext]) -> dict[str, Any]:
        """Persist the terminal RCA result via the repository."""
        detail = state.get("detail") or {}
        status = state.get("rca_status")
        summary = state.get("rca_summary") or ""
        repo = runtime.context.repo
        if status == "FAILED":
            repo.save_failed(state["analysis_id"], summary, detail)
        elif status == "PARTIAL":
            repo.save_partial(state["analysis_id"], summary=summary, detail=detail)
        else:
            repo.save_success(state["analysis_id"], summary=summary, detail=detail)
        return {"detail": detail}

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _request_detail_from_state(state: RcaAgentState) -> dict[str, Any]:
        """Extract request metadata for the persisted detail payload."""

        return {
            "query": state.get("query"),
            "time_range": state.get("time_range") or {},
            "filters": state.get("filters") or {},
            "options": state.get("options") or {},
        }

    @classmethod
    def _terminal_update_from_report(
        cls, state: RcaAgentState, report: RcaResult, validation: ValidationResult
    ) -> dict[str, Any]:
        """Build graph state fields shared by every terminal RCA path."""

        return {
            "report": report,
            "validation": validation,
            "rca_status": ServerErrorRcaOrchestrator.db_status(report),
            "rca_summary": report.summary or report.root_cause or "RCA inconclusive",
            "detail": cls._build_detail_from_state(state, report, validation),
        }

    @classmethod
    def _build_detail_from_state(
        cls, state: RcaAgentState, report: RcaResult, validation: ValidationResult
    ) -> dict[str, Any]:
        """Build the final detail JSON from graph state and RCA outputs."""

        incident = state["incident"]
        evidence = state.get("evidence") or {}
        evidence_plan = _evidence_plan_from_state(state)
        detail = build_rca_detail(
            request=cls._request_detail_from_state(state),
            incident=incident,
            evidence_plan=evidence_plan,
            evidence=evidence,
            result=report,
            validation=validation,
            request_plan=state.get("request_plan"),
        )
        detail["trajectory"] = state.get("trajectory") or []
        return detail


def _evidence_plan_from_state(state: RcaAgentState) -> EvidencePlan:
    """Resolve the required source list from request plan or collected evidence."""

    request_plan = state.get("request_plan")
    if request_plan and request_plan.mode != "evidence_rca":
        return EvidencePlan(required_sources=[])
    if request_plan and request_plan.evidence:
        required = [
            source for source in ("loki", "tempo", "influxdb") if getattr(request_plan.evidence, source).enabled
        ]
        return EvidencePlan(required_sources=required or ["loki"])
    evidence = state.get("evidence") or {}
    required = [source for source in ("loki", "tempo", "influxdb") if evidence.get(source)]
    return EvidencePlan(required_sources=required or ["loki"])


def _inconclusive_report(evidence: dict[str, SourceEvidence]) -> RcaResult:
    """Create a fallback inconclusive report when synthesis has no usable output."""

    limitations = [limitation for item in evidence.values() for limitation in item.limitations]
    return RcaResult(
        status="INCONCLUSIVE",
        summary="RCA inconclusive: insufficient evidence was collected",
        confidence="low",
        limitations=limitations or ["no supporting evidence was found"],
        next_checks=["Verify the time range, filters, and observability ingestion for the requested scope."],
    )


def _source_status(item: Any) -> str | None:
    """Read an evidence status from either a model or a plain dict."""

    if isinstance(item, dict):
        return item.get("status")
    return getattr(item, "status", None)


def _required_sources(state: RcaAgentState) -> set[str]:
    """Return the source set that the graph still needs to satisfy."""

    return set(_evidence_plan_from_state(state).required_sources)


QUERY_ERROR_MARKERS = (
    "invalid query",
    "malformed",
    "parse",
    "query shape",
    "query-shape",
    "syntax",
    "unexpected",
    "unknown field",
    "unknown label",
    "unknown measurement",
    "unsupported field",
    "unsupported label",
)


def _source_attempt_count(state: RcaAgentState, source: str) -> int:
    """Count collect tool attempts already made for one source."""

    return sum(
        1
        for entry in state.get("trajectory", [])
        if entry.get("source") == source and entry.get("tool") in COLLECT_TOOL_SOURCES
    )


def _repairable_query_failure(item: SourceEvidence | None) -> bool:
    """Detect failed evidence that may be fixed by rewriting the query once."""

    if not item or item.status != "FAILED":
        return False
    text = " ".join([*(item.limitations or []), item.error or ""]).lower()
    return any(marker in text for marker in QUERY_ERROR_MARKERS)


def _source_terminal(state: RcaAgentState, source: str) -> bool:
    """Return whether one source has reached a final collection state."""

    item = (state.get("evidence") or {}).get(source)
    if item is None:
        return False
    if _repairable_query_failure(item) and _source_attempt_count(state, source) < 2:
        return False
    return _source_status(item) in VALID_EVIDENCE_STATUSES


def _collection_complete(state: RcaAgentState) -> bool:
    """Return whether every required source has terminal evidence."""

    required = _required_sources(state)
    return bool(required) and all(_source_terminal(state, source) for source in required)


def _tool_call_name(tool_call: Any) -> str | None:
    """Extract a tool-call name from LangChain dict or object formats."""

    if isinstance(tool_call, dict):
        return tool_call.get("name")
    return getattr(tool_call, "name", None)


def _tool_call_id(tool_call: Any) -> str:
    """Extract a stable tool-call id for the tool response message."""

    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(getattr(tool_call, "id", "") or "")


def _tool_call_args(tool_call: Any) -> dict[str, Any]:
    """Extract tool-call arguments as a plain dictionary."""

    args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
    return dict(args or {}) if isinstance(args, dict) else {}


def _collector_request(
    tool_name: str,
    args: dict[str, Any],
    state: RcaAgentState,
) -> tuple[str, IncidentContext, dict[str, Any], dict[str, Any]]:
    """Translate a collect_* tool call into collector kwargs and audit args."""

    source = COLLECT_TOOL_SOURCES[tool_name]
    if tool_name == "collect_logs":
        limit = args.get("limit")
        max_limit = limit if isinstance(limit, int) and limit > 0 else _max_evidence(state)
        incident = _incident_with_overrides(
            state.get("incident"),
            start=args.get("start"),
            end=args.get("end"),
            max_evidence_per_source=max_limit,
        )
        return (
            source,
            incident,
            {"logql": args.get("logql"), "limit": max_limit},
            {
                "logql": args.get("logql"),
                "start": args.get("start"),
                "end": args.get("end"),
                "limit": limit,
            },
        )
    if tool_name == "collect_traces":
        limit = args.get("limit")
        max_limit = limit if isinstance(limit, int) and limit > 0 else None
        incident = _incident_with_overrides(state.get("incident"), start=args.get("start"), end=args.get("end"))
        return (
            source,
            incident,
            {
                "trace_id": args.get("trace_id"),
                "traceql": args.get("traceql"),
                "limit": max_limit,
            },
            {
                "trace_id": args.get("trace_id"),
                "traceql": args.get("traceql"),
                "start": args.get("start"),
                "end": args.get("end"),
                "limit": limit,
            },
        )
    incident = _incident_with_overrides(state.get("incident"))
    return source, incident, {"influxql": args.get("influxql")}, {"influxql": args.get("influxql")}


def _tool_message_count(state: RcaAgentState) -> int:
    """Count completed tool response messages in graph state."""

    return sum(1 for message in state.get("messages", []) if getattr(message, "type", None) == "tool")


def _tool_budget_exhausted(state: RcaAgentState) -> bool:
    """Stop the ReAct loop after the allowed attempts per required source."""

    return _tool_message_count(state) >= max(1, len(_required_sources(state)) * 2)


def _can_collect_source(state: RcaAgentState, source: str) -> bool:
    """Allow first collection and one retry for repairable query failures."""

    required = _required_sources(state)
    if source not in required:
        return False
    attempts = _source_attempt_count(state, source)
    if attempts == 0:
        return True
    return attempts == 1 and _repairable_query_failure((state.get("evidence") or {}).get(source))


def _has_allowed_collect_source(state: RcaAgentState, tool_calls: list[Any]) -> bool:
    """Check whether the agent proposed at least one collectable source."""

    return any(
        (source := COLLECT_TOOL_SOURCES.get(_tool_call_name(tool_call) or "")) and _can_collect_source(state, source)
        for tool_call in tool_calls
    )


def _route_after_request_plan(state: RcaAgentState) -> Literal["finalize", "direct_answer", "clarification", "agent"]:
    """Route from request planning to the correct graph branch."""

    if state.get("planner_error"):
        return "finalize"
    mode = state["request_plan"].mode
    if mode == "direct_answer":
        return "direct_answer"
    if mode == "needs_clarification":
        return "clarification"
    return "agent"


def build_server_error_analysis_graph(checkpointer=None):
    """Compile the RCA graph: request_plan -> agent/tools or terminal paths."""
    nodes = ServerErrorAnalysisGraphNodes()
    graph = StateGraph(RcaAgentState, context_schema=ServerErrorRunContext)

    graph.add_node("request_plan", nodes.request_plan)
    graph.add_node("direct_answer", nodes.direct_answer)
    graph.add_node("clarification", nodes.clarification)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", nodes.tools)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("validate", nodes.validate)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "request_plan")
    graph.add_conditional_edges(
        "request_plan",
        _route_after_request_plan,
        {
            "finalize": "finalize",
            "direct_answer": "direct_answer",
            "clarification": "clarification",
            "agent": "agent",
        },
    )
    graph.add_edge("direct_answer", "finalize")
    graph.add_edge("clarification", "finalize")
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
