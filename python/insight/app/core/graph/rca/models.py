import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

EVIDENCE_SOURCES = ("trace", "log", "metric")


class RcaEvidenceItem(BaseModel):
    evidence_id: str
    source: Literal["trace", "log", "metric"]
    signal: str
    observation: str
    supports_cause: bool


class RcaHypothesis(BaseModel):
    cause: str = Field(min_length=1, max_length=4000)
    supporting_evidence: list[str] = Field(default_factory=list, max_length=20)
    contradicting_evidence: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)


class RcaResult(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    conclusion_strength: Literal["CONFIRMED", "LIKELY", "INCONCLUSIVE"] = "INCONCLUSIVE"
    summary: str
    probable_cause: str
    evidence: list[RcaEvidenceItem]
    mitigation: list[str]
    limitations: list[str]
    affected_service: str | None = None
    affected_endpoint: str | None = None
    hypotheses: list[RcaHypothesis] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def canonicalize_probable_cause(self):
        if not self.probable_cause.strip():
            self.probable_cause = ""
            return self
        if not self.hypotheses:
            raise ValueError("probable_cause requires at least one ranked hypothesis")
        self.probable_cause = self.hypotheses[0].cause
        return self


class IncidentTimeRange(BaseModel):
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def validate_range(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("time range start and end must be provided together")
        if self.start is None:
            return self
        if self.start.utcoffset() is None or self.end.utcoffset() is None:
            raise ValueError("time range must include a timezone")
        if self.start >= self.end:
            raise ValueError("time range start must be before end")
        return self


class IncidentScope(BaseModel):
    trace_id: str | None = Field(default=None, min_length=1, max_length=256)
    service_name: str | None = Field(default=None, min_length=1, max_length=255)
    status_code: str | None = Field(default=None, min_length=1, max_length=32)
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    time_range: IncidentTimeRange = Field(default_factory=IncidentTimeRange)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ToolTraceEntry(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)


class EvidenceRecord(BaseModel):
    evidence_id: str
    source: Literal["trace", "log", "metric"]
    capability: str
    signal: str
    observation: str
    tool: str
    query: dict[str, Any] = Field(default_factory=dict)


class EvidenceResult(BaseModel):
    source: Literal["trace", "log", "metric"]
    capability: str = ""
    # NO_DATA is a successful observation of an empty window, not an execution problem:
    # keeping it apart from FAILED/PARTIAL is what stops "nothing happened" from reading
    # like "we could not look".
    status: Literal["OK", "NO_DATA", "PARTIAL", "FAILED", "SKIPPED"]
    summary: str = ""
    discovered_trace_ids: list[str] = Field(default_factory=list)
    truncated: bool = False
    limitations: list[str] = Field(default_factory=list)
    tool_trace: SkipJsonSchema[list[ToolTraceEntry]] = Field(default_factory=list)
    records: SkipJsonSchema[list[EvidenceRecord]] = Field(default_factory=list)


class ValidatedSourceFilters(BaseModel):
    source: Literal["trace", "log", "metric"]
    verified_filters: dict[str, Any] = Field(default_factory=dict)
    ignored_filters: dict[str, str] = Field(default_factory=dict)
    discovery: dict[str, Any] = Field(default_factory=dict)


class EvidenceTask(BaseModel):
    capability: str
    focus: str = ""
    # Set by the planner's validator, not by the model: "broad" means the request named no
    # target for this source, so the collector discovers one instead of skipping.
    breadth: SkipJsonSchema[Literal["scoped", "broad"]] = "scoped"


class DraftEvidencePlan(BaseModel):
    hypotheses: list[str] = Field(default_factory=list, max_length=5)
    tasks: list[EvidenceTask] = Field(default_factory=list)
    reasoning: str = ""


class RcaAnalysisState(TypedDict, total=False):
    query: str | None
    available_capabilities: list[str]
    hypotheses: list[str]
    evidence_gaps: list[str]
    investigation_round: int
    scope: dict[str, Any]
    filters: dict[str, Any]
    task: dict[str, Any]
    plan_size: int
    evidence_plan: dict[str, Any]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    merged_evidence: dict[str, Any]
    result_validation: dict[str, Any]
    session_id: str
    analysis_result: dict | None
    error_message: str | None


@dataclass(slots=True)
class RcaRunContext:
    analysis_config: dict[str, Any]
    llm: BaseChatModel | None = None
    collector_factory: Any = None


@dataclass(slots=True)
class SourceCollector:
    tools: dict[str, Any]
    runner_factory: Any = None
    evidence_store: Any = None
