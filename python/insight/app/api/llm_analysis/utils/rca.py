import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException, status
from langchain_core.callbacks import get_usage_metadata_callback
from sqlalchemy.orm import Session

from app.api.llm_analysis.repo.repo import LogAnalysisRepository, RcaAnalysisRepository
from app.api.llm_analysis.request.req import (
    PostRcaQueryBody,
    RcaRecordFilter,
)
from app.api.llm_analysis.response.res import (
    Message,
    RcaAnalysisRecord,
    RcaQueryResult,
    RcaRecordPage,
)
from app.api.llm_analysis.utils.llm_model import create_chat_model
from app.api.llm_analysis.utils.session import CommonSessionService
from app.core.graph.rca import (
    SOURCE_SPECS,
    EvidenceStore,
    IncidentScope,
    RcaRunContext,
    RequestBudget,
    build_investigation_runner,
    build_investigation_toolset,
)
from config.ConfigManager import ConfigManager

logger = logging.getLogger(__name__)

_EVIDENCE_RECORD_CONTEXT_RATIO = 0.60
_SYNTHESIS_EVIDENCE_CONTEXT_RATIO = 0.80
# The tool-call ledger is bounded by the request budget, but a model that keeps issuing
# blocked calls can still grow it; the record keeps the head and says what was cut.
_MAX_PERSISTED_TOOL_CALLS = 100


def _summarize_token_usage(usage_metadata: dict | None) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in (usage_metadata or {}).values():
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return {key: value for key, value in totals.items() if value}


# Warn once per model so a busy endpoint does not repeat the same line every request.
_warned_unknown_context: set[str] = set()


def _positive_int(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _resolve_context_window(
    llm,
    analysis_config: dict,
    *,
    explicit_context_length: int | None,
    model_name: str,
) -> int:
    """Resolve the input context window, preferring what the operator declared.

    The window is a property of the *endpoint*, not of the model name: Ollama sizes it
    from the host's VRAM, so one server serves a model at 4k and another at 256k. No
    registry can know that, which is why the connection's own value wins over anything
    we can infer, and why an unknown window warns instead of failing quietly.
    """
    if declared := _positive_int(explicit_context_length):
        return declared
    # num_ctx is the window create_chat_model sent to Ollama; profile is langchain's
    # catalogue, which is empty for custom endpoints.
    profile = getattr(llm, "profile", None)
    for candidate in (
        getattr(llm, "num_ctx", None),
        profile.get("max_input_tokens") if isinstance(profile, dict) else None,
    ):
        if known := _positive_int(candidate):
            return known

    # The default lives in ConfigManager only; an absent key is a configuration bug.
    fallback = _positive_int(analysis_config["fallback_context_window_tokens"])
    if fallback is None:
        raise ValueError("rca_analysis.fallback_context_window_tokens must be a positive integer")
    if model_name not in _warned_unknown_context:
        _warned_unknown_context.add(model_name)
        logger.warning(
            "Unknown context window for model %r; assuming %d tokens. If this endpoint "
            "serves less, the server will silently drop the oldest messages. Set the "
            "connection's context_length to the window it actually allocates.",
            model_name or "<unnamed>",
            fallback,
        )
    return fallback


def _derive_token_budgets(
    llm,
    analysis_config: dict,
    *,
    explicit_context_length: int | None = None,
    model_name: str = "",
) -> dict[str, int]:
    context_window = _resolve_context_window(
        llm,
        analysis_config,
        explicit_context_length=explicit_context_length,
        model_name=model_name,
    )
    single_tool_max = min(
        int(
            context_window
            * analysis_config.get("tool_result_context_window_pct", 15)
            / 100
        ),
        int(analysis_config.get("tool_result_absolute_max_tokens", 25_000)),
    )
    return {
        "context_window_tokens": context_window,
        "single_tool_max_tokens": single_tool_max,
        "evidence_record_budget_tokens": int(
            context_window * _EVIDENCE_RECORD_CONTEXT_RATIO
        ),
        "synthesis_evidence_max_tokens": int(
            context_window * _SYNTHESIS_EVIDENCE_CONTEXT_RATIO
        ),
    }


def _persisted_tool_calls(context) -> dict:
    toolset = getattr(context, "investigation_toolset", None)
    if toolset is None:
        return {"tool_calls": []}
    ledger = toolset.ledger()
    persisted = {"tool_calls": ledger[:_MAX_PERSISTED_TOOL_CALLS]}
    if len(ledger) > _MAX_PERSISTED_TOOL_CALLS:
        persisted["tool_calls_truncated"] = len(ledger) - _MAX_PERSISTED_TOOL_CALLS
    return persisted


class RcaAnalysisService:
    """Coordinate RCA API operations, agents, graph execution, and persistence."""

    def __init__(self, db: Session, mcp_manager=None, rca_graph=None):
        self.db = db
        self.session_repo = LogAnalysisRepository(db)
        self.analysis_repo = RcaAnalysisRepository(db)
        self.mcp_manager = mcp_manager
        self.rca_graph = rca_graph
        self.config = ConfigManager()
        self.analysis_config = self.config.get_rca_analysis_config()

    async def query_rca(self, body: PostRcaQueryBody) -> RcaQueryResult:
        """Run RCA through the service-owned record lifecycle and graph."""
        started_at = time.perf_counter()
        session = CommonSessionService(self.db).get_or_create_session(
            analysis_type="rca",
            session_id=body.session_id,
            connection_id=body.connection_id,
            model_name=body.model_name,
        )
        resolved = PostRcaQueryBody.model_validate(
            {
                **body.model_dump(
                    mode="json",
                    exclude={"connection_id", "model_name"},
                ),
                "session_id": session.SESSION_ID,
            }
        )
        request_json = resolved.model_dump(mode="json", exclude_none=True)
        record = self.analysis_repo.create_record(
            trace_id=resolved.scope.trace_id,
            session_id=session.SESSION_ID,
            request_json=request_json,
        )

        context = None
        try:
            with TemporaryDirectory(prefix=f"rca-{record.ID}-") as directory:
                context = await self._create_rca_context(
                    session.MODEL_NAME,
                    session.CONNECTION_ID,
                    storage_dir=Path(directory),
                    scope=resolved.scope,
                )
                self.db.close()
                with get_usage_metadata_callback() as usage_callback:
                    graph_result = await self._get_rca_graph().ainvoke(
                        {
                            "session_id": session.SESSION_ID,
                            "query": resolved.query,
                            "scope": resolved.scope.model_dump(mode="json"),
                            "filters": resolved.filters,
                        },
                        context=context,
                    )
                graph_result = {
                    **graph_result,
                    "llm_token_usage": _summarize_token_usage(
                        usage_callback.usage_metadata
                    ),
                }
        except Exception as exc:
            error_message = str(exc)
            self.analysis_repo.finalize(
                record.ID,
                status="FAILED",
                summary=error_message,
                detail={"error_message": error_message, **_persisted_tool_calls(context)},
            )
            self._log_operational_summary(
                record.ID,
                started_at,
                {"result_validation": {"status": "FAILED"}, "error_message": error_message},
                context,
            )
            raise

        merged_evidence = graph_result.get("merged_evidence") or {}
        detail = {
            "analysis_result": graph_result.get("analysis_result"),
            "result_validation": graph_result.get("result_validation"),
            "evidence_status": merged_evidence.get("sources") or {},
            "errors": [graph_result["error_message"]] if graph_result.get("error_message") else [],
            # Every wrapped tool call of the request — including empty, blocked and failed
            # ones — so the record explains what was looked at, not only what was cited.
            **_persisted_tool_calls(context),
        }
        analysis_result = graph_result.get("analysis_result") or {}
        summary = analysis_result.get("summary") or graph_result.get("error_message") or ""
        validation = graph_result.get("result_validation") or {}
        validation_status = validation.get("status")
        if validation_status == "FAILED":
            detail["analysis_result"] = None
            final_status = "FAILED"
            summary = graph_result.get("error_message") or "RCA failed"
        elif validation.get("no_telemetry"):
            # Every source ran and the window was empty. Reporting that as a failed
            # analysis tells an operator to go fix a pipeline that is working.
            final_status = "PARTIAL"
            summary = "No telemetry data was found in the requested scope and time window."
        elif not analysis_result:
            final_status = "FAILED"
            summary = summary or "RCA failed"
        elif validation_status == "PARTIAL":
            final_status = "PARTIAL"
        else:
            final_status = "SUCCEEDED"
        updated_record = self.analysis_repo.finalize(
            record.ID,
            status=final_status,
            summary=summary,
            detail=detail,
        )
        self._log_operational_summary(record.ID, started_at, graph_result, context)

        return RcaQueryResult(
            session_id=session.SESSION_ID,
            message=Message(message_type="ai", message=summary),
            analysis=self._to_record(updated_record),
        )

    def list_records(self, params: RcaRecordFilter) -> RcaRecordPage:
        """Return paginated RCA records with normalized detail envelopes."""
        total, items = self.analysis_repo.list_records(
            status=params.status,
            from_dt=params.from_dt,
            to_dt=params.to_dt,
            page=params.page,
            size=params.size,
        )
        return RcaRecordPage(
            total=total,
            page=params.page,
            size=params.size,
            items=[self._to_record(item) for item in items],
        )

    def get_record(self, analysis_id: int) -> RcaAnalysisRecord:
        """Return one RCA record or raise 404 when missing."""
        record = self.analysis_repo.get_by_id(analysis_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis Not Found")
        return self._to_record(record)

    def _get_rca_graph(self):
        """Return the application-managed compiled graph or fail when runtime is unavailable."""
        if self.rca_graph is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RCA graph is not initialized",
            )
        return self.rca_graph

    async def _create_rca_context(
        self,
        model_name,
        connection_id=None,
        *,
        storage_dir: Path | None = None,
        scope: IncidentScope | None = None,
    ) -> RcaRunContext:
        """Build everything one request shares: budget, store, toolset and the single agent."""
        llm = create_chat_model(
            self.session_repo,
            model_name,
            connection_id=connection_id,
        )
        connection = (
            self.session_repo.get_connection_by_id(connection_id)
            if connection_id is not None
            else None
        )
        token_budgets = _derive_token_budgets(
            llm,
            self.analysis_config,
            explicit_context_length=getattr(connection, "CONTEXT_LENGTH", None),
            model_name=model_name,
        )
        config = self.analysis_config
        timeout_seconds = float(config["analysis_timeout_seconds"] or 0)
        budget = RequestBudget(
            tool_call_limit=int(config["investigation_tool_call_limit"]),
            model_call_limit=int(config["investigation_model_call_limit"]),
            deadline_seconds=timeout_seconds if timeout_seconds > 0 else None,
        )
        evidence_store = EvidenceStore(
            storage_dir=storage_dir,
            model_name=model_name,
            single_tool_max_tokens=token_budgets["single_tool_max_tokens"],
            record_budget_tokens=token_budgets["evidence_record_budget_tokens"],
        )
        toolset = await build_investigation_toolset(
            scope=scope or IncidentScope(),
            source_tools=self._source_tools(),
            datasources=config.get("datasources", {}),
            evidence_store=evidence_store,
            budget=budget,
        )
        runner = (
            build_investigation_runner(llm=llm, toolset=toolset, budget=budget)
            if toolset.queryable_sources
            else None
        )
        return RcaRunContext(
            analysis_config={
                **config,
                **token_budgets,
                "model_name": model_name,
            },
            llm=llm,
            budget=budget,
            evidence_store=evidence_store,
            investigation_toolset=toolset,
            investigation_runner=runner,
        )

    def _source_tools(self) -> dict[str, dict]:
        """Allowlisted raw MCP tools per source, for the sources whose required tools are all present."""
        if not self.mcp_manager or not hasattr(self.mcp_manager, "get_tools_for_mcp"):
            return {}
        available = {}
        for source, spec in SOURCE_SPECS.items():
            allowed = {*spec["required_tools"], *spec["optional_tools"]}
            by_name = {
                tool.name: tool
                for tool in self.mcp_manager.get_tools_for_mcp(spec["mcp"])
                if getattr(tool, "name", "") in allowed
            }
            if set(spec["required_tools"]).issubset(by_name):
                available[source] = by_name
        return available

    @staticmethod
    def _log_operational_summary(analysis_id: int, started_at: float, graph_result: dict, context=None):
        """Emit one bounded RCA operation log line per analysis.

        ``tool_calls`` carries per-source call counts and durations from the request's
        toolset; it is the measurement the deadline and budget defaults are tuned from.
        """
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        evidence_status = ((graph_result.get("merged_evidence") or {}).get("sources")) or {}
        result_status = ((graph_result.get("result_validation") or {}).get("status")) or "UNKNOWN"
        llm_tokens = graph_result.get("llm_token_usage") or {}
        toolset = getattr(context, "investigation_toolset", None)
        tool_calls = toolset.call_summary() if toolset is not None else {}
        logger.info(
            "rca_analysis_completed analysis_id=%s duration_ms=%s llm_tokens=%s evidence_status=%s "
            "result_status=%s tool_calls=%s",
            analysis_id,
            duration_ms,
            llm_tokens,
            evidence_status,
            result_status,
            tool_calls,
        )

    @staticmethod
    def _to_record(record):
        """Map a persistence model into the public API response model."""
        return RcaAnalysisRecord(
            id=record.ID,
            trace_id=record.TRACE_ID,
            session_id=record.SESSION_ID,
            status=record.STATUS,
            summary=record.SUMMARY,
            request=record.REQUEST_JSON or {},
            detail=record.DETAIL_JSON,
            created_at=record.CREATED_AT,
            updated_at=record.UPDATED_AT,
        )
