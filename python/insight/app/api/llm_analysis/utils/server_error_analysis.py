from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.api.llm_analysis.repo.repo import LogAnalysisRepository, ServerErrorAnalysisRepository
from app.api.llm_analysis.request.req import (
    PostServerErrorQueryBody,
    ServerErrorRecordFilter,
)
from app.api.llm_analysis.response.res import (
    Message,
    ServerErrorAnalysisRecord,
    ServerErrorQueryResult,
    ServerErrorRecordPage,
)
from app.api.llm_analysis.utils.llm_api_key import CredentialService
from app.core.graph.server_error_analysis_graph import (
    GRAPH_RECURSION_LIMIT,
    IncidentContext,
    LangChainStructuredLLM,
    RcaSynthesizer,
    ServerErrorRcaOrchestrator,
    ServerErrorRunContext,
    normalize_server_error_detail,
)
from app.core.llm.ollama_client import OllamaClient
from app.core.llm.openai_client import OpenAIClient
from config.ConfigManager import ConfigManager

OPENAI_COMPAT_BASE_URLS = {
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "anthropic": "https://api.anthropic.com/v1/",
}


def _parse_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def incident_from_request(
    *,
    session_id: str,
    analysis_id: int | None = None,
    query: str | None = None,
    filters: dict[str, Any] | None = None,
    time_range: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> IncidentContext:
    filters = filters or {}
    time_range = time_range or {}
    raw_trace = trace_id or filters.get("trace_id")
    trace = raw_trace.strip() if isinstance(raw_trace, str) and raw_trace.strip() else None
    return IncidentContext(
        session_id=session_id,
        analysis_id=analysis_id,
        query=query,
        trace_id=trace,
        service_name=filters.get("service_name"),
        node_id=filters.get("node_id"),
        infra_id=filters.get("infra_id"),
        level=filters.get("level"),
        message_filter=filters.get("message"),
        time_range_start=_parse_dt(time_range.get("start")),
        time_range_end=_parse_dt(time_range.get("end")),
    )


def incident_brief(incident: IncidentContext) -> str:
    scope = {
        "query": incident.query,
        "service_name": incident.service_name,
        "node_id": incident.node_id,
        "infra_id": incident.infra_id,
        "trace_id": incident.trace_id,
        "level": incident.level,
        "message": incident.message_filter,
        "time_range_start": incident.time_range_start.isoformat() if incident.time_range_start else None,
        "time_range_end": incident.time_range_end.isoformat() if incident.time_range_end else None,
    }
    scope_text = ", ".join(f"{k}={v}" for k, v in scope.items() if v) or "(no explicit scope)"
    return (
        "Investigate the root cause of this observability incident.\n"
        f"Scope: {scope_text}\n"
        "Use the request plan, write source queries, then collect evidence."
    )


class ServerErrorAnalysisService:
    """Coordinate observability RCA API operations, graph execution, and persistence."""

    def __init__(self, db: Session, mcp_manager=None, server_error_graph=None):
        self.db = db
        self.session_repo = LogAnalysisRepository(db)
        self.analysis_repo = ServerErrorAnalysisRepository(db)
        self.mcp_manager = mcp_manager
        self.server_error_graph = server_error_graph
        self.config = ConfigManager()
        self.analysis_config = self.config.get_server_error_analysis_config()

    async def query(self, body: PostServerErrorQueryBody) -> ServerErrorQueryResult:
        """Run a manual observability RCA query as a new analysis record."""
        session = self._get_or_create_session(body.session_id, body.provider, body.model_name)
        trace_id = body.trace_id
        record = self.analysis_repo.create(trace_id=trace_id, session_id=session.SESSION_ID, detail={})
        analysis_id = record.ID

        time_range = body.time_range.model_dump(exclude_none=True) if body.time_range else {}
        filters = body.filters.model_dump(exclude_none=True)
        incident = incident_from_request(
            session_id=session.SESSION_ID,
            analysis_id=analysis_id,
            query=body.query,
            filters=filters,
            time_range=time_range,
            trace_id=trace_id,
        )

        if not self.analysis_repo.mark_running(analysis_id):
            return self._result_from_record(self.analysis_repo.get_by_id(analysis_id) or record)

        orchestrator = self._create_rca_orchestrator(session.PROVIDER, session.MODEL_NAME)
        graph = self._get_server_error_graph()
        try:
            result = await graph.ainvoke(
                {
                    "analysis_id": analysis_id,
                    "trace_id": trace_id,
                    "time_range": time_range,
                    "query": body.query,
                    "filters": filters,
                    "options": body.options.model_dump(),
                    "incident": incident,
                    "messages": [HumanMessage(incident_brief(incident))],
                },
                context=self._graph_context(orchestrator),
                config={
                    "recursion_limit": GRAPH_RECURSION_LIMIT,
                    "configurable": {"thread_id": f"sea-{analysis_id}"},
                },
            )
        except Exception as exc:
            # The graph owns persistence via its finalize node; if it crashes (transport
            # error, recursion limit, etc.) the record would be stuck RUNNING, so fail it here.
            self.analysis_repo.save_failed(
                analysis_id,
                f"RCA graph execution failed: {exc}",
                detail={
                    "errors": [{"type": "graph_error", "message": str(exc)}],
                    "result": {"status": "FAILED", "summary": "RCA graph execution failed"},
                },
            )
            return self._result_from_record(self.analysis_repo.get_by_id(analysis_id))

        updated_record = self.analysis_repo.get_by_id(result.get("analysis_id") or analysis_id)
        message = result.get("rca_summary") or (updated_record.SUMMARY if updated_record else "")
        return ServerErrorQueryResult(
            message=Message(message_type="ai", message=message),
            analysis=self._to_record(updated_record) if updated_record else None,
        )

    def list_records(self, params: ServerErrorRecordFilter) -> ServerErrorRecordPage:
        """Return paginated server-error analysis records with normalized detail envelopes."""
        total, items = self.analysis_repo.list_records(
            status=params.status,
            from_dt=params.from_dt,
            to_dt=params.to_dt,
            page=params.page,
            size=params.size,
        )
        return ServerErrorRecordPage(
            total=total,
            page=params.page,
            size=params.size,
            items=[self._to_record(item) for item in items],
        )

    def get_record(self, analysis_id: int) -> ServerErrorAnalysisRecord:
        """Return one server-error analysis record or raise 404 when missing."""
        record = self.analysis_repo.get_by_id(analysis_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis Not Found")
        return self._to_record(record)

    def _get_server_error_graph(self):
        """Return the application-managed compiled graph or fail when runtime is unavailable."""
        if self.server_error_graph is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server error analysis graph is not initialized",
            )
        return self.server_error_graph

    def _graph_context(self, orchestrator) -> ServerErrorRunContext:
        """Build runtime context injected into the LangGraph execution."""
        return ServerErrorRunContext(
            repo=self.analysis_repo,
            orchestrator=orchestrator,
        )

    def _create_rca_orchestrator(self, provider, model_name):
        """Create the deterministic RCA orchestrator used by the canonical graph path."""
        client = self._create_llm_client(provider, model_name)
        return ServerErrorRcaOrchestrator.from_mcp(
            self.mcp_manager,
            synthesizer=RcaSynthesizer(LangChainStructuredLLM(client.llm)),
            chat_model=client.llm,
            config=self.analysis_config,
        )

    def _create_llm_client(self, provider, model_name):
        provider_value = self._provider_value(provider)
        provider_config = CredentialService(repo=self.session_repo).get_provider_config(provider=provider_value)
        if provider_value == "ollama":
            client = OllamaClient(provider_config.base_url)
        elif provider_value == "openai-compatible":
            client = OpenAIClient(provider_config.api_key, base_url=provider_config.base_url)
        else:
            client = OpenAIClient(provider_config.api_key, base_url=OPENAI_COMPAT_BASE_URLS.get(provider_value))
        client.setup(model_name)
        return client

    def _get_or_create_session(self, session_id, provider, model_name):
        """Reuse a requested chat session or create one with server-error defaults."""
        if session_id:
            session = self.session_repo.get_session_by_id(session_id)
            if not session:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session Not Found")
            return session

        if not provider or not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="provider and model_name are required when session_id is not provided",
            )
        provider_value = self._provider_value(provider)
        session_data = {
            "USER_ID": "1",
            "SESSION_ID": f"server_error_{uuid4().hex}",
            "PROVIDER": provider_value,
            "MODEL_NAME": model_name,
        }
        return self.session_repo.create_session(session_data)

    def _result_from_record(self, record) -> ServerErrorQueryResult:
        """Build a query result from an existing record without re-running the graph."""
        return ServerErrorQueryResult(
            message=Message(message_type="ai", message=record.SUMMARY or ""),
            analysis=self._to_record(record),
        )

    @staticmethod
    def _provider_value(provider) -> str:
        """Return a plain provider string from enum-like or string provider inputs."""
        return getattr(provider, "value", provider)

    @staticmethod
    def _to_record(record):
        """Map a persistence model into the public API response model."""
        return ServerErrorAnalysisRecord(
            id=record.ID,
            trace_id=record.TRACE_ID,
            session_id=record.SESSION_ID,
            status=record.STATUS,
            summary=record.SUMMARY,
            detail=normalize_server_error_detail(record.DETAIL_JSON, trace_id=record.TRACE_ID),
            created_at=record.CREATED_AT,
            updated_at=record.UPDATED_AT,
        )
