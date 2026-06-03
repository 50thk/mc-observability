from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import aiosqlite
from fastapi import HTTPException, status
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy.orm import Session

from app.api.llm_analysis.repo.repo import LogAnalysisRepository, ServerErrorAnalysisRepository
from app.api.llm_analysis.request.req import (
    PostServerErrorDetectBody,
    PostServerErrorQueryBody,
    ServerErrorRecordFilter,
)
from app.api.llm_analysis.response.res import (
    Message,
    ServerErrorAnalysisRecord,
    ServerErrorDetectResult,
    ServerErrorQueryResult,
    ServerErrorRecordPage,
)
from app.api.llm_analysis.utils.llm_api_key import CredentialService
from app.core.graph.server_error_analysis_graph import (
    ServerErrorAnalysisResult,
    create_server_error_analysis_graph,
)
from app.core.llm.ollama_client import OllamaClient
from app.core.llm.openai_client import OpenAIClient
from app.core.prompts.prompt_factory import PromptFactory
from config.ConfigManager import ConfigManager


class ServerErrorAnalysisService:
    def __init__(self, db: Session, mcp_manager=None):
        self.db = db
        self.session_repo = LogAnalysisRepository(db)
        self.analysis_repo = ServerErrorAnalysisRepository(db)
        self.mcp_manager = mcp_manager
        self.config = ConfigManager()

    async def query(self, body: PostServerErrorQueryBody) -> ServerErrorQueryResult:
        record = self._get_existing_analysis(body.analysis_id)
        session_id = body.session_id or (record.SESSION_ID if record else None)
        session = self._get_or_create_session(session_id, body.provider, body.model_name)
        trace_id = body.trace_id or (record.TRACE_ID if record else None)

        agent = await self._create_agent(session.PROVIDER, session.MODEL_NAME)
        graph = create_server_error_analysis_graph(self.analysis_repo, agent, self.config)
        result = await graph.ainvoke(
            {
                "mode": "manual",
                "analysis_id": body.analysis_id,
                "session_id": session.SESSION_ID,
                "trace_id": trace_id,
                "time_range": {},
                "user_message": body.message,
                "record_detail": record.DETAIL_JSON if record else {},
                "agent_context": {},
                "analysis_result": None,
            },
            config={"configurable": {"thread_id": session.SESSION_ID}},
        )

        analysis_id = result.get("analysis_id") or body.analysis_id
        updated_record = self.analysis_repo.get_by_id(analysis_id) if analysis_id else None
        analysis_result = result.get("analysis_result") or {}
        message = analysis_result.get("summary") or (updated_record.SUMMARY if updated_record else "")
        return ServerErrorQueryResult(
            message=Message(message_type="ai", message=message),
            analysis=self._to_record(updated_record) if updated_record else None,
        )

    async def detect(self, body: PostServerErrorDetectBody) -> ServerErrorDetectResult:
        analysis_config = self.config.get_server_error_analysis_config()
        end = body.time_range_end or datetime.now(UTC)
        start = body.time_range_start or end - timedelta(minutes=analysis_config["detection_lookback_minutes"])
        session = self._get_or_create_session(None, body.provider, body.model_name)
        candidates = await self._query_5xx_candidates(start, end, body.limit)
        analysis_ids = []

        for candidate in candidates:
            agent = await self._create_agent(session.PROVIDER, session.MODEL_NAME)
            graph = create_server_error_analysis_graph(self.analysis_repo, agent, self.config)
            result = await graph.ainvoke(
                {
                    "mode": "auto",
                    "session_id": session.SESSION_ID,
                    "trace_id": candidate.get("trace_id"),
                    "time_range": {"start": start.isoformat(), "end": end.isoformat()},
                    "user_message": None,
                    "record_detail": candidate,
                    "agent_context": {},
                    "analysis_result": None,
                },
                config={"configurable": {"thread_id": session.SESSION_ID}},
            )
            if result.get("analysis_id"):
                analysis_ids.append(result["analysis_id"])

        return ServerErrorDetectResult(accepted=True, analysis_ids=analysis_ids)

    def list_records(self, params: ServerErrorRecordFilter) -> ServerErrorRecordPage:
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
        record = self.analysis_repo.get_by_id(analysis_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis Not Found")
        return self._to_record(record)

    async def rerun(self, analysis_id: int) -> ServerErrorQueryResult:
        record = self.analysis_repo.get_by_id(analysis_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis Not Found")

        self.analysis_repo.reset_for_rerun(analysis_id)
        body = PostServerErrorQueryBody(
            session_id=record.SESSION_ID,
            analysis_id=record.ID,
            trace_id=record.TRACE_ID,
            message="Re-run HTTP 5xx analysis for this trace.",
        )
        return await self.query(body)

    def _get_existing_analysis(self, analysis_id: int | None):
        if analysis_id is None:
            return None
        record = self.analysis_repo.get_by_id(analysis_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis Not Found")
        return record

    def _get_or_create_session(self, session_id, provider, model_name):
        if session_id:
            session = self.session_repo.get_session_by_id(session_id)
            if not session:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session Not Found")
            return session

        analysis_config = self.config.get_server_error_analysis_config()
        provider_value = self._provider_value(provider or analysis_config["default_provider"])
        session_data = {
            "USER_ID": "1",
            "SESSION_ID": f"server_error_{uuid4().hex}",
            "PROVIDER": provider_value,
            "MODEL_NAME": model_name or analysis_config["default_model_name"],
        }
        return self.session_repo.create_session(session_data)

    async def _create_agent(self, provider, model_name):
        provider_value = self._provider_value(provider)
        credential = CredentialService(repo=self.session_repo).get_provider_credential(provider=provider_value)

        if provider_value == "ollama":
            client = OllamaClient(credential)
            client.setup(model_name)
        elif provider_value == "google":
            client = OpenAIClient(credential, base_url="https://generativelanguage.googleapis.com/v1beta/openai")
            client.setup(model_name)
        elif provider_value == "anthropic":
            client = OpenAIClient(credential, base_url="https://api.anthropic.com/v1/")
            client.setup(model_name)
        else:
            client = OpenAIClient(credential)
            client.setup(model_name)

        Path("checkpoints").mkdir(exist_ok=True)
        conn = await aiosqlite.connect("checkpoints/checkpoints.sqlite", check_same_thread=False)
        checkpointer = AsyncSqliteSaver(conn)
        prompt_service = PromptFactory.create_prompt_service("server_error", self.config)
        system_prompt = prompt_service._build_system_prompt(0)
        tools = self.mcp_manager.get_all_tools() if self.mcp_manager else []
        return client.create_agent_runner(
            tools=tools or [],
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            response_format=ServerErrorAnalysisResult,
        )

    async def _query_5xx_candidates(self, start, end, limit):
        tool = self._find_tool("query_loki_logs")
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="query_loki_logs tool not available",
            )

        result = await tool.ainvoke(
            {
                "query": '{status=~"5.."}',
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": limit,
            }
        )
        return self._extract_candidates(result, limit)

    def _find_tool(self, name):
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_all_tools() or []:
            if getattr(tool, "name", None) == name:
                return tool
        return None

    def _extract_candidates(self, result, limit):
        if isinstance(result, dict) and isinstance(result.get("candidates"), list):
            return result["candidates"][:limit]
        if isinstance(result, list):
            return result[:limit]
        return [
            {
                "trace_id": None,
                "dedup_basis": "trace_id",
                "no_trace_context": {"raw_result_type": type(result).__name__},
            }
        ]

    @staticmethod
    def _provider_value(provider) -> str:
        return getattr(provider, "value", provider)

    @staticmethod
    def _to_record(record):
        return ServerErrorAnalysisRecord(
            id=record.ID,
            trace_id=record.TRACE_ID,
            session_id=record.SESSION_ID,
            status=record.STATUS,
            summary=record.SUMMARY,
            detail=record.DETAIL_JSON,
            created_at=record.CREATED_AT,
            updated_at=record.UPDATED_AT,
        )
