from datetime import datetime
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.graph.rca.models import IncidentScope


class ConnectionProviderType(str, Enum):
    openai = "openai"
    ollama = "ollama"


def is_official_openai_base_url(base_url: str | None) -> bool:
    return base_url is None or urlparse(base_url).hostname == "api.openai.com"


class PostConnectionBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    provider: ConnectionProviderType
    base_url: str | None = Field(default=None, min_length=1, pattern=r"^https?://")
    api_key: str | None = Field(default=None, min_length=1)
    default_model: str | None = Field(default=None, min_length=1, max_length=255)
    enabled: bool = True
    is_default: bool = False

    @model_validator(mode="after")
    def validate_connection(self):
        if self.provider == ConnectionProviderType.ollama and not self.base_url:
            raise ValueError("base_url is required for ollama provider")
        if (
            self.provider == ConnectionProviderType.openai
            and is_official_openai_base_url(self.base_url)
            and not self.api_key
        ):
            raise ValueError("api_key is required for the default OpenAI endpoint")
        if self.is_default and not self.default_model:
            raise ValueError("default_model is required for the default connection")
        if self.is_default and not self.enabled:
            raise ValueError("the default connection must be enabled")
        return self


class SetDefaultConnectionBody(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=255)


class PatchConnectionBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    provider: ConnectionProviderType | None = None
    base_url: str | None = Field(default=None, min_length=1, pattern=r"^https?://")
    api_key: str | None = Field(default=None, min_length=1)
    default_model: str | None = Field(default=None, min_length=1, max_length=255)
    enabled: bool | None = None

    @model_validator(mode="after")
    def require_update(self):
        if not self.model_fields_set:
            raise ValueError("at least one connection field is required")
        return self


class PostSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_type: Literal["log", "alert", "rca"] = "log"
    connection_id: int | None = Field(default=None, ge=1)
    model_name: str | None = Field(default=None, min_length=1, max_length=255)


class SessionIdPath(BaseModel):
    sessionId: str = Field(description="The session ID for the request.")


class PostQueryBody(BaseModel):
    session_id: str | None = Field(default=None, description="Existing session ID")
    connection_id: int | None = Field(default=None, ge=1)
    model_name: str | None = Field(default=None, min_length=1, max_length=255)
    message: str = Field(
        ...,
        description="The message or query to send to the LLM for log analysis",
        example="Analyze these error logs and find the root cause",
    )

    @model_validator(mode="after")
    def validate_session_override(self):
        if self.session_id and (self.connection_id is not None or self.model_name is not None):
            raise ValueError("connection_id and model_name cannot override an existing session")
        return self


class PostRcaQueryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Existing chat session ID",
    )
    connection_id: int | None = Field(default=None, ge=1, description="LLM connection for a new session")
    query: str | None = Field(
        default=None,
        max_length=8000,
        description="Natural-language RCA request",
    )
    scope: IncidentScope = Field(default_factory=IncidentScope)
    filters: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="LLM model when a new session is needed",
    )

    @model_validator(mode="after")
    def validate_request(self):
        if self.session_id and (self.connection_id is not None or self.model_name is not None):
            raise ValueError("connection_id and model_name cannot override an existing session")
        if not self.scope.trace_id and self.scope.time_range.start is None:
            raise ValueError("trace_id or bounded time range is required")
        return self


class RcaRecordFilter(BaseModel):
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "PARTIAL"] | None = Field(default=None)
    from_dt: datetime | None = Field(default=None, alias="from")
    to_dt: datetime | None = Field(default=None, alias="to")
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


class RcaAnalysisIdPath(BaseModel):
    analysis_id: int = Field(..., ge=1)


MIN_SCHEDULE_INTERVAL_MINUTES = 5
MAX_SCHEDULE_INTERVAL_MINUTES = 10_080


def _validated_rca_request(request: dict) -> dict:
    """Check the stored body against the live RCA contract without rewriting it.

    Validation must not become a transform: the schedule stores exactly what the user
    sent, so a later contract change never silently alters a saved request.
    """
    PostRcaQueryBody.model_validate(request)
    return request


class PostRcaScheduleBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    enabled: bool = True
    interval_minutes: int = Field(
        ...,
        ge=MIN_SCHEDULE_INTERVAL_MINUTES,
        le=MAX_SCHEDULE_INTERVAL_MINUTES,
    )
    request: dict

    @model_validator(mode="after")
    def validate_body(self):
        if not self.name.strip():
            raise ValueError("name must not be blank")
        _validated_rca_request(self.request)
        return self


class PatchRcaScheduleBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    enabled: bool | None = None
    interval_minutes: int | None = Field(
        default=None,
        ge=MIN_SCHEDULE_INTERVAL_MINUTES,
        le=MAX_SCHEDULE_INTERVAL_MINUTES,
    )
    request: dict | None = None

    @model_validator(mode="after")
    def validate_body(self):
        if not self.model_fields_set:
            raise ValueError("at least one schedule field is required")
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be blank")
        if self.request is not None:
            _validated_rca_request(self.request)
        return self


class RcaScheduleIdPath(BaseModel):
    schedule_id: int = Field(..., ge=1)
