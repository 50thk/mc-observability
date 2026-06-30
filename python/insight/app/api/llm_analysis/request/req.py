from datetime import datetime
from enum import Enum
from typing import Literal

from fastapi import Query
from pydantic import BaseModel, Field, model_validator


class ProviderType(str, Enum):
    openai = "openai"
    openai_compatible = "openai-compatible"
    ollama = "ollama"
    google = "google"
    anthropic = "anthropic"


class APIProviderType(str, Enum):
    openai = "openai"
    openai_compatible = "openai-compatible"
    ollama = "ollama"
    google = "google"
    anthropic = "anthropic"


class GetAPIKeyPath(BaseModel):
    provider: ProviderType


class PostSessionBody(BaseModel):
    provider: ProviderType = Field(..., description="The LLM provider to use", example="openai")
    model_name: str = Field(..., description="The specific model name to use for analysis", example="gpt-5-mini")


class SessionIdPath(BaseModel):
    sessionId: str = Field(description="The session ID for the request.")


class PostQueryBody(BaseModel):
    session_id: str = Field(..., description="The session ID to send the message to", example="session_123")
    message: str = Field(
        ...,
        description="The message or query to send to the LLM for log analysis",
        example="Analyze these error logs and find the root cause",
    )


class GetAPIKeyFilter(BaseModel):
    provider: APIProviderType | None = Field(Query(default=None, description="The LLM provider to use", example="openai"))


class PostAPIKeyBody(BaseModel):
    provider: APIProviderType = Field(..., description="The LLM provider to use")
    api_key: str | None = Field(default=None, min_length=1, description="API key for the LLM provider")
    base_url: str | None = Field(default=None, min_length=1, description="Base URL for endpoint-based providers")

    @model_validator(mode="after")
    def validate_provider_config(self):
        if self.provider == APIProviderType.ollama:
            if not self.base_url:
                raise ValueError("base_url is required for ollama provider")
            return self

        if self.provider == APIProviderType.openai_compatible:
            if not self.api_key:
                raise ValueError("api_key is required for openai-compatible provider")
            if not self.base_url:
                raise ValueError("base_url is required for openai-compatible provider")
            return self

        if not self.api_key:
            raise ValueError("api_key is required for this provider")
        return self


class DeleteAPIKeyFilter(BaseModel):
    provider: APIProviderType = Field(Query(description="The LLM provider to use", example="openai"))


class ServerErrorQueryTimeRange(BaseModel):
    start: datetime | None = Field(default=None)
    end: datetime | None = Field(default=None)
    timezone: str | None = Field(default=None)


class ServerErrorQueryFilters(BaseModel):
    trace_id: str | None = Field(default=None)
    service_name: str | None = Field(default=None)
    node_id: str | None = Field(default=None)
    infra_id: str | None = Field(default=None)
    level: str | None = Field(default=None)
    message: str | None = Field(default=None)


class ServerErrorQueryOptions(BaseModel):
    max_evidence_per_source: int = Field(default=20, ge=1, le=100)


class PostServerErrorQueryBody(BaseModel):
    session_id: str | None = Field(default=None, description="Existing chat session ID")
    trace_id: str | None = Field(default=None, description="Trace ID to analyze")
    query: str | None = Field(default=None, min_length=1, description="Observability RCA query")
    time_range: ServerErrorQueryTimeRange | None = Field(default=None)
    filters: ServerErrorQueryFilters = Field(default_factory=ServerErrorQueryFilters)
    options: ServerErrorQueryOptions = Field(default_factory=ServerErrorQueryOptions)
    provider: ProviderType | None = Field(default=None, description="LLM provider when a new session is needed")
    model_name: str | None = Field(default=None, description="LLM model when a new session is needed")

    @model_validator(mode="after")
    def validate_query(self):
        if not self.query:
            raise ValueError("query is required")
        if self.trace_id and not self.filters.trace_id:
            self.filters.trace_id = self.trace_id
        if self.filters.trace_id and not self.trace_id:
            self.trace_id = self.filters.trace_id
        return self


class ServerErrorRecordFilter(BaseModel):
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "PARTIAL"] | None = Field(default=None)
    from_dt: datetime | None = Field(default=None, alias="from")
    to_dt: datetime | None = Field(default=None, alias="to")
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


class ServerErrorAnalysisIdPath(BaseModel):
    analysis_id: int = Field(..., ge=1)
