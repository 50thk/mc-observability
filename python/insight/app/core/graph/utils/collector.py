import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from app.core.graph.server_error_analysis_graph import EvidenceRef, IncidentContext, SourceEvidence

BLOCKED_TOOL_PREFIXES = ("create_", "update_", "install_", "add_")
BLOCKED_TOOL_NAMES = {"analyze_loki_labels"}
INFLUX_TAG_ALLOWLIST = {"ns_id", "infra_id", "node_id"}
# Valid Loki/InfluxDB identifier (label/field) token; guards config-supplied names from query injection.
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_MUTATING_INFLUX_RE = re.compile(
    r"\b(ALTER|CREATE|DELETE|DROP|GRANT|INSERT|INTO|LOAD|REVOKE|TRUNCATE|UPDATE)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ToolCallResult:
    ok: bool
    payload: Any = None
    preview: str = ""
    limitation: str | None = None
    tool_name: str | None = None
    mcp_name: str | None = None
    args: dict[str, Any] | None = None


def tool_call_detail(result: ToolCallResult) -> dict[str, Any]:
    detail = {"tool_name": result.tool_name, "args": result.args}
    if result.mcp_name:
        detail["mcp_name"] = result.mcp_name
    return detail


class McpToolRunner:
    def __init__(self, mcp_manager, *, timeout_seconds: float = 20, max_preview_chars: int = 8000):
        self.mcp_manager = mcp_manager
        self.timeout_seconds = timeout_seconds
        self.max_preview_chars = max_preview_chars

    def list_tool_names(self, mcp_name: str | None = None) -> list[str]:
        tools = self._tools(mcp_name)
        return [getattr(tool, "name", "") for tool in tools if getattr(tool, "name", None)]

    async def call_tool(self, name: str, args: dict[str, Any], *, mcp_name: str | None = None) -> ToolCallResult:
        if name in BLOCKED_TOOL_NAMES or name.startswith(BLOCKED_TOOL_PREFIXES):
            return ToolCallResult(
                False,
                limitation=f"tool {name} refused by read-only policy",
                tool_name=name,
                mcp_name=mcp_name,
                args=args,
            )
        if limitation := self.scoped_lookup_limitation(mcp_name):
            return ToolCallResult(False, limitation=limitation, tool_name=name, mcp_name=mcp_name, args=args)
        tool = self._find_tool(name, mcp_name)
        if tool is None:
            return ToolCallResult(
                False, limitation=f"tool {name} not available", tool_name=name, mcp_name=mcp_name, args=args
            )
        try:
            payload = await asyncio.wait_for(tool.ainvoke(args), timeout=self.timeout_seconds)
        except TimeoutError:
            return ToolCallResult(False, limitation=f"tool {name} timed out", tool_name=name, mcp_name=mcp_name, args=args)
        except Exception as exc:
            return ToolCallResult(
                False, limitation=f"tool {name} failed: {exc}", tool_name=name, mcp_name=mcp_name, args=args
            )
        preview = str(payload)[: self.max_preview_chars]
        return ToolCallResult(True, payload=payload, preview=preview, tool_name=name, mcp_name=mcp_name, args=args)

    def scoped_lookup_limitation(self, mcp_name: str | None) -> str | None:
        if not mcp_name or self.mcp_manager is None or hasattr(self.mcp_manager, "get_tools_for_mcp"):
            return None
        return f"scoped MCP lookup for {mcp_name} requires get_tools_for_mcp"

    def _find_tool(self, name: str, mcp_name: str | None = None):
        for tool in self._tools(mcp_name):
            if getattr(tool, "name", None) == name:
                return tool
        return None

    def _tools(self, mcp_name: str | None):
        if self.mcp_manager is None:
            return []
        if mcp_name:
            if hasattr(self.mcp_manager, "get_tools_for_mcp"):
                return self.mcp_manager.get_tools_for_mcp(mcp_name) or []
            return []
        if hasattr(self.mcp_manager, "get_all_tools"):
            return self.mcp_manager.get_all_tools() or []
        return []


class BaseEvidenceCollector:
    def __init__(self, runner: McpToolRunner):
        self.runner = runner

    async def collect(self, incident: IncidentContext, **kwargs: Any) -> SourceEvidence:
        raise NotImplementedError

    @staticmethod
    def _has_payload(payload: Any) -> bool:
        payload = _parse_tool_payload(payload)
        if payload is None:
            return False
        if payload == "":
            return False
        if isinstance(payload, (list, tuple, set, dict)) and not payload:
            return False
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("result", payload))
            if data in ([], {}, "", None):
                return False
        return True

    @staticmethod
    def _window(incident: IncidentContext) -> tuple[str, str]:
        end = incident.time_range_end or datetime.now(UTC)
        start = incident.time_range_start or end - timedelta(minutes=15)
        return start.isoformat(), end.isoformat()


class TraceEvidenceCollector(BaseEvidenceCollector):
    async def collect(
        self,
        incident: IncidentContext,
        *,
        trace_id: str | None = None,
        traceql: str | None = None,
        limit: int | None = None,
    ) -> SourceEvidence:
        trace_id = trace_id or incident.trace_id
        if trace_id and traceql:
            return SourceEvidence(
                source="tempo", status="FAILED", limitations=["provide either trace_id or traceql, not both"]
            )
        if trace_id:
            return await self._collect_by_trace_id(trace_id)
        if not traceql:
            return SourceEvidence(source="tempo", status="SKIPPED", limitations=["trace_id or traceql is required"])
        if error := _guard_traceql(traceql):
            return SourceEvidence(source="tempo", status="FAILED", executed_query=traceql, limitations=[error])
        return await self._search_traces(incident, traceql=traceql, limit=limit)

    async def _collect_by_trace_id(self, trace_id: str) -> SourceEvidence:
        tool_name, limitation = self._get_trace_tool_name()
        if not tool_name:
            return SourceEvidence(
                source="tempo", status="FAILED", limitations=[limitation or "no Tempo trace tool discovered"]
            )
        result = await self.runner.call_tool(tool_name, {"trace_id": trace_id}, mcp_name="tempo")
        if not result.ok:
            return SourceEvidence(
                source="tempo", status="FAILED", limitations=[result.limitation or "trace query failed"]
            )
        if not self._has_trace_payload(result.payload):
            return SourceEvidence(
                source="tempo", status="NO_DATA", tool_calls=[tool_call_detail(result)], summary="no trace data"
            )
        return SourceEvidence(
            source="tempo",
            status="OK",
            summary=result.preview,
            tool_calls=[tool_call_detail(result)],
            refs=[EvidenceRef(id="trace-1", source="tempo", summary=result.preview, tool_name=tool_name)],
        )

    async def _search_traces(
        self, incident: IncidentContext, *, traceql: str, limit: int | None = None
    ) -> SourceEvidence:
        tool_name, limitation = self._search_tool_name()
        if not tool_name:
            return SourceEvidence(
                source="tempo",
                status="FAILED",
                limitations=[limitation or "no Tempo trace search tool discovered"],
            )
        start, end = self._window(incident)
        args = {"query": traceql, "start": start, "end": end}
        if limit is not None:
            args["limit"] = limit
        result = await self.runner.call_tool(tool_name, args, mcp_name="tempo")
        if not result.ok:
            return SourceEvidence(
                source="tempo",
                status="FAILED",
                executed_query=traceql,
                limitations=[result.limitation or "trace search failed"],
            )
        if not self._has_trace_payload(result.payload):
            return SourceEvidence(
                source="tempo",
                status="NO_DATA",
                executed_query=traceql,
                tool_calls=[tool_call_detail(result)],
                summary="no trace data",
            )
        trace_ids = _extract_trace_ids(result.payload)
        return SourceEvidence(
            source="tempo",
            status="OK",
            summary=result.preview,
            executed_query=traceql,
            filters_applied=self._filters(incident),
            tool_calls=[tool_call_detail(result)],
            refs=[
                EvidenceRef(
                    id="trace-search-1", source="tempo", summary=result.preview, tool_name=tool_name, query=traceql
                )
            ],
            observations=[f"traceql-search found trace_id candidate: {trace_ids[0]}"] if trace_ids else [],
            raw_evidence={"trace_ids": trace_ids} if trace_ids else {},
        )

    @staticmethod
    def _has_trace_payload(payload: Any) -> bool:
        payload = _parse_tool_payload(payload)
        if isinstance(payload, dict) and "traces" in payload:
            return bool(payload.get("traces"))
        return BaseEvidenceCollector._has_payload(payload)

    @staticmethod
    def _filters(incident: IncidentContext) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "service_name": incident.service_name,
                "node_id": incident.node_id,
                "infra_id": incident.infra_id,
            }.items()
            if value
        }

    def _get_trace_tool_name(self) -> tuple[str | None, str | None]:
        if limitation := self.runner.scoped_lookup_limitation("tempo"):
            return None, limitation
        return ("get-trace", None) if "get-trace" in self.runner.list_tool_names("tempo") else (None, None)

    def _search_tool_name(self) -> tuple[str | None, str | None]:
        if limitation := self.runner.scoped_lookup_limitation("tempo"):
            return None, limitation
        return (
            ("traceql-search", None)
            if "traceql-search" in self.runner.list_tool_names("tempo")
            else (None, None)
        )


class LogEvidenceCollector(BaseEvidenceCollector):
    def __init__(
        self,
        runner: McpToolRunner,
        *,
        datasource_uid: str | None = None,
        limit: int = 20,
        schema_config: dict[str, Any] | None = None,
    ):
        super().__init__(runner)
        self.datasource_uid = datasource_uid
        self.limit = limit
        self.schema = LokiSchema.from_config(schema_config)

    async def collect(
        self,
        incident: IncidentContext,
        *,
        logql: str,
        limit: int | None = None,
    ) -> SourceEvidence:
        datasource_limitation = None
        if self.datasource_uid:
            datasource_uid = self.datasource_uid
        else:
            datasource_uid, datasource_limitation = await self._resolve_datasource_uid()
        if not datasource_uid:
            return SourceEvidence(
                source="loki",
                status="FAILED",
                limitations=[datasource_limitation or "Loki datasource uid unavailable"],
            )
        if error := _guard_logql(logql):
            return SourceEvidence(source="loki", status="FAILED", executed_query=logql, limitations=[error])
        start, end = self._window(incident)
        limit = limit or incident.max_evidence_per_source or self.limit
        args = {
            "datasourceUid": datasource_uid,
            "logql": logql,
            "startRfc3339": start,
            "endRfc3339": end,
            "limit": limit,
        }
        result = await self.runner.call_tool("query_loki_logs", args, mcp_name="grafana")
        if not result.ok:
            return SourceEvidence(
                source="loki",
                status="FAILED",
                executed_query=logql,
                limitations=[result.limitation or "log query failed"],
            )
        limitations = []
        observations = []
        if not (incident.trace_id or incident.service_name or incident.node_id or incident.infra_id):
            observations.append("log query is broad; no trace_id/service/node/infra scope")
        if not self._has_payload(result.payload):
            return SourceEvidence(
                source="loki",
                status="NO_DATA",
                executed_query=logql,
                tool_calls=[tool_call_detail(result)],
                observations=observations,
            )
        status = "OK"
        if _is_loki_result_truncated(result.payload):
            status = "PARTIAL"
            limitations.append("log result was truncated")
        trace_ids = _extract_trace_ids(result.payload)
        if trace_ids:
            observations.append(f"derived trace_id candidate from Loki logs: {trace_ids[0]}")
        raw_evidence = {"trace_ids": trace_ids} if trace_ids else {}
        return SourceEvidence(
            source="loki",
            status=status,
            summary=result.preview,
            executed_query=logql,
            filters_applied=self._filters(incident),
            tool_calls=[tool_call_detail(result)],
            refs=[
                EvidenceRef(id="log-1", source="loki", summary=result.preview, tool_name="query_loki_logs", query=logql)
            ],
            observations=observations,
            raw_evidence=raw_evidence,
            limitations=limitations,
        )

    async def _resolve_datasource_uid(self) -> tuple[str | None, str | None]:
        result = await self.runner.call_tool("list_datasources", {"type": "loki", "limit": 100}, mcp_name="grafana")
        if not result.ok:
            return None, result.limitation
        payload = _parse_tool_payload(result.payload)
        if isinstance(payload, dict):
            datasources = payload.get("datasources") or payload.get("data") or []
        else:
            datasources = payload
        if not isinstance(datasources, list):
            return None, None
        loki = [
            item for item in datasources if isinstance(item, dict) and item.get("type") == "loki" and item.get("uid")
        ]
        default = next((item for item in loki if item.get("isDefault")), None)
        chosen = default or (loki[0] if loki else None)
        return (chosen.get("uid"), None) if chosen else (None, None)

    @staticmethod
    def _filters(incident: IncidentContext) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "trace_id": incident.trace_id,
                "service_name": incident.service_name,
                "node_id": incident.node_id,
                "infra_id": incident.infra_id,
                "level": incident.level,
                "message": incident.message_filter,
            }.items()
            if value
        }


class MetricEvidenceCollector(BaseEvidenceCollector):
    def __init__(
        self,
        runner: McpToolRunner,
        *,
        database: str = "insight",
        schema_config: dict[str, Any] | None = None,
    ):
        super().__init__(runner)
        self.config = InfluxSchemaConfig.from_config(schema_config, default_database=database)

    async def collect(self, incident: IncidentContext, *, influxql: str) -> SourceEvidence:
        if error := _guard_influxql(influxql):
            return SourceEvidence(source="influxdb", status="FAILED", executed_query=influxql, limitations=[error])
        result = await self.runner.call_tool("execute_influxql", {"influxql_query": influxql}, mcp_name="influxdb")
        if not result.ok:
            return SourceEvidence(
                source="influxdb",
                status="FAILED",
                executed_query=influxql,
                limitations=[result.limitation or "metric query failed"],
            )
        if payload_error := self._metric_payload_error(result.payload):
            return SourceEvidence(
                source="influxdb",
                status="FAILED",
                executed_query=influxql,
                tool_calls=[tool_call_detail(result)],
                limitations=[payload_error],
            )
        if not self._has_metric_payload(result.payload):
            return SourceEvidence(
                source="influxdb", status="NO_DATA", executed_query=influxql, tool_calls=[tool_call_detail(result)]
            )
        return SourceEvidence(
            source="influxdb",
            status="OK",
            summary=result.preview,
            executed_query=influxql,
            tool_calls=[tool_call_detail(result)],
            refs=[
                EvidenceRef(
                    id="metric-1",
                    source="influxdb",
                    summary=result.preview,
                    tool_name="execute_influxql",
                    query=influxql,
                )
            ],
        )

    @staticmethod
    def _has_metric_payload(payload: Any) -> bool:
        payload = _parse_tool_payload(payload)
        if isinstance(payload, dict):
            if "series" in payload:
                series = payload.get("series")
                if not isinstance(series, list):
                    return False
                return any(MetricEvidenceCollector._series_has_values(item) for item in series)
            if "results" in payload:
                results = payload.get("results")
                if not isinstance(results, list):
                    return False
                data_keys = {"series", "results", "data", "result"}
                return any(
                    MetricEvidenceCollector._has_metric_payload(item)
                    for item in results
                    if not isinstance(item, dict) or data_keys.intersection(item)
                )
            if "data" in payload:
                return MetricEvidenceCollector._has_metric_payload(payload.get("data"))
            if "result" in payload:
                return MetricEvidenceCollector._has_metric_payload(payload.get("result"))
        if isinstance(payload, list):
            return any(MetricEvidenceCollector._has_metric_payload(item) for item in payload)
        return False

    @staticmethod
    def _series_has_values(series: Any) -> bool:
        if not isinstance(series, dict):
            return False
        values = series.get("values")
        if not isinstance(values, list):
            return False
        columns = series.get("columns")
        metric_indexes = None
        if isinstance(columns, list) and columns:
            metric_indexes = [idx for idx, column in enumerate(columns) if str(column).lower() != "time"]
        return any(MetricEvidenceCollector._row_has_metric_value(row, metric_indexes) for row in values)

    @staticmethod
    def _row_has_metric_value(row: Any, metric_indexes: list[int] | None) -> bool:
        if not isinstance(row, (list, tuple)):
            return False
        if metric_indexes is None:
            cells = row[1:]
        else:
            cells = [row[idx] for idx in metric_indexes if idx < len(row)]
        return any(MetricEvidenceCollector._metric_value_present(cell) for cell in cells)

    @staticmethod
    def _metric_value_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return True

    @staticmethod
    def _metric_payload_error(payload: Any) -> str | None:
        payload = _parse_tool_payload(payload)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
            for key in ("results", "data", "result"):
                if key in payload:
                    nested = MetricEvidenceCollector._metric_payload_error(payload.get(key))
                    if nested:
                        return nested
        if isinstance(payload, list):
            for item in payload:
                nested = MetricEvidenceCollector._metric_payload_error(item)
                if nested:
                    return nested
        return None


def _query_guard(query: str | None, dialect: str) -> str | None:
    if not isinstance(query, str) or not query.strip():
        return f"{dialect} query is required"
    stripped = query.strip()
    if len(stripped) > 4000:
        return f"{dialect} query is too long"
    if "```" in stripped:
        return f"{dialect} query must not include markdown fences"
    if ";" in stripped.rstrip(";"):
        return f"{dialect} query must contain one statement"
    return None


def _guard_logql(logql: str | None) -> str | None:
    if error := _query_guard(logql, "LogQL"):
        return error
    stripped = logql.strip()
    if not stripped.startswith("{"):
        return "LogQL must start with a stream selector"
    if re.match(r"^\{\s*\}", stripped):
        return 'LogQL selector must not be empty; use a scoped selector such as {service=~".+"}'
    return None


def _guard_traceql(traceql: str | None) -> str | None:
    if error := _query_guard(traceql, "TraceQL"):
        return error
    stripped = traceql.strip()
    if not stripped.startswith("{"):
        return "TraceQL must start with a selector"
    if re.match(r"^\{\s*\}$", stripped):
        return "TraceQL selector must not be empty"
    return None


def _guard_influxql(influxql: str | None) -> str | None:
    if error := _query_guard(influxql, "InfluxQL"):
        return error
    stripped = influxql.strip()
    if not stripped.upper().startswith("SELECT"):
        return "InfluxQL must be read-only SELECT"
    if _MUTATING_INFLUX_RE.search(stripped):
        return "InfluxQL must be read-only"
    if not re.search(r"\btime\b\s*(?:=|>=|<=|>|<)", stripped, re.IGNORECASE):
        return "InfluxQL must include a time predicate"
    return None


def _extract_trace_ids(payload: Any) -> list[str]:
    found: list[str] = []
    _collect_trace_ids(_parse_tool_payload(payload), found)
    return list(dict.fromkeys(found))[:5]


def _is_loki_result_truncated(payload: Any) -> bool:
    return _find_bool_key(_parse_tool_payload(payload), "resultsTruncated")


def _find_bool_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if value.get(key) is True:
            return True
        return any(_find_bool_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_find_bool_key(item, key) for item in value)
    return False


def _collect_trace_ids(value: Any, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in {"trace_id", "traceID", "traceId", "traceid"} and isinstance(item, str) and item.strip():
                found.append(item.strip())
            _collect_trace_ids(item, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_trace_ids(item, found)
        return
    if isinstance(value, str):
        found.extend(re.findall(r'(?:trace_id|traceID|traceId|traceid)[=:"\s]+([A-Za-z0-9._:-]{8,})', value))


class LokiSchema(BaseModel):
    labels: dict[str, str] = Field(
        default_factory=lambda: {
            "ns_id": "NS_ID",
            "infra_id": "INFRA_ID",
            "node_id": "NODE_ID",
            "host": "host",
            "service_name": "service",
            "level": "level",
            "source": "source",
        }
    )
    parsed_fields: dict[str, str] = Field(default_factory=lambda: {"trace_id": "trace_id", "level": "level"})

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "LokiSchema":
        config = config or {}
        defaults = cls()
        allowed_labels = set(defaults.labels) | {"trace_id"}
        labels = dict(defaults.labels)
        labels.update(
            {
                key: value
                for key, value in (config.get("labels") or {}).items()
                if key in allowed_labels and isinstance(value, str) and _IDENTIFIER_RE.match(value)
            }
        )
        parsed = dict(defaults.parsed_fields)
        parsed.update(
            {
                key: value
                for key, value in (config.get("parsed_fields") or {}).items()
                if key in {"trace_id", "level"} and isinstance(value, str) and _IDENTIFIER_RE.match(value)
            }
        )
        return cls(labels=labels, parsed_fields=parsed)


class InfluxSchemaConfig(BaseModel):
    database: str = "insight"
    retention_policy: str = "autogen"
    measurements: list[str] = Field(
        default_factory=lambda: ["cpu", "disk", "diskio", "mem", "net", "processes", "procstat", "swap", "system"]
    )
    fields: list[str] = Field(default_factory=lambda: ["usage_idle", "used_percent", "load1", "value"])
    tags: set[str] = Field(default_factory=lambda: set(INFLUX_TAG_ALLOWLIST))

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, *, default_database: str) -> "InfluxSchemaConfig":
        config = config or {}
        defaults = cls(database=default_database)
        measurements = config.get("measurements") or defaults.measurements
        if isinstance(measurements, str):
            measurements = [measurements]
        fields = config.get("fields") or defaults.fields
        if isinstance(fields, str):
            fields = [fields]
        tags = config.get("tags") or list(defaults.tags)
        return cls(
            database=config.get("database") or default_database,
            retention_policy=config.get("retention_policy") or "autogen",
            measurements=[item for item in measurements if isinstance(item, str)] or defaults.measurements,
            fields=[item for item in fields if isinstance(item, str)] or defaults.fields,
            tags={item for item in tags if isinstance(item, str)},
        )


def _parse_tool_payload(payload: Any) -> Any:
    """Normalize MCP content-block payloads into the JSON object they carry when possible."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if isinstance(payload, list):
        parsed_items = []
        changed = False
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parsed_items.append(_parse_tool_payload(item["text"]))
                changed = True
            else:
                parsed_items.append(_parse_tool_payload(item))
        if changed and len(parsed_items) == 1:
            return parsed_items[0]
        return parsed_items
    return payload
