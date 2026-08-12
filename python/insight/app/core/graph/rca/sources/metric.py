"""Metric source: InfluxDB (Telegraf measurements) through its MCP.

The agent never writes InfluxQL. It picks a measurement, fields, an aggregation, tag
filters and grouping from the static catalog in SOURCE_SPECS; code validates every value
against that catalog, assembles the query, and injects the database and time window.
"""

import re
from typing import Any

from langchain_core.tools import StructuredTool

from ..specs import SOURCE_SPECS
from .base import (
    SourceContext,
    SourceUnavailableError,
    baseline_window,
    clamp_limit,
    escape_identifier,
    escape_influx_value,
    incident_window,
    rejected,
    required_tool,
    tabular_values,
)

_SPEC = SOURCE_SPECS["metric"]
_CATALOG: dict[str, dict[str, tuple[str, ...]]] = _SPEC["catalog"]
AGGREGATIONS = ("mean", "max", "min", "last", "count")
_INTERVAL = re.compile(r"\d+(?:ns|u|µ|ms|s|m|h|d|w)")
_MAX_FIELDS = 8


def build_tools(context: SourceContext) -> list[StructuredTool]:
    tag_values_tool = required_tool(context, "get_tag_values")
    query_tool = required_tool(context, "execute_influxql")
    database = str(context.datasources.get("influx_database") or "").strip()
    if not database:
        raise SourceUnavailableError("influx_database_not_configured")
    incident = incident_window(context.scope)
    baseline = baseline_window(context.scope)

    async def query_metrics(
        measurement: str,
        fields: list[str],
        aggregation: str = "mean",
        tag_filters: dict[str, Any] | None = None,
        group_by: list[str] | None = None,
        limit: int = _SPEC["default_limit"],
        compare_baseline: bool = False,
    ) -> Any:
        measurement = measurement.strip()
        entry = _CATALOG.get(measurement)
        if entry is None:
            return rejected("unknown_measurement", allowed_measurements=sorted(_CATALOG))
        wanted = list(dict.fromkeys(str(field).strip() for field in fields or []))
        if not wanted:
            return rejected("fields_required", allowed_fields=list(entry["fields"]))
        if len(wanted) > _MAX_FIELDS:
            return rejected("too_many_fields", max_fields=_MAX_FIELDS)
        if unknown := [field for field in wanted if field not in entry["fields"]]:
            return rejected("unknown_field", unknown=unknown, allowed_fields=list(entry["fields"]))
        aggregation = aggregation.strip().lower()
        if aggregation not in AGGREGATIONS:
            return rejected("unknown_aggregation", allowed_aggregations=list(AGGREGATIONS))
        filters: dict[str, str] = {}
        for key, value in (tag_filters or {}).items():
            if key not in entry["tag_keys"]:
                return rejected("unknown_tag_key", unknown=key, allowed_tag_keys=list(entry["tag_keys"]))
            if not isinstance(value, (str, int, float, bool)):
                return rejected("invalid_tag_value", tag_key=key)
            filters[key] = str(value)
        interval: str | None = None
        group_tags: list[str] = []
        for item in group_by or []:
            item = str(item).strip()
            if item in entry["tag_keys"]:
                if item not in group_tags:
                    group_tags.append(item)
            elif _INTERVAL.fullmatch(item) and interval is None:
                interval = item
            else:
                return rejected(
                    "unknown_group_by",
                    unknown=item,
                    allowed_tag_keys=list(entry["tag_keys"]),
                    hint="use a catalog tag key or at most one interval such as 1m or 5m",
                )
        bounded = clamp_limit(limit, _SPEC)
        if bounded is None:
            return rejected("invalid_limit", min=1, max=_SPEC["max_limit"])

        influxql = build_influxql(
            measurement,
            wanted,
            aggregation,
            filters,
            incident,
            interval=interval,
            group_tags=group_tags,
            limit=bounded,
        )
        public_args = {
            "measurement": measurement,
            "fields": wanted,
            "aggregation": aggregation,
            "tag_filters": filters,
            "group_by": [*([interval] if interval else []), *group_tags],
            "limit": bounded,
            "compare_baseline": bool(compare_baseline),
        }

        async def execute():
            result: dict[str, Any] = {
                "influxql": influxql,
                "incident": await context.invoke(
                    query_tool,
                    "execute_influxql",
                    {"influxql_query": influxql, "database_name": database},
                ),
            }
            if compare_baseline:
                baseline_influxql = build_influxql(
                    measurement,
                    wanted,
                    aggregation,
                    filters,
                    baseline,
                    interval=interval,
                    group_tags=group_tags,
                    limit=bounded,
                )
                result["baseline_influxql"] = baseline_influxql
                result["baseline"] = await context.invoke(
                    query_tool,
                    "execute_influxql",
                    {"influxql_query": baseline_influxql, "database_name": database},
                )
            return result

        return await context.run(
            name="query_metrics",
            args=public_args,
            evidence_query=True,
            execute=execute,
        )

    async def get_tag_values(measurement: str, tag_key: str) -> Any:
        measurement = measurement.strip()
        tag_key = tag_key.strip()
        entry = _CATALOG.get(measurement)
        if entry is None:
            return rejected("unknown_measurement", allowed_measurements=sorted(_CATALOG))
        if tag_key not in entry["tag_keys"]:
            return rejected("unknown_tag_key", unknown=tag_key, allowed_tag_keys=list(entry["tag_keys"]))
        backend_args = {"database_name": database, "measurement_name": measurement, "tag_key": tag_key}

        async def execute():
            raw = await context.invoke(tag_values_tool, "get_tag_values", backend_args)
            return {"measurement": measurement, "tag_key": tag_key, "values": tabular_values(raw)}

        return await context.run(
            name="get_tag_values",
            args={"measurement": measurement, "tag_key": tag_key},
            evidence_query=False,
            execute=execute,
        )

    return [
        StructuredTool.from_function(
            coroutine=query_metrics,
            name="query_metrics",
            description=(
                "Aggregate one Telegraf measurement over the incident window. measurement, fields, tag_filters "
                "and group_by must come from the metric catalog in your instructions; aggregation is one of "
                f"{', '.join(AGGREGATIONS)}. group_by accepts catalog tag keys and at most one interval such as "
                "1m. compare_baseline=true also runs the same query over the equal-length window immediately "
                f"before the incident. limit defaults to {_SPEC['default_limit']} and is capped at "
                f"{_SPEC['max_limit']}. The database and time window are fixed by code."
            ),
        ),
        StructuredTool.from_function(
            coroutine=get_tag_values,
            name="get_tag_values",
            description=(
                "List the values one catalog tag key (node_id, device, interface, pid, ...) takes in a "
                "measurement, to choose exact tag_filters for query_metrics."
            ),
        ),
    ]


def build_influxql(
    measurement: str,
    fields: list[str],
    aggregation: str,
    tag_filters: dict[str, str],
    window: tuple[str, str],
    *,
    interval: str | None,
    group_tags: list[str],
    limit: int,
) -> str:
    start, end = window
    clauses = [f"time >= '{start}'", f"time <= '{end}'"]
    clauses.extend(
        f'"{escape_identifier(key)}" = \'{escape_influx_value(value)}\'' for key, value in sorted(tag_filters.items())
    )
    selects = ", ".join(f'{aggregation.upper()}("{escape_identifier(field)}")' for field in fields)
    groups = [*([f"time({interval})"] if interval else []), *(f'"{escape_identifier(tag)}"' for tag in group_tags)]
    group_clause = f" GROUP BY {', '.join(groups)}" if groups else ""
    return f'SELECT {selects} FROM "{escape_identifier(measurement)}" WHERE {" AND ".join(clauses)}{group_clause} LIMIT {limit}'
