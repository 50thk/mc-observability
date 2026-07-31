_LOKI_DISCOVERY = ("list_datasources", "list_loki_label_names", "list_loki_label_values")
_INFLUX_DISCOVERY = (
    "list_influxdb_databases",
    "list_measurements",
    "get_measurement_schema",
    "get_tag_values",
)

# Per-capability usage notes injected into the collector agent's system prompt. Code owns
# the query text, so these do not teach a query language — they teach which parameters to
# pick, in what order, and what to do when a call comes back empty. A one-line
# `when_to_use` was enough for planning but left the collector guessing at runtime.
_LOGS_SEARCH_INSTRUCTIONS = """
This capability answers: what did the service log during the window, and which lines
describe the failure.
Order of work: read the discovery block for the labels and verified filters that already
apply, then run one broad query, then one narrower query using terms taken from what the
first query returned.
Choosing terms: use literal substrings that would appear in a log line (an error name, an
exception class, a status code, a host). Do not pass regular expressions, field names, or
speculative words that were not seen in the incident description or in earlier output.
When a query returns nothing: drop the most specific term and try once more. If it is
still empty, report status NO_DATA and say what you searched for. Never infer a cause from
the absence of logs.
"""

_LOGS_VOLUME_INSTRUCTIONS = """
This capability answers: how much was logged in the window and by which streams.
Counts of zero are a real measurement, not a missing answer — report them as data.
"""

_TRACES_GET_INSTRUCTIONS = """
This capability answers: where a single known request spent its time and which span failed.
The span table is already flattened and sorted with error spans first, then the slowest
ones. Read it directly; only inspect spilled evidence when the table is not enough.
"""

_TRACES_SEARCH_INSTRUCTIONS = """
This capability answers: which requests failed or were slow in the window, and on which
service or endpoint.
Order of work: start from the widest search the scope allows, read what came back, then
narrow with one more search only if the first result is too broad to interpret.
Choosing parameters: set error_only when the question is about failures, errors, or status
codes. Set min_duration_ms only when the question is about latency, and keep it well below
the window length so that normal requests are still excluded rather than everything.
Setting both at once usually returns nothing — prefer one.
When a search returns nothing: relax one constraint and try once more. If it is still
empty, report status NO_DATA rather than concluding that nothing failed.
"""

_METRICS_INSTRUCTIONS = """
This capability answers: whether an infrastructure signal moved during the window compared
with the immediately preceding baseline.
Order of work: read the discovery block for the measurements and fields that actually
exist, pick the one that matches the reported symptom, then query it once.
Choosing parameters: measurement and field must come verbatim from discovery. Pick the
aggregation that suits the question — max for saturation and spikes, mean for sustained
load, last for the final state.
When the result carries no rows: the window has no data for that series. Report status
NO_DATA and say which measurement and field were queried.
"""


CAPABILITY_SPECS = {
    "logs.search": {
        "source": "log",
        "mcp": "grafana",
        "required_tools": (*_LOKI_DISCOVERY, "query_loki_logs"),
        "optional_tools": (),
        "when_to_use": "Search bounded logs for terms or representative records related to the reported behavior.",
        "llm_instructions": _LOGS_SEARCH_INSTRUCTIONS,
        "timeout_seconds": 120,
        "max_rows": 50,
        "max_queries": 2,
        "agentic": True,
    },
    "logs.volume": {
        "source": "log",
        "mcp": "grafana",
        "required_tools": (*_LOKI_DISCOVERY, "query_loki_stats"),
        "optional_tools": (),
        "when_to_use": "Estimate log volume and stream size in the incident window.",
        "llm_instructions": _LOGS_VOLUME_INSTRUCTIONS,
        "timeout_seconds": 30,
        "agentic": False,
    },
    "traces.get": {
        "source": "trace",
        "mcp": "tempo",
        "required_tools": ("get-trace",),
        "optional_tools": (),
        "when_to_use": "Retrieve a known trace_id and inspect its request path and spans.",
        "llm_instructions": _TRACES_GET_INSTRUCTIONS,
        "timeout_seconds": 30,
        # Bounds the flattened span table handed to synthesis (errors first, then slowest).
        "max_rows": 30,
        "agentic": False,
    },
    "traces.search": {
        "source": "trace",
        "mcp": "tempo",
        "required_tools": ("traceql-search",),
        "optional_tools": ("get-attribute-values",),
        "when_to_use": "Search bounded traces when no concrete trace_id is available.",
        "llm_instructions": _TRACES_SEARCH_INSTRUCTIONS,
        "timeout_seconds": 120,
        "max_rows": 30,
        "max_queries": 2,
        "agentic": True,
    },
    "metrics.infrastructure": {
        "source": "metric",
        "mcp": "influxdb",
        "required_tools": (*_INFLUX_DISCOVERY, "execute_influxql"),
        "optional_tools": (),
        "when_to_use": "Compare infrastructure CPU, memory, or load against the preceding baseline.",
        "llm_instructions": _METRICS_INSTRUCTIONS,
        # One query only: each call already fans out over the incident and baseline windows.
        "timeout_seconds": 120,
        "max_rows": 50,
        "max_queries": 1,
        "agentic": True,
    },
}
