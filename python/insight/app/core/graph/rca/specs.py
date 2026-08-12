"""Per-source specs for the central RCA investigation agent.

One toolset per source (log, trace, metric): which MCP tools it needs, what the agent is
told about it, its timeout and result limits, and — for metric — the static catalog of
measurements, fields and tag keys fixed by the Telegraf templates.
"""

# Common Telegraf tags: [global_tags] in telegraf_global plus the agent host tag.
_COMMON_METRIC_TAG_KEYS = ("ns_id", "infra_id", "node_id", "host")

# Metric measurement/field/tag keys are fixed by the Telegraf templates the manager renders
# (java/mc-o11y-manager/src/main/resources/telegraf_inputs_*). Each entry is the plugin's
# default field set minus that template's `fieldexclude`/`fieldinclude`, so there is nothing
# to discover per request. `dcgm` is the starlark conversion of DCGM exporter metrics
# (telegraf_processors_starlark; field names from GpuMetricKeyField.java) and exists only on
# GPU nodes. Tag *values* stay dynamic and are discovered with get_tag_values.
METRIC_CATALOG = {
    # inputs.cpu: percpu + totalcpu, collect_cpu_time = false, fieldexclude usage_guest*
    "cpu": {
        "fields": (
            "usage_user",
            "usage_system",
            "usage_idle",
            "usage_active",
            "usage_nice",
            "usage_iowait",
            "usage_irq",
            "usage_softirq",
            "usage_steal",
        ),
        "tag_keys": (*_COMMON_METRIC_TAG_KEYS, "cpu"),
    },
    # inputs.disk: fieldexclude inode*
    "disk": {
        "fields": ("total", "free", "used", "used_percent"),
        "tag_keys": (*_COMMON_METRIC_TAG_KEYS, "device", "fstype", "path", "mode"),
    },
    # inputs.diskio: fieldexclude weighted_io_time, merged*
    "diskio": {
        "fields": (
            "reads",
            "writes",
            "read_bytes",
            "write_bytes",
            "read_time",
            "write_time",
            "io_time",
            "iops_in_progress",
        ),
        "tag_keys": (*_COMMON_METRIC_TAG_KEYS, "name"),
    },
    # inputs.mem: fieldexclude commit*, dirty, high*, huge*, laundry, low*, mapped, page*,
    # slab, *claim*, swap*, vmalloc*, wired*, write*
    "mem": {
        "fields": (
            "total",
            "available",
            "available_percent",
            "used",
            "used_percent",
            "free",
            "active",
            "inactive",
            "buffered",
            "cached",
            "shared",
        ),
        "tag_keys": _COMMON_METRIC_TAG_KEYS,
    },
    # inputs.net: ignore_protocol_stats = true, fieldexclude speed
    "net": {
        "fields": (
            "bytes_sent",
            "bytes_recv",
            "packets_sent",
            "packets_recv",
            "err_in",
            "err_out",
            "drop_in",
            "drop_out",
        ),
        "tag_keys": (*_COMMON_METRIC_TAG_KEYS, "interface"),
    },
    # inputs.processes: fieldexclude paging, total_threads
    "processes": {
        "fields": (
            "total",
            "running",
            "sleeping",
            "blocked",
            "stopped",
            "zombies",
            "dead",
            "idle",
            "unknown",
            "parked",
        ),
        "tag_keys": _COMMON_METRIC_TAG_KEYS,
    },
    # inputs.procstat: fieldinclude cpu_usage, memory_usage; tag_with pid, user
    "procstat": {
        "fields": ("cpu_usage", "memory_usage"),
        "tag_keys": (*_COMMON_METRIC_TAG_KEYS, "process_name", "pid", "user"),
    },
    # inputs.swap: defaults
    "swap": {
        "fields": ("total", "used", "free", "used_percent", "in", "out"),
        "tag_keys": _COMMON_METRIC_TAG_KEYS,
    },
    # inputs.system: fieldexclude uptime_format
    "system": {
        "fields": ("load1", "load5", "load15", "n_cpus", "n_users", "n_unique_users", "uptime"),
        "tag_keys": _COMMON_METRIC_TAG_KEYS,
    },
    # inputs.prometheus (DCGM exporter) -> processors.starlark: DCGM_FI_DEV_X -> dcgm.x
    "dcgm": {
        "fields": (
            "sm_clock",
            "mem_clock",
            "memory_temp",
            "gpu_temp",
            "fan_speed",
            "power_usage",
            "total_energy_consumption",
            "p_state",
            "pcie_tx_throughput",
            "pcie_rx_throughput",
            "pcie_replay_counter",
            "gpu_util",
            "mem_copy_util",
            "enc_util",
            "dec_util",
            "xid_errors",
            "clocks_event_reasons",
            "xid_errors_count",
            "fb_total",
            "fb_free",
            "fb_used",
            "ecc_sbe_vol_total",
            "ecc_dbe_vol_total",
            "ecc_sbe_agg_total",
            "ecc_dbe_agg_total",
            "nvlink_bandwidth_total",
            "vgpu_license_status",
            "uncorrectable_remapped_rows",
            "correctable_remapped_rows",
            "row_remap_failure",
        ),
        "tag_keys": (*_COMMON_METRIC_TAG_KEYS, "gpu", "UUID", "device", "modelName", "Hostname", "pci_bus_id"),
    },
}


def render_metric_catalog() -> str:
    lines = []
    for measurement, entry in METRIC_CATALOG.items():
        lines.append(f"- {measurement}: fields [{', '.join(entry['fields'])}]; tags [{', '.join(entry['tag_keys'])}]")
    return "\n".join(lines)


_LOG_SOURCE_INSTRUCTIONS = """
Log source (Loki). Answers: what the service logged in the incident window and how much.
The Loki datasource and the time window are fixed by code; every log tool already runs
inside them. Write selection-only LogQL: a stream selector {label="value", ...} followed by
optional line filters (|=, !=, |~, !~). Parsers, formatters, aggregations, ranges and
offsets are rejected. `component` is the service label emitted by this platform's log
pipeline. Discover label names or values only when you do not know them. Start broad, then
narrow once with literals you saw in earlier output. An empty result is data: relax one
constraint once, then report what you searched for. Never infer a cause from absent logs.
"""

_TRACE_SOURCE_INSTRUCTIONS = """
Trace source (Tempo). Answers: which requests failed or were slow, and where one request
spent its time. The time window is fixed by code. Write a single selection-only TraceQL
spanset filter { ... } combining comparisons with && and ||; pipelines, aggregates and
structural operators between spansets are rejected. Search the window when no trace_id is
known; when a log line or a search exposes a trace_id, open it with get_trace and read the
span table (error spans first, then the slowest). Setting both an error and a duration
constraint usually returns nothing — prefer one. Relax one constraint after an empty search;
then report NO_DATA rather than concluding that nothing failed.
"""

_METRIC_SOURCE_INSTRUCTIONS = (
    """
Metric source (InfluxDB, Telegraf). Answers: whether an infrastructure signal moved during
the window, optionally compared with the immediately preceding baseline. The database and
time window are fixed by code. Measurements, fields and tag keys are fixed too — pick them
verbatim from this catalog:
"""
    + render_metric_catalog()
    + """
Tag values (node_id, device, interface, pid, ...) vary per node: take them from the scope
when given, otherwise discover them with get_tag_values. Choose max for spikes and
saturation, mean for sustained load, last for the final state. A measurement listed here
may still have no rows for a node whose plugin is off: that is NO_DATA, not an error.
"""
)


SOURCE_SPECS = {
    "log": {
        "mcp": "grafana",
        "summary": "What the service logged in the window and how much; Loki via LogQL.",
        # list_datasources is called by code once per request to resolve the Loki UID; it
        # is never exposed to the agent.
        "required_tools": (
            "list_datasources",
            "list_loki_label_names",
            "list_loki_label_values",
            "query_loki_logs",
            "query_loki_stats",
        ),
        "optional_tools": (),
        "llm_instructions": _LOG_SOURCE_INSTRUCTIONS,
        "timeout_seconds": 120,
        "default_limit": 50,
        "max_limit": 200,
    },
    "trace": {
        "mcp": "tempo",
        "summary": "Which requests failed or were slow and where one request spent its time; Tempo via TraceQL.",
        "required_tools": ("traceql-search", "get-trace"),
        "optional_tools": ("get-attribute-values",),
        "llm_instructions": _TRACE_SOURCE_INSTRUCTIONS,
        "timeout_seconds": 120,
        "default_limit": 20,
        # Also bounds the flattened span table returned by get_trace.
        "max_limit": 50,
    },
    "metric": {
        "mcp": "influxdb",
        "summary": "Whether an infrastructure signal (cpu, mem, disk, net, ...) moved versus the preceding baseline.",
        "required_tools": ("get_tag_values", "execute_influxql"),
        "optional_tools": (),
        "llm_instructions": _METRIC_SOURCE_INSTRUCTIONS,
        "timeout_seconds": 120,
        "default_limit": 50,
        "max_limit": 500,
        "catalog": METRIC_CATALOG,
    },
}
