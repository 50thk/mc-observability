import logging

from app.core.mcp.multi_mcp_manager import MCPManager
from config.ConfigManager import ConfigManager

logger = logging.getLogger(__name__)

# Feature -> MCP server names (servers themselves are declared in config.yaml mcp_servers).
LOG_ANALYSIS_SERVERS = ("grafana",)
ALERT_ANALYSIS_SERVERS = ("mariadb", "influxdb")
SERVER_ERROR_ANALYSIS_SERVERS = ("grafana", "tempo", "influxdb")


def _build_manager(server_names) -> MCPManager:
    """Build an MCPManager for a feature from the config-declared server map."""
    servers = ConfigManager().get_mcp_config()
    missing = [name for name in server_names if name not in servers]
    if missing:
        logger.warning(f"MCP servers missing from config mcp_servers: {missing}")
    return MCPManager({name: servers[name] for name in server_names if name in servers})


# Each dependency keeps its own try/finally around the yield (no generator delegation):
# FastAPI throws endpoint exceptions into the generator at the yield point, and only a
# directly enclosing finally runs stop_all() in the request task. Delegating via
# `async for` would defer cleanup to GC-time asyncgen finalization in another task,
# which breaks anyio cancel scopes inside the MCP transports and leaks connections.


async def get_log_analysis_context():
    """Dependency: MCPManager for log analysis (Grafana)."""
    manager = _build_manager(LOG_ANALYSIS_SERVERS)
    try:
        logger.info(f"Connecting MCPs for log analysis: {list(manager.connections)}")
        await manager.start_all()
        yield manager
    finally:
        logger.info("Disconnecting MCPs for log analysis...")
        await manager.stop_all()


async def get_alert_analysis_context():
    """Dependency: MCPManager for alert analysis (MariaDB, InfluxDB)."""
    manager = _build_manager(ALERT_ANALYSIS_SERVERS)
    try:
        logger.info(f"Connecting MCPs for alert analysis: {list(manager.connections)}")
        await manager.start_all()
        yield manager
    finally:
        logger.info("Disconnecting MCPs for alert analysis...")
        await manager.stop_all()


async def get_server_error_analysis_context():
    """Dependency: MCPManager for HTTP 5xx analysis (Grafana, Tempo, InfluxDB)."""
    manager = _build_manager(SERVER_ERROR_ANALYSIS_SERVERS)
    try:
        logger.info(f"Connecting MCPs for server error analysis: {list(manager.connections)}")
        await manager.start_all()
        yield manager
    finally:
        logger.info("Disconnecting MCPs for server error analysis...")
        await manager.stop_all()
