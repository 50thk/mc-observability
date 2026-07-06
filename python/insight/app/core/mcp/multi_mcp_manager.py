import asyncio
import logging
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPManager:
    """Connect config-declared MCP servers and expose their tools as LangChain tools.

    Connections are declared as ``{name: {"url": ..., "transport": ..., "enabled": bool?}}``
    (see ``config.yaml`` ``llm.mcp.mcp_servers``). Transport handling and session setup are
    delegated to ``langchain-mcp-adapters``; adding a new MCP server needs no new code here.
    """

    # Transports that require a url (stdio uses command/args instead).
    _URL_TRANSPORTS = ("sse", "streamable_http", "websocket")

    def __init__(self, connections: dict[str, dict] | None = None):
        self.connections: dict[str, dict] = {}
        self.all_tools: list = []
        self.tools_by_mcp: dict[str, list] = {}
        self._exit_stack = AsyncExitStack()
        for name, conn in (connections or {}).items():
            self.add_server(name, **conn)

    def add_server(self, name: str, url: str | None = None, transport: str = "streamable_http",
                   enabled: bool = True, **extra):
        """Register one MCP server connection; actual connect happens in start_all().

        Extra keys (headers, timeout, auth, command/args for stdio, ...) are passed
        through verbatim to the langchain-mcp-adapters connection.
        """
        if not enabled:
            logger.info(f"MCP server '{name}' is disabled; skipping")
            return
        if transport in self._URL_TRANSPORTS and not url:
            # Misconfigured entry must degrade like a dead server, not 500 the request.
            logger.error(f"MCP server '{name}' ({transport}) has no url; skipping")
            return
        connection = {"transport": transport, **extra}
        if url is not None:
            connection["url"] = url
        self.connections[name] = connection

    async def __aenter__(self):
        await self.start_all()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop_all()

    async def start_all(self):
        """Open one persistent session per registered server and load its tools.

        One server failing must not block the others; failed servers simply
        contribute no tools (callers see them via get_tools_for_mcp(...) == []).
        """
        # start_all reflects exactly this run — never accumulate across restarts.
        self.all_tools = []
        self.tools_by_mcp = {}
        client = MultiServerMCPClient(self.connections)
        for name in self.connections:
            try:
                logger.info(f"Starting '{name}' MCP client...")
                session = await self._exit_stack.enter_async_context(client.session(name))
                tools = await load_mcp_tools(session)
                self.tools_by_mcp[name] = tools
                self.all_tools.extend(tools)
                logger.info(f"'{name}' MCP client started with {len(tools)} tools")
            except Exception as e:
                logger.error(f"Failed to start '{name}' MCP client: {e}")

        logger.info(f"Total MCP tools loaded: {len(self.all_tools)}")

    async def stop_all(self):
        """Close all sessions through the manager-owned async context stack."""
        try:
            await self._exit_stack.aclose()
            logger.info("All MCP clients stopped successfully")
        except asyncio.CancelledError as e:
            logger.error(f"MCP client cleanup was cancelled: {e}")
        except Exception as e:
            logger.error(f"Failed to stop MCP clients: {e}")
        finally:
            self._exit_stack = AsyncExitStack()
            self.all_tools = []
            self.tools_by_mcp = {}

    def get_all_tools(self):
        """Return tools from all connected MCP servers."""
        return self.all_tools

    def get_tools_for_mcp(self, name: str):
        """Return the cached LangChain tools loaded from one MCP server (unknown -> [])."""
        return self.tools_by_mcp.get(name, [])
