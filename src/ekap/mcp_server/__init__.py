"""MCP server and tools."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.leading_space import leading_space_sources, leading_space_spec
from ekap.mcp_server.repeated_blank import repeated_blank_sources, repeated_blank_spec
from ekap.mcp_server.server import MCPServer as _MCPServer
from ekap.mcp_server.server import ToolResult, ToolSpec


class MCPServer(_MCPServer):
    """Server that also lists leading whitespace and repeated blank lines."""

    def list_tools(self) -> tuple[ToolSpec, ...]:
        tools = super().list_tools()
        extra = (leading_space_spec(), repeated_blank_spec())
        names = {tool.name for tool in tools}
        return (*tools, *(spec for spec in extra if spec.name not in names))

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        if name == "leading_space_sources":
            return leading_space_sources(self, args)
        if name == "repeated_blank_sources":
            return repeated_blank_sources(self, args)
        return super().call(name, arguments)


__all__ = [
    "MCPServer",
    "ToolResult",
    "ToolSpec",
]
