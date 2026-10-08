"""MCP server and tools."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.leading_space import leading_space_sources, leading_space_spec
from ekap.mcp_server.server import MCPServer as _MCPServer
from ekap.mcp_server.server import ToolResult, ToolSpec


class MCPServer(_MCPServer):
    """Server that also lists sources with leading spaces or tabs."""

    def list_tools(self) -> tuple[ToolSpec, ...]:
        tools = super().list_tools()
        spec = leading_space_spec()
        if any(tool.name == spec.name for tool in tools):
            return tools
        return (*tools, spec)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if name == "leading_space_sources":
            return leading_space_sources(self, dict(arguments or {}))
        return super().call(name, arguments)


__all__ = [
    "MCPServer",
    "ToolResult",
    "ToolSpec",
]
