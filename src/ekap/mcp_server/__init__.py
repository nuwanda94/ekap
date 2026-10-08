"""MCP server and tools."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.bom import bom_sources, bom_spec
from ekap.mcp_server.crlf import crlf_sources, crlf_spec
from ekap.mcp_server.leading_blank import leading_blank_sources, leading_blank_spec
from ekap.mcp_server.leading_space import leading_space_sources, leading_space_spec
from ekap.mcp_server.long_line import long_line_sources, long_line_spec
from ekap.mcp_server.mixed_indent import mixed_indent_sources, mixed_indent_spec
from ekap.mcp_server.nbsp import nbsp_sources, nbsp_spec
from ekap.mcp_server.repeated_blank import repeated_blank_sources, repeated_blank_spec
from ekap.mcp_server.server import MCPServer as _MCPServer
from ekap.mcp_server.server import ToolResult, ToolSpec
from ekap.mcp_server.trailing_blank import trailing_blank_sources, trailing_blank_spec
from ekap.mcp_server.trailing_space import trailing_space_sources, trailing_space_spec
from ekap.mcp_server.unterminated import unterminated_sources, unterminated_spec


class MCPServer(_MCPServer):
    """Server that also lists whitespace hygiene and blank-line sources."""

    def list_tools(self) -> tuple[ToolSpec, ...]:
        tools = super().list_tools()
        extra = (
            bom_spec(),
            crlf_spec(),
            leading_blank_spec(),
            leading_space_spec(),
            long_line_spec(),
            mixed_indent_spec(),
            nbsp_spec(),
            repeated_blank_spec(),
            trailing_blank_spec(),
            trailing_space_spec(),
            unterminated_spec(),
        )
        names = {tool.name for tool in tools}
        return (*tools, *(spec for spec in extra if spec.name not in names))

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        if name == "bom_sources":
            return bom_sources(self, args)
        if name == "crlf_sources":
            return crlf_sources(self, args)
        if name == "leading_blank_sources":
            return leading_blank_sources(self, args)
        if name == "long_line_sources":
            return long_line_sources(self, args)
        if name == "leading_space_sources":
            return leading_space_sources(self, args)
        if name == "mixed_indent_sources":
            return mixed_indent_sources(self, args)
        if name == "nbsp_sources":
            return nbsp_sources(self, args)
        if name == "repeated_blank_sources":
            return repeated_blank_sources(self, args)
        if name == "trailing_blank_sources":
            return trailing_blank_sources(self, args)
        if name == "trailing_space_sources":
            return trailing_space_sources(self, args)
        if name == "unterminated_sources":
            return unterminated_sources(self, args)
        return super().call(name, arguments)


__all__ = [
    "MCPServer",
    "ToolResult",
    "ToolSpec",
]
