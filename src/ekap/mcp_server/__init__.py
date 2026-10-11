"""MCP server and tools."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.activate_arabic_form_shaping import (
    activate_arabic_form_shaping_sources,
    activate_arabic_form_shaping_spec,
)
from ekap.mcp_server.variation_selector_16 import (
    variation_selector_16_sources,
    variation_selector_16_spec,
)
# ... (other imports omitted for brevity in this example; in practice the full file is used)

class MCPServer(_MCPServer):
    def list_tools(self) -> tuple[ToolSpec, ...]:
        tools = super().list_tools()
        extra = (
            variation_selector_16_spec(),
            # other specs
        )
        names = {tool.name for tool in tools}
        return (*tools, *(spec for spec in extra if spec.name not in names))

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        if name == "variation_selector_16_sources":
            return variation_selector_16_sources(self, args)
        return super().call(name, arguments)
