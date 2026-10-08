"""Tab listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "tab_sources"
_TAB = "\t"


def tab_spec() -> ToolSpec:
    """Advertised spec for sources that contain a tab."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+0009. Does not return text.",
        mutates=False,
    )


def tab_counts(text: str) -> tuple[int, int, int]:
    """Return tab count, indent-tab line count, and inline tab count."""
    total = text.count(_TAB)
    indent_lines = 0
    indent_tabs = 0
    for line in text.splitlines():
        if not line.startswith(_TAB):
            continue
        indent_lines += 1
        indent_tabs += len(line) - len(line.lstrip(_TAB))
    return total, indent_lines, total - indent_tabs


def tab_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+0009. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "tab_sources only accepts metadata"},
        )
    metadata = args.get("metadata")
    metadata_error = _metadata_error(metadata)
    if metadata_error is not None:
        return ToolResult(name=_TOOL, ok=False, data={"error": metadata_error})
    if server._ingester is None:
        return ToolResult(name=_TOOL, ok=False, data={"error": "no ingester configured"})
    try:
        documents = server._ingester.find(metadata=metadata)
    except (TypeError, ValueError) as exc:
        return ToolResult(name=_TOOL, ok=False, data={"error": str(exc)})
    flagged = []
    for document in documents:
        total, indent_lines, inline = tab_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "tab_count": total,
                "indent_tab_lines": indent_lines,
                "inline_tab_count": inline,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={
            "sources": len(documents),
            "flagged": flagged,
            "executed": False,
        },
    )
