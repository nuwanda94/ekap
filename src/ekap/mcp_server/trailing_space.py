"""Trailing-whitespace source listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "trailing_space_sources"


def trailing_space_spec() -> ToolSpec:
    """Advertised spec for sources whose lines end with spaces or tabs."""
    return ToolSpec(
        _TOOL,
        "List sources whose lines end with spaces or tabs. Does not return text.",
        mutates=False,
    )


def trailing_space_counts(text: str) -> tuple[int, int]:
    """Count lines ending with ASCII spaces and lines ending with tabs.

    CRLF and bare CR are line boundaries. A line whose trailing run contains both
    counts in both totals. The empty segment after a final terminator is not a line.
    Interior spaces and tabs do not count.
    """
    space_lines = 0
    tab_lines = 0

    def consider(line: str) -> None:
        nonlocal space_lines, tab_lines
        saw_space = False
        saw_tab = False
        index = len(line) - 1
        while index >= 0 and line[index] in " \t":
            if line[index] == " ":
                saw_space = True
            else:
                saw_tab = True
            index -= 1
        if saw_space:
            space_lines += 1
        if saw_tab:
            tab_lines += 1

    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "\r":
            consider(text[start:index])
            index += 2 if index + 1 < length and text[index + 1] == "\n" else 1
            start = index
            continue
        if text[index] == "\n":
            consider(text[start:index])
            index += 1
            start = index
            continue
        index += 1
    if start < length:
        consider(text[start:])
    return space_lines, tab_lines


def trailing_space_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List matching sources with trailing spaces or tabs. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "trailing_space_sources only accepts metadata"},
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
        space_lines, tab_lines = trailing_space_counts(document.text)
        if space_lines == 0 and tab_lines == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "space_lines": space_lines,
                "tab_lines": tab_lines,
                "mixed": space_lines > 0 and tab_lines > 0,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={"sources": len(documents), "flagged": flagged, "executed": False},
    )
