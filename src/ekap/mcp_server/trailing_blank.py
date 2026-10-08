"""Trailing blank-line source listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "trailing_blank_sources"


def trailing_blank_spec() -> ToolSpec:
    """Advertised spec for sources that end with blank lines."""
    return ToolSpec(
        _TOOL,
        "List sources that end with one or more blank lines. Does not return text.",
        mutates=False,
    )


def _is_blank(line: str) -> bool:
    return line.strip(" \t") == ""


def trailing_blank_counts(text: str) -> tuple[int, int]:
    """Return trailing blank-line count and content-line count.

    A blank line is empty or only ASCII spaces and tabs. CRLF and bare CR are
    line boundaries. The empty segment after a final terminator is not a line.
    Only the blank run after the last content line counts as trailing.
    """
    trailing = 0
    content_lines = 0
    start = 0
    index = 0
    length = len(text)

    def consider(line: str) -> None:
        nonlocal trailing, content_lines
        if _is_blank(line):
            trailing += 1
            return
        trailing = 0
        content_lines += 1

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
    return trailing, content_lines


def trailing_blank_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List matching sources that end with blank lines. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "trailing_blank_sources only accepts metadata"},
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
        trailing, content_lines = trailing_blank_counts(document.text)
        if trailing == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "trailing_blank_lines": trailing,
                "content_lines": content_lines,
                "blank_only": content_lines == 0,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={"sources": len(documents), "flagged": flagged, "executed": False},
    )
