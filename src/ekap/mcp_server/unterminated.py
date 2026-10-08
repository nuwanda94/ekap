"""Unterminated source listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "unterminated_sources"


def unterminated_spec() -> ToolSpec:
    """Advertised spec for sources that lack a final line terminator."""
    return ToolSpec(
        _TOOL,
        "List sources that do not end with a line terminator. Does not return text.",
        mutates=False,
    )


def _is_terminator(text: str) -> bool:
    return text.endswith(("\n", "\r"))


def unterminated_counts(text: str) -> tuple[int, int]:
    """Return line count and last-line character count.

    CRLF and bare CR are line boundaries. The empty segment after a final
    terminator is not a line. A source with no terminator still has its
    final segment as a line.
    """
    lines = 0
    last_chars = 0
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "\r":
            lines += 1
            last_chars = index - start
            index += 2 if index + 1 < length and text[index + 1] == "\n" else 1
            start = index
            continue
        if text[index] == "\n":
            lines += 1
            last_chars = index - start
            index += 1
            start = index
            continue
        index += 1
    if start < length:
        lines += 1
        last_chars = length - start
    return lines, last_chars


def unterminated_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List matching sources that do not end with CR or LF. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "unterminated_sources only accepts metadata"},
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
        if _is_terminator(document.text):
            continue
        lines, last_chars = unterminated_counts(document.text)
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "lines": lines,
                "last_line_chars": last_chars,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={"sources": len(documents), "flagged": flagged, "executed": False},
    )
