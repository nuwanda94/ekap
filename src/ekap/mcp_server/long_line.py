"""Long-line listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "long_line_sources"
_DEFAULT_MAX = 120


def long_line_spec() -> ToolSpec:
    """Advertised spec for sources with a line longer than a limit."""
    return ToolSpec(
        _TOOL,
        "List sources with a line longer than max_chars. Does not return text.",
        mutates=False,
    )


def long_line_counts(text: str, max_chars: int) -> tuple[int, int, int]:
    """Return line count, long-line count, and the longest line length.

    The empty segment after a final terminator is not a line. A line is long
    only when its length is strictly greater than max_chars.
    """
    lines = text.splitlines()
    longest = max((len(line) for line in lines), default=0)
    long_count = sum(len(line) > max_chars for line in lines)
    return len(lines), long_count, longest


def _max_chars_error(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return "max_chars must be a positive integer"
    return None


def long_line_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources with a line longer than max_chars. Never mutates."""
    extra = set(args) - {"metadata", "max_chars"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "long_line_sources only accepts metadata and max_chars"},
        )
    metadata = args.get("metadata")
    metadata_error = _metadata_error(metadata)
    if metadata_error is not None:
        return ToolResult(name=_TOOL, ok=False, data={"error": metadata_error})
    max_chars = args.get("max_chars", _DEFAULT_MAX)
    max_error = _max_chars_error(max_chars)
    if max_error is not None:
        return ToolResult(name=_TOOL, ok=False, data={"error": max_error})
    if server._ingester is None:
        return ToolResult(name=_TOOL, ok=False, data={"error": "no ingester configured"})
    try:
        documents = server._ingester.find(metadata=metadata)
    except (TypeError, ValueError) as exc:
        return ToolResult(name=_TOOL, ok=False, data={"error": str(exc)})
    limit = _DEFAULT_MAX if max_chars is None else max_chars
    flagged = []
    for document in documents:
        line_count, long_count, longest = long_line_counts(document.text, limit)
        if long_count == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "line_count": line_count,
                "long_line_count": long_count,
                "longest": longest,
                "max_chars": limit,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={
            "sources": len(documents),
            "max_chars": limit,
            "flagged": flagged,
            "executed": False,
        },
    )
