"""Line-separator listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "line_separator_sources"
_LINE_SEPARATOR = "\u2028"


def line_separator_spec() -> ToolSpec:
    """Advertised spec for sources that contain a line separator."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+2028. Does not return text.",
        mutates=False,
    )


def line_separator_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return line-separator count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+2028. Python ``splitlines``
    treats the line separator as a line break, so counts are taken from the raw text.
    """
    total = text.count(_LINE_SEPARATOR)
    runs = 0
    in_run = False
    for char in text:
        if char == _LINE_SEPARATOR:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return total, runs, text.startswith(_LINE_SEPARATOR), text.endswith(_LINE_SEPARATOR)


def line_separator_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+2028. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "line_separator_sources only accepts metadata"},
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
        total, runs, starts, ends = line_separator_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "line_separator_count": total,
                "run_count": runs,
                "starts_with_line_separator": starts,
                "ends_with_line_separator": ends,
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
