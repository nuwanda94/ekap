"""Right-to-left mark listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "right_to_left_mark_sources"
_RIGHT_TO_LEFT_MARK = "\u200f"


def right_to_left_mark_spec() -> ToolSpec:
    """Advertised spec for sources that contain a right-to-left mark."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+200F. Does not return text.",
        mutates=False,
    )


def right_to_left_mark_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return right-to-left-mark count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+200F. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, and left-to-right
    marks are not counted.
    """
    total = text.count(_RIGHT_TO_LEFT_MARK)
    runs = 0
    in_run = False
    for char in text:
        if char == _RIGHT_TO_LEFT_MARK:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_RIGHT_TO_LEFT_MARK),
        text.endswith(_RIGHT_TO_LEFT_MARK),
    )


def right_to_left_mark_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+200F. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "right_to_left_mark_sources only accepts metadata"},
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
        total, runs, starts, ends = right_to_left_mark_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "right_to_left_mark_count": total,
                "run_count": runs,
                "starts_with_right_to_left_mark": starts,
                "ends_with_right_to_left_mark": ends,
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
