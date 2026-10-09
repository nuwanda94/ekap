"""Left-to-right isolate listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "left_to_right_isolate_sources"
_LEFT_TO_RIGHT_ISOLATE = "\u2066"


def left_to_right_isolate_spec() -> ToolSpec:
    """Advertised spec for sources that contain left-to-right isolates."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+2066. Does not return text.",
        mutates=False,
    )


def left_to_right_isolate_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return left-to-right-isolate count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+2066. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, left-to-right
    or right-to-left marks, Arabic letter marks, left-to-right or
    right-to-left embeddings, pop directional formatting, and left-to-right
    or right-to-left overrides are not counted.
    """
    total = text.count(_LEFT_TO_RIGHT_ISOLATE)
    runs = 0
    in_run = False
    for char in text:
        if char == _LEFT_TO_RIGHT_ISOLATE:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_LEFT_TO_RIGHT_ISOLATE),
        text.endswith(_LEFT_TO_RIGHT_ISOLATE),
    )


def left_to_right_isolate_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+2066. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "left_to_right_isolate_sources only accepts metadata"},
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
        total, runs, starts, ends = left_to_right_isolate_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "left_to_right_isolate_count": total,
                "run_count": runs,
                "starts_with_left_to_right_isolate": starts,
                "ends_with_left_to_right_isolate": ends,
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
