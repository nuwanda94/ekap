"""Left-to-right embedding listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "left_to_right_embedding_sources"
_LEFT_TO_RIGHT_EMBEDDING = "\u202a"


def left_to_right_embedding_spec() -> ToolSpec:
    """Advertised spec for sources that contain a left-to-right embedding."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+202A. Does not return text.",
        mutates=False,
    )


def left_to_right_embedding_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return left-to-right-embedding count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+202A. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, left-to-right
    or right-to-left marks, and Arabic letter marks are not counted.
    """
    total = text.count(_LEFT_TO_RIGHT_EMBEDDING)
    runs = 0
    in_run = False
    for char in text:
        if char == _LEFT_TO_RIGHT_EMBEDDING:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_LEFT_TO_RIGHT_EMBEDDING),
        text.endswith(_LEFT_TO_RIGHT_EMBEDDING),
    )


def left_to_right_embedding_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+202A. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "left_to_right_embedding_sources only accepts metadata"},
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
        total, runs, starts, ends = left_to_right_embedding_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "left_to_right_embedding_count": total,
                "run_count": runs,
                "starts_with_left_to_right_embedding": starts,
                "ends_with_left_to_right_embedding": ends,
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
