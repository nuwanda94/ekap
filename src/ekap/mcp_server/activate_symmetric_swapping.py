"""Activate symmetric swapping listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "activate_symmetric_swapping_sources"
_ACTIVATE_SYMMETRIC_SWAPPING = "\u206b"


def activate_symmetric_swapping_spec() -> ToolSpec:
    """Advertised spec for sources that contain activate symmetric swapping."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+206B. Does not return text.",
        mutates=False,
    )


def activate_symmetric_swapping_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return activate-symmetric-swapping count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+206B. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, left-to-right
    or right-to-left marks, Arabic letter marks, left-to-right or
    right-to-left embeddings, pop directional formatting, left-to-right or
    right-to-left overrides, left-to-right, right-to-left, or first strong
    isolates, pop directional isolates, and inhibit symmetric swapping are
    not counted.
    """
    total = text.count(_ACTIVATE_SYMMETRIC_SWAPPING)
    runs = 0
    in_run = False
    for char in text:
        if char == _ACTIVATE_SYMMETRIC_SWAPPING:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_ACTIVATE_SYMMETRIC_SWAPPING),
        text.endswith(_ACTIVATE_SYMMETRIC_SWAPPING),
    )


def activate_symmetric_swapping_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+206B. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "activate_symmetric_swapping_sources only accepts metadata"},
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
        total, runs, starts, ends = activate_symmetric_swapping_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "activate_symmetric_swapping_count": total,
                "run_count": runs,
                "starts_with_activate_symmetric_swapping": starts,
                "ends_with_activate_symmetric_swapping": ends,
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
