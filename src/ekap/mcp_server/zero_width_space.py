"""Zero-width space listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "zero_width_space_sources"
_ZERO_WIDTH_SPACE = "\u200b"


def zero_width_space_spec() -> ToolSpec:
    """Advertised spec for sources that contain a zero-width space."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+200B. Does not return text.",
        mutates=False,
    )


def zero_width_space_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return zero-width-space count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+200B. BOM, word joiners,
    and zero-width joiners or non-joiners are not counted.
    """
    total = text.count(_ZERO_WIDTH_SPACE)
    runs = 0
    in_run = False
    for char in text:
        if char == _ZERO_WIDTH_SPACE:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_ZERO_WIDTH_SPACE),
        text.endswith(_ZERO_WIDTH_SPACE),
    )


def zero_width_space_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+200B. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "zero_width_space_sources only accepts metadata"},
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
        total, runs, starts, ends = zero_width_space_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "zero_width_space_count": total,
                "run_count": runs,
                "starts_with_zero_width_space": starts,
                "ends_with_zero_width_space": ends,
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
