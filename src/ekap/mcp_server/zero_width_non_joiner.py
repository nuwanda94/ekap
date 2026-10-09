"""Zero-width non-joiner listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "zero_width_non_joiner_sources"
_ZERO_WIDTH_NON_JOINER = "\u200c"


def zero_width_non_joiner_spec() -> ToolSpec:
    """Advertised spec for sources that contain a zero-width non-joiner."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+200C. Does not return text.",
        mutates=False,
    )


def zero_width_non_joiner_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return zero-width-non-joiner count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+200C. BOM, zero-width
    spaces, word joiners, and zero-width joiners are not counted.
    """
    total = text.count(_ZERO_WIDTH_NON_JOINER)
    runs = 0
    in_run = False
    for char in text:
        if char == _ZERO_WIDTH_NON_JOINER:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_ZERO_WIDTH_NON_JOINER),
        text.endswith(_ZERO_WIDTH_NON_JOINER),
    )


def zero_width_non_joiner_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+200C. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "zero_width_non_joiner_sources only accepts metadata"},
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
        total, runs, starts, ends = zero_width_non_joiner_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "zero_width_non_joiner_count": total,
                "run_count": runs,
                "starts_with_zero_width_non_joiner": starts,
                "ends_with_zero_width_non_joiner": ends,
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
