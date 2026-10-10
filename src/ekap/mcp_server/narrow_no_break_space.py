"""Narrow no-break space listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "narrow_no_break_space_sources"
_NNBSP = "\u202f"


def narrow_no_break_space_spec() -> ToolSpec:
    """Advertised spec for sources that contain a narrow no-break space."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+202F. Does not return text.",
        mutates=False,
    )


def narrow_no_break_space_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return narrow-no-break-space count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+202F. Ordinary whitespace,
    BOM, zero-width spaces, word joiners, zero-width joiners or non-joiners,
    left-to-right or right-to-left marks, Arabic letter marks, left-to-right
    or right-to-left embeddings, pop directional formatting, left-to-right or
    right-to-left overrides, left-to-right, right-to-left, or first strong
    isolates, pop directional isolates, inhibit symmetric swapping, activate
    symmetric swapping, inhibit arabic form shaping, activate arabic form
    shaping, national digit shapes, nominal digit shapes, and soft hyphens
    are not counted.
    """
    total = text.count(_NNBSP)
    runs = 0
    in_run = False
    for char in text:
        if char == _NNBSP:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_NNBSP),
        text.endswith(_NNBSP),
    )


def narrow_no_break_space_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+202F. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "narrow_no_break_space_sources only accepts metadata"},
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
        total, runs, starts, ends = narrow_no_break_space_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "narrow_no_break_space_count": total,
                "run_count": runs,
                "starts_with_narrow_no_break_space": starts,
                "ends_with_narrow_no_break_space": ends,
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
