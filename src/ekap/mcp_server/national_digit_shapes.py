"""National digit shapes listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "national_digit_shapes_sources"
_NATIONAL_DIGIT_SHAPES = "\u206e"


def national_digit_shapes_spec() -> ToolSpec:
    """Advertised spec for sources that contain national digit shapes."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+206E. Does not return text.",
        mutates=False,
    )


def national_digit_shapes_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return national-digit-shapes count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+206E. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, left-to-right
    or right-to-left marks, Arabic letter marks, left-to-right or
    right-to-left embeddings, pop directional formatting, left-to-right or
    right-to-left overrides, left-to-right, right-to-left, or first strong
    isolates, pop directional isolates, inhibit symmetric swapping,
    activate symmetric swapping, inhibit arabic form shaping, and activate
    arabic form shaping are not counted.
    """
    total = text.count(_NATIONAL_DIGIT_SHAPES)
    runs = 0
    in_run = False
    for char in text:
        if char == _NATIONAL_DIGIT_SHAPES:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_NATIONAL_DIGIT_SHAPES),
        text.endswith(_NATIONAL_DIGIT_SHAPES),
    )


def national_digit_shapes_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+206E. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "national_digit_shapes_sources only accepts metadata"},
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
        total, runs, starts, ends = national_digit_shapes_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "national_digit_shapes_count": total,
                "run_count": runs,
                "starts_with_national_digit_shapes": starts,
                "ends_with_national_digit_shapes": ends,
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
