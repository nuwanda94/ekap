"""Pop directional formatting listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "pop_directional_formatting_sources"
_POP_DIRECTIONAL_FORMATTING = "\u202c"


def pop_directional_formatting_spec() -> ToolSpec:
    """Advertised spec for sources that contain pop directional formatting."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+202C. Does not return text.",
        mutates=False,
    )


def pop_directional_formatting_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return pop-directional-formatting count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+202C. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, left-to-right
    or right-to-left marks, Arabic letter marks, and left-to-right or
    right-to-left embeddings are not counted.
    """
    total = text.count(_POP_DIRECTIONAL_FORMATTING)
    runs = 0
    in_run = False
    for char in text:
        if char == _POP_DIRECTIONAL_FORMATTING:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_POP_DIRECTIONAL_FORMATTING),
        text.endswith(_POP_DIRECTIONAL_FORMATTING),
    )


def pop_directional_formatting_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+202C. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "pop_directional_formatting_sources only accepts metadata"},
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
        total, runs, starts, ends = pop_directional_formatting_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "pop_directional_formatting_count": total,
                "run_count": runs,
                "starts_with_pop_directional_formatting": starts,
                "ends_with_pop_directional_formatting": ends,
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
