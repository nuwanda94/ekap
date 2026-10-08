"""Vertical-tab listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "vertical_tab_sources"
_VERTICAL_TAB = "\v"


def vertical_tab_spec() -> ToolSpec:
    """Advertised spec for sources that contain a vertical tab."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+000B. Does not return text.",
        mutates=False,
    )


def vertical_tab_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return vertical-tab count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+000B. Python ``splitlines``
    treats a vertical tab as a line break, so counts are taken from the raw text.
    """
    total = text.count(_VERTICAL_TAB)
    runs = 0
    in_run = False
    for char in text:
        if char == _VERTICAL_TAB:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return total, runs, text.startswith(_VERTICAL_TAB), text.endswith(_VERTICAL_TAB)


def vertical_tab_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+000B. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "vertical_tab_sources only accepts metadata"},
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
        total, runs, starts, ends = vertical_tab_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "vertical_tab_count": total,
                "run_count": runs,
                "starts_with_vertical_tab": starts,
                "ends_with_vertical_tab": ends,
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
