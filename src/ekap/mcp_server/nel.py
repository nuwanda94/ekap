"""Next-line listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "nel_sources"
_NEL = "\u0085"


def nel_spec() -> ToolSpec:
    """Advertised spec for sources that contain a next-line character."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+0085. Does not return text.",
        mutates=False,
    )


def nel_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return next-line count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+0085. Python ``splitlines``
    treats NEL as a line break, so counts are taken from the raw text.
    """
    total = text.count(_NEL)
    runs = 0
    in_run = False
    for char in text:
        if char == _NEL:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return total, runs, text.startswith(_NEL), text.endswith(_NEL)


def nel_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+0085. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "nel_sources only accepts metadata"},
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
        total, runs, starts, ends = nel_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "nel_count": total,
                "run_count": runs,
                "starts_with_nel": starts,
                "ends_with_nel": ends,
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
