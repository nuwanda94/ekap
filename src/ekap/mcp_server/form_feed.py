"""Form-feed listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "form_feed_sources"
_FORM_FEED = "\f"


def form_feed_spec() -> ToolSpec:
    """Advertised spec for sources that contain a form feed."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+000C. Does not return text.",
        mutates=False,
    )


def form_feed_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return form-feed count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+000C. Python ``splitlines``
    treats a form feed as a line break, so counts are taken from the raw text.
    """
    total = text.count(_FORM_FEED)
    runs = 0
    in_run = False
    for char in text:
        if char == _FORM_FEED:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return total, runs, text.startswith(_FORM_FEED), text.endswith(_FORM_FEED)


def form_feed_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+000C. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "form_feed_sources only accepts metadata"},
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
        total, runs, starts, ends = form_feed_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "form_feed_count": total,
                "run_count": runs,
                "starts_with_form_feed": starts,
                "ends_with_form_feed": ends,
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
