"""Non-breaking space listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "nbsp_sources"
_NBSP = "\u00a0"


def nbsp_spec() -> ToolSpec:
    """Advertised spec for sources that contain a non-breaking space."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+00A0. Does not return text.",
        mutates=False,
    )


def nbsp_counts(text: str) -> tuple[int, bool, bool]:
    """Return U+00A0 count and whether the text starts or ends with one."""
    return text.count(_NBSP), text.startswith(_NBSP), text.endswith(_NBSP)


def nbsp_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+00A0. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "nbsp_sources only accepts metadata"},
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
        count, leading, trailing = nbsp_counts(document.text)
        if count == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "nbsp_count": count,
                "leading": leading,
                "trailing": trailing,
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
