"""BOM (U+FEFF) listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "bom_sources"
_BOM = "\ufeff"


def bom_spec() -> ToolSpec:
    """Advertised spec for sources that contain U+FEFF."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+FEFF. Does not return text.",
        mutates=False,
    )


def bom_counts(text: str) -> tuple[int, bool]:
    """Count U+FEFF characters and whether the text starts with one."""
    return text.count(_BOM), text.startswith(_BOM)


def bom_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+FEFF. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "bom_sources only accepts metadata"},
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
        count, leading = bom_counts(document.text)
        if count == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "bom_count": count,
                "leading": leading,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={"sources": len(documents), "flagged": flagged, "executed": False},
    )
