"""Carriage-return listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "crlf_sources"
_CR = "\r"


def crlf_spec() -> ToolSpec:
    """Advertised spec for sources that contain a carriage return."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+000D. Does not return text.",
        mutates=False,
    )


def crlf_counts(text: str) -> tuple[int, int, int]:
    """Return CR count, CRLF pair count, and bare CR count."""
    cr = text.count(_CR)
    crlf = text.count("\r\n")
    return cr, crlf, cr - crlf


def crlf_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+000D. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "crlf_sources only accepts metadata"},
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
        cr, crlf, bare = crlf_counts(document.text)
        if cr == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "cr_count": cr,
                "crlf_count": crlf,
                "bare_cr_count": bare,
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
