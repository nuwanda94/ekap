"""Word joiner listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "word_joiner_sources"
_WORD_JOINER = "\u2060"


def word_joiner_spec() -> ToolSpec:
    """Advertised spec for sources that contain a word joiner."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+2060. Does not return text.",
        mutates=False,
    )


def word_joiner_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return word-joiner count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+2060. BOM, zero-width
    spaces, and zero-width joiners or non-joiners are not counted.
    """
    total = text.count(_WORD_JOINER)
    runs = 0
    in_run = False
    for char in text:
        if char == _WORD_JOINER:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_WORD_JOINER),
        text.endswith(_WORD_JOINER),
    )


def word_joiner_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+2060. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "word_joiner_sources only accepts metadata"},
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
        total, runs, starts, ends = word_joiner_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "word_joiner_count": total,
                "run_count": runs,
                "starts_with_word_joiner": starts,
                "ends_with_word_joiner": ends,
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
