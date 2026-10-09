"""Arabic letter mark listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "arabic_letter_mark_sources"
_ARABIC_LETTER_MARK = "\u061c"


def arabic_letter_mark_spec() -> ToolSpec:
    """Advertised spec for sources that contain an Arabic letter mark."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+061C. Does not return text.",
        mutates=False,
    )


def arabic_letter_mark_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return Arabic-letter-mark count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+061C. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, and left-to-right
    or right-to-left marks are not counted.
    """
    total = text.count(_ARABIC_LETTER_MARK)
    runs = 0
    in_run = False
    for char in text:
        if char == _ARABIC_LETTER_MARK:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_ARABIC_LETTER_MARK),
        text.endswith(_ARABIC_LETTER_MARK),
    )


def arabic_letter_mark_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+061C. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "arabic_letter_mark_sources only accepts metadata"},
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
        total, runs, starts, ends = arabic_letter_mark_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "arabic_letter_mark_count": total,
                "run_count": runs,
                "starts_with_arabic_letter_mark": starts,
                "ends_with_arabic_letter_mark": ends,
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
