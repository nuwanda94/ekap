"""Replacement character listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "replacement_character_sources"
_REPLACEMENT = "\ufffd"


def replacement_character_spec() -> ToolSpec:
    """Advertised spec for sources that contain a replacement character."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+FFFD. Does not return text.",
        mutates=False,
    )


def replacement_character_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return replacement-character count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+FFFD. Ordinary spaces, em
    spaces, en spaces, medium mathematical spaces, Ogham space marks,
    Mongolian vowel separators, invisible separators, invisible times,
    invisible plus, function applications, combining grapheme joiners,
    braille pattern blanks, object replacement characters, and other
    whitespace or control characters are not counted.
    """
    total = text.count(_REPLACEMENT)
    runs = 0
    in_run = False
    for char in text:
        if char == _REPLACEMENT:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_REPLACEMENT),
        text.endswith(_REPLACEMENT),
    )


def replacement_character_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+FFFD. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "replacement_character_sources only accepts metadata"},
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
        total, runs, starts, ends = replacement_character_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "replacement_character_count": total,
                "run_count": runs,
                "starts_with_replacement_character": starts,
                "ends_with_replacement_character": ends,
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
