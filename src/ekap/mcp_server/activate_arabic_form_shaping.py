"""Activate arabic form shaping listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "activate_arabic_form_shaping_sources"
_ACTIVATE_ARABIC_FORM_SHAPING = "\u206d"


def activate_arabic_form_shaping_spec() -> ToolSpec:
    """Advertised spec for sources that contain activate arabic form shaping."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+206D. Does not return text.",
        mutates=False,
    )


def activate_arabic_form_shaping_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return activate-arabic-form-shaping count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+206D. BOM, zero-width
    spaces, word joiners, zero-width joiners or non-joiners, left-to-right
    or right-to-left marks, Arabic letter marks, left-to-right or
    right-to-left embeddings, pop directional formatting, left-to-right or
    right-to-left overrides, left-to-right, right-to-left, or first strong
    isolates, pop directional isolates, inhibit symmetric swapping,
    activate symmetric swapping, and inhibit arabic form shaping are not
    counted.
    """
    total = text.count(_ACTIVATE_ARABIC_FORM_SHAPING)
    runs = 0
    in_run = False
    for char in text:
        if char == _ACTIVATE_ARABIC_FORM_SHAPING:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_ACTIVATE_ARABIC_FORM_SHAPING),
        text.endswith(_ACTIVATE_ARABIC_FORM_SHAPING),
    )


def activate_arabic_form_shaping_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+206D. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "activate_arabic_form_shaping_sources only accepts metadata"},
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
        total, runs, starts, ends = activate_arabic_form_shaping_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "activate_arabic_form_shaping_count": total,
                "run_count": runs,
                "starts_with_activate_arabic_form_shaping": starts,
                "ends_with_activate_arabic_form_shaping": ends,
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
