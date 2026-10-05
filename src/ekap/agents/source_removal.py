"""Gate executor that withdraws an ingested source only when invoked."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ekap.rag import Document, Ingester

Executor = Callable[[dict[str, Any]], Document]


def source_removal_executor(ingester: Ingester) -> Executor:
    """Return an executor the approval gate can run for ``remove_source``.

    The body is the mapping ``ApprovalGate.approve`` already builds. Only
    ``action == "remove_source"`` is handled; any other action fails closed and
    does not touch the corpus. This function does not call the executor.
    """
    if not isinstance(ingester, Ingester):
        raise TypeError("ingester must be an Ingester")

    def execute(body: dict[str, Any]) -> Document:
        if not isinstance(body, dict):
            raise TypeError("body must be a dict")
        action = body.get("action")
        target = body.get("target")
        if action != "remove_source":
            raise ValueError("executor only handles remove_source")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target is required")
        return ingester.remove(target)

    return execute
