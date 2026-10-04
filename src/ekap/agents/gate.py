"""Human approval gate. Mutating actions stay pending until explicitly approved."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

Executor = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class PendingAction:
    """A side effect that has not run until status is executed."""

    action_id: str
    action: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    reason: str = ""


class ApprovalGate:
    """draft → approve → execute, or reject without touching external state."""

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor
        self._actions: dict[str, PendingAction] = {}
        self.executed: list[dict[str, Any]] = []

    def draft(
        self,
        action: str,
        target: str,
        payload: Mapping[str, Any] | None = None,
    ) -> PendingAction:
        """Record a pending action. Does not call the executor."""
        cleaned_action = _require_text(action, "action")
        cleaned_target = _require_text(target, "target")
        action_id = f"action-{len(self._actions) + 1}"
        pending = PendingAction(
            action_id=action_id,
            action=cleaned_action,
            target=cleaned_target,
            payload=dict(payload or {}),
        )
        self._actions[action_id] = pending
        return pending

    def submit_draft(self, draft: Mapping[str, Any]) -> PendingAction:
        """Queue an already-built tool draft. Does not execute it."""
        action = draft.get("action", "")
        target = draft.get("target", "")
        if not isinstance(action, str) or not isinstance(target, str):
            raise TypeError("draft action and target must be strings")
        extra = {
            key: value
            for key, value in draft.items()
            if key not in {"action", "target", "status", "executed"}
        }
        return self.draft(action, target, extra)

    def approve(self, action_id: str) -> PendingAction:
        """Execute a pending action exactly once."""
        current = self._require_pending(action_id)
        if self._executor is None:
            raise RuntimeError("no executor configured")
        body = {
            "action_id": current.action_id,
            "action": current.action,
            "target": current.target,
            "payload": dict(current.payload),
        }
        self._executor(body)
        self.executed.append(body)
        executed = PendingAction(
            action_id=current.action_id,
            action=current.action,
            target=current.target,
            payload=dict(current.payload),
            status="executed",
        )
        self._actions[action_id] = executed
        return executed

    def reject(self, action_id: str, reason: str = "") -> PendingAction:
        """Mark a pending action rejected. Does not call the executor."""
        current = self._require_pending(action_id)
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        rejected = PendingAction(
            action_id=current.action_id,
            action=current.action,
            target=current.target,
            payload=dict(current.payload),
            status="rejected",
            reason=reason.strip(),
        )
        self._actions[action_id] = rejected
        return rejected

    def get(self, action_id: str) -> PendingAction:
        if action_id not in self._actions:
            raise KeyError(f"unknown action: {action_id}")
        return self._actions[action_id]

    def _require_pending(self, action_id: str) -> PendingAction:
        current = self.get(action_id)
        if current.status != "pending":
            raise ValueError(f"action {action_id} is {current.status}, not pending")
        return current


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()
