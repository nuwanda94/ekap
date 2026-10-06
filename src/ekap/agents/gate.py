"""Human approval gate. Mutating actions stay pending until explicitly approved."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

Executor = Callable[[dict[str, Any]], Any]
_STATUSES = frozenset({"pending", "executed", "rejected", "cancelled"})


@dataclass(frozen=True, slots=True)
class PendingAction:
    """A side effect that has not run until status is executed."""

    action_id: str
    action: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One audited gate transition. Does not include executor results."""

    action_id: str
    action: str
    target: str
    status: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GateSummary:
    """Counts of recorded actions. Does not include payloads or executor results."""

    pending: int = 0
    executed: int = 0
    rejected: int = 0
    cancelled: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.executed + self.rejected + self.cancelled


class ApprovalGate:
    """draft → approve → execute, or reject/cancel without touching external state."""

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor
        self._actions: dict[str, PendingAction] = {}
        self._decisions: list[DecisionRecord] = []
        self.executed: list[dict[str, Any]] = []

    def draft(
        self,
        action: str,
        target: str,
        payload: Mapping[str, Any] | None = None,
    ) -> PendingAction:
        """Record a pending action. Does not call the executor.

        The returned action is a copy. Mutating its payload does not change
        the stored record.
        """
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
        self._record(pending)
        return _copy_action(pending)

    def submit_draft(self, draft: Mapping[str, Any]) -> PendingAction:
        """Queue an already-built tool draft. Does not execute it.

        Only a pending draft is accepted. A cancelled or executed tool draft
        fails closed and is not recorded.
        """
        action = draft.get("action", "")
        target = draft.get("target", "")
        if not isinstance(action, str) or not isinstance(target, str):
            raise TypeError("draft action and target must be strings")
        status = draft.get("status", "pending")
        if not isinstance(status, str):
            raise TypeError("draft status must be a string")
        if status != "pending":
            raise ValueError(f"draft is {status}, not pending")
        extra = {
            key: value
            for key, value in draft.items()
            if key not in {"action", "target", "status", "executed"}
        }
        return self.draft(action, target, extra)

    def approve(self, action_id: str) -> PendingAction:
        """Execute a pending action exactly once.

        The returned action is a copy. Mutating its payload does not change
        the stored record.
        """
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
        self._record(executed)
        return _copy_action(executed)

    def reject(self, action_id: str, reason: str = "") -> PendingAction:
        """Mark a pending action rejected. Does not call the executor."""
        return self._close(action_id, "rejected", reason)

    def cancel(self, action_id: str, reason: str = "") -> PendingAction:
        """Mark a pending action cancelled. Does not call the executor."""
        return self._close(action_id, "cancelled", reason)

    def get(self, action_id: str) -> PendingAction:
        """Return one recorded action. Does not call the executor.

        The returned action is a copy, so an audit cannot mutate the stored
        payload.
        """
        if action_id not in self._actions:
            raise KeyError(f"unknown action: {action_id}")
        return _copy_action(self._actions[action_id])

    def actions(self, status: str | None = None) -> tuple[PendingAction, ...]:
        """Return recorded actions in insertion order. Does not call the executor.

        An optional status keeps only matching actions. A blank or unknown
        status fails closed. Payloads are copies, so an audit cannot mutate
        the stored action.
        """
        cleaned = _clean_status(status)
        listed: list[PendingAction] = []
        for action in self._actions.values():
            if cleaned is not None and action.status != cleaned:
                continue
            listed.append(_copy_action(action))
        return tuple(listed)

    def summary(self) -> GateSummary:
        """Count recorded actions by status. Does not call the executor."""
        counts = {status: 0 for status in _STATUSES}
        for action in self._actions.values():
            counts[action.status] += 1
        return GateSummary(
            pending=counts["pending"],
            executed=counts["executed"],
            rejected=counts["rejected"],
            cancelled=counts["cancelled"],
        )

    def decisions(self) -> tuple[DecisionRecord, ...]:
        """Return gate transitions in the order they were recorded."""
        return tuple(self._decisions)

    def _close(self, action_id: str, status: str, reason: str) -> PendingAction:
        current = self._require_pending(action_id)
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        closed = PendingAction(
            action_id=current.action_id,
            action=current.action,
            target=current.target,
            payload=dict(current.payload),
            status=status,
            reason=reason.strip(),
        )
        self._actions[action_id] = closed
        self._record(closed)
        return _copy_action(closed)

    def _record(self, action: PendingAction) -> None:
        self._decisions.append(
            DecisionRecord(
                action_id=action.action_id,
                action=action.action,
                target=action.target,
                status=action.status,
                reason=action.reason,
            )
        )

    def _require_pending(self, action_id: str) -> PendingAction:
        current = self.get(action_id)
        if current.status != "pending":
            raise ValueError(f"action {action_id} is {current.status}, not pending")
        return current


def _copy_action(action: PendingAction) -> PendingAction:
    return PendingAction(
        action_id=action.action_id,
        action=action.action,
        target=action.target,
        payload=dict(action.payload),
        status=action.status,
        reason=action.reason,
    )


def _clean_status(status: str | None) -> str | None:
    if status is None:
        return None
    if not isinstance(status, str):
        raise TypeError("status must be a string")
    cleaned = status.strip()
    if cleaned not in _STATUSES:
        raise ValueError("status must be pending, executed, rejected, or cancelled")
    return cleaned


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()
