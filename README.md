# EKAP — Enterprise Knowledge & Action Agent Platform

> A single, production-oriented multi-agent system that turns internal knowledge into **safe, auditable actions**.

[![CI](https://github.com/nuwanda94/ekap/actions/workflows/ci.yml/badge.svg)](https://github.com/nuwanda94/ekap/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

EKAP is deliberately built to prove five capabilities in one coherent codebase — not five disconnected demos.

| Capability | What it demonstrates |
|------------|----------------------|
| **RAG** | Answers *only* from approved documents and cites every claim |
| **MCP** | Exposes real tools any agent can call through the Model Context Protocol |
| **Planning** | Explicit plan → retrieve/search → synthesize, with full citation trail |
| **Evals** | 20+ golden questions in `src/ekap/evals/golden.py`, scored on every CI run |
| **Approval Gate** | Side-effecting actions are drafted only; a human must approve before execute |

Clear package boundaries, no hidden network or LLM calls in unit tests, and a human-in-the-loop control plane for anything that mutates state.

---

## Package layout

```text
src/ekap/
  rag/           # Ingester, Retriever, cite() — in-memory, pure Python
  mcp_server/    # In-process MCP server + tool catalog
  agents/        # Orchestrator (plan → retrieve → synthesize) + approval gate
  evals/         # Golden set + non-zero-on-regression runner
tests/           # Mirrors src/ekap/; no real keys or network required
```

Public APIs are typed (Python 3.12+). Side effects stay behind the approval-gate pattern once an action is drafted.

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest --cov=ekap -q
uv run python -m ekap.evals   # scores the golden set; exits non-zero on regression
```

CI is green only when **lint**, **unit tests**, and **golden evals** all succeed.

Install is zero-runtime-deps by design. Dev extras pull in `pytest`, `pytest-cov`, and `ruff` only.

---

## Minimal example

Ingest two sources, retrieve a ranked chunk, and surface a citation — no network, no LLM key:

```python
from ekap.rag import Ingester, Retriever

ingester = Ingester()
ingester.add(
    "policy-1",
    "Expense reports over 500 dollars require manager approval.",
    metadata={"kind": "policy"},
)
ingester.add(
    "handbook-1",
    "Vacation accrues at 1.5 days per month.",
    metadata={"kind": "handbook"},
)

citations = Retriever(ingester).query_with_citations(
    "Who must approve expense reports over 500?",
    top_k=1,
)
assert citations[0].source_id == "policy-1"
assert "manager approval" in citations[0].snippet
```

Plan → synthesize with a citation trail:

```python
from ekap.agents import Orchestrator
from ekap.rag import Ingester, Retriever

ingester = Ingester()
ingester.add("handbook-1", "Vacation accrues at 1.5 days per month.")
draft = Orchestrator(Retriever(ingester)).run("How fast does vacation accrue?")
assert draft.status == "draft"
assert draft.citations[0].source_id == "handbook-1"
assert "handbook-1" in draft.text
```

Mutating tool results stay pending until explicitly approved (see `ApprovalGate` and `queue_mutation_drafts` in `ekap.agents`).

---

## Core modules

### RAG (`ekap.rag`)

In-memory ingest and retrieval with citation objects (`source_id` + snippet). Suitable for unit tests and local demos; swap the store when you need persistence.

### MCP server (`ekap.mcp_server`)

In-process server with call-style tool invocation. Includes search/draft tools plus a large catalog of **read-only** Unicode and whitespace hygiene inspectors (e.g. `replacement_character_sources`, space/control/bidi markers). These tools never mutate the corpus and fail closed on bad arguments.

For the full tool list and contracts, see modules under `src/ekap/mcp_server/` — do not treat README as the catalog of record.

### Agents (`ekap.agents`)

- **Orchestrator** — structured plan, retrieval/tool use, draft answer with citation trail.
- **Approval gate** — draft → approve → execute, or reject with state unchanged.
- Source-removal helpers for controlled corpus edits under the same gate pattern.

### Evals (`ekap.evals`)

Golden questions in `src/ekap/evals/golden.py` plus an automatic scorer. CI invokes `python -m ekap.evals` so regressions fail the pipeline.

---

## Design principles

1. **One concern per change** — conventional commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`).
2. **Tests first for public behavior** — every feature lands with unit tests that fail without the new code.
3. **No secrets in unit tests** — inject fakes; no live LLM or network calls.
4. **Fail closed** — invalid tool args and missing dependencies do not partially execute side effects.
5. **Human approval for mutation** — drafts are cheap; execution is explicit.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local workflow and contribution expectations.

---

## Non-goals

- Production vector databases or embedding services (in-memory token overlap is intentional).
- Hosted MCP transports or multi-process servers (in-process tools only).
- Real LLM calls in unit tests or CI (inject fakes/stubs).
- Expanding the Unicode hygiene catalog without a clear product reason.

---

## Status

Alpha reference implementation (v0.1.0).

The **control plane** (RAG, orchestrator, approval gate, evals) is intentionally small and testable. The MCP package additionally ships an expanded **read-only** catalog of Unicode/whitespace inspectors used for corpus hygiene checks; that catalog is large by design of the tooling surface, not because the agent core is unbounded.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
