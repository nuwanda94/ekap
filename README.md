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
| **Evals** | 20+ golden questions scored automatically on every CI run |
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

Install is zero-runtime-deps by design. Dev extras pull in `pytest`, `pytest-cov`, and `ruff` only.

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

Golden questions plus an automatic scorer. CI invokes `python -m ekap.evals` so regressions fail the pipeline.

---

## Design principles

1. **One concern per change** — conventional commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`).
2. **Tests first for public behavior** — every feature lands with unit tests that fail without the new code.
3. **No secrets in unit tests** — inject fakes; no live LLM or network calls.
4. **Fail closed** — invalid tool args and missing dependencies do not partially execute side effects.
5. **Human approval for mutation** — drafts are cheap; execution is explicit.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local workflow and PR expectations.

---

## Status

Alpha reference implementation (v0.1.0). The surface area is intentionally small and testable so the control-plane patterns stay obvious.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
