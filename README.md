# EKAP — Enterprise Knowledge & Action Agent Platform

> A single, production-oriented multi-agent system that turns internal knowledge into **safe, auditable actions**.

EKAP is deliberately built to prove five capabilities in one coherent codebase:

| Capability | What it demonstrates |
|------------|----------------------|
| **RAG** | Answers *only* from approved documents and cites every claim |
| **MCP** | Exposes real tools any agent can call through the Model Context Protocol |
| **Planning** | Explicit plan → retrieve/search → synthesize, with full citation trail |
| **Evals** | 20+ golden questions scored automatically on every PR |
| **Approval Gate** | Side-effecting actions are drafted only; a human must click Approve |

This is not five disconnected demos. It is one system with clear boundaries, auditability, and a human-in-the-loop control plane.

---

## Architecture (high level)

```
User / API Request
        │
        ▼
┌───────────────────────┐
│  Orchestrator Agent   │  ← Planning
│  (plan → retrieve →   │
│   tool-use → draft)   │
└───────────┬───────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌─────────┐   ┌──────────────┐
│ RAG     │   │ MCP Server   │  ← real tools
│ Layer   │   │ (2–4 tools)  │
└────┬────┘   └──────┬───────┘
     │               │
     └───────┬───────┘
             ▼
┌───────────────────────┐
│ Draft Answer / Action │
│ + full citation trail │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Approval Gate         │  ← human-in-the-loop
│ (pending until Approve)│
└───────────┬───────────┘
            │ (only after Approve)
            ▼
      Execute / Respond
            │
            ▼
┌───────────────────────┐
│ Eval Harness          │  ← CI-gated
│ (golden set + scoring)│
└───────────────────────┘
```

---

## Repository Layout (target)

```
ekap/
── agents/
│   ── orchestrator.py      # planning + research loop
│   ── approval_gate.py
── mcp_server/
│   ── tools/
│   ── server.py
── rag/
│   ── ingest.py
│   ── retriever.py
── evals/
│   ── golden_set.json
│   ── runner.py
── ui/                      # approval dashboard or Slack handler
── api/
── .github/workflows/
│   ── ci.yml
── README.md
```

---

## Development Model

This repository is advanced by an **hourly automation** that opens exactly one PR per run.

- PR titles are conventional: `chore:`, `feat:`, `fix:`, or `test:`
- Every change lands via pull request
- CI must be green before merge
- Features are implemented first; the subsequent run writes the corresponding tests when needed

**Do not push directly to `main`.** All work goes through PRs.

---

## Roadmap (in order)

1. Project skeleton + GitHub Actions CI  
2. RAG core (ingest, retrieve, citations)  
3. MCP server with 2–3 real tools  
4. Orchestrator / planning agent  
5. Approval-gate agent  
6. Eval harness (20 golden questions + automatic scoring)

Later: multi-tenancy, richer observability, Slack/Teams approval UI, production packaging.

---

## Local Development (once skeleton lands)

```bash
# Python 3.12+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run CI-equivalent checks locally
# (exact commands will appear in the first chore PR)
```

---

## Design Principles

- **Citation fidelity over fluency** — if it cannot be cited, it is not answered as fact.
- **Tools are the only side-effect path** — no hidden writes.
- **Human approval for mutation** — drafts are cheap; execution is gated.
- **Evaluation is a first-class citizen** — regressions fail the build.
- **Small, reviewable PRs** — one concern per change.

---

## Status

This repository is under active, automated construction.  
Watch the [Pull Requests](https://github.com/nuwanda94/ekap/pulls) tab for the current pulse of the system.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
