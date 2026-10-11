# Changelog

All notable changes to EKAP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-11

### Added
- In-memory RAG core: `Ingester`, `Retriever`, and `cite()` helper with citation objects (source id + snippet).
- In-process MCP server with tool catalog including `search_docs`, draft actions, and read-only Unicode/whitespace hygiene inspectors.
- Orchestrator for plan → retrieve/tool → synthesize with full citation trail and run trace.
- Approval gate: draft → approve → execute path; reject leaves state unchanged.
- Evals: 22 golden questions and runner that exits non-zero on regression; invoked from CI.
- Source-removal helpers under the approval-gate pattern.
- Zero-runtime-deps package layout, typed public APIs, ruff + pytest + golden-evals CI.
- `Ingester.clear()` to reset the in-memory corpus while preserving chunk settings.

### Notes
- Alpha reference implementation. No network calls or real LLM keys required for unit tests.
- Side effects stay behind the approval gate.
