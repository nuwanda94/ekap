# Changelog

All notable changes to EKAP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0),
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
- `Ingester.clear()`, `__len__`, `source_ids`, `__contains__`, and `get` for corpus management.
- Observability properties on `Ingester` for chunk character-length statistics (mean, max, min, median, stdev, variance, range, mode, histogram, IQR, MAD, CV, skewness, kurtosis, excess kurtosis, geometric/harmonic/quadratic/cubic/quartic/quintic/sextic/septic/octic/nonic/decic means, and related counters).

### Changed
- Package structure under `src/ekap/` with mirrored tests.

## [Unreleased]

### Added
- `Ingester.chunk_undecic_mean_chars` for undecic mean observability of chunk character lengths.
- `Ingester.chunk_duodecic_mean_chars` for duodecic mean observability of chunk character lengths.
- `Ingester.chunk_tredecic_mean_chars` for tredecic mean observability of chunk character lengths.
- `Ingester.chunk_quattuordecic_mean_chars` for quattuordecic mean observability of chunk character lengths.
- `Ingester.chunk_quindecic_mean_chars` for quindecic mean observability of chunk character lengths.
- `Ingester.chunk_mean_tokens` for mean token count per chunk (using the retrieval tokenizer).

### Fixed
- Restored valid Python syntax across RAG core and tests after incremental observability additions.
