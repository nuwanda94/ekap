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
- `Ingester.__len__` reports the number of ingested documents.
- `Ingester.source_ids` returns source ids in insertion order.
- `Ingester.__contains__` supports membership testing for source ids (whitespace-tolerant for strings; returns False for non-strings).
- `Ingester.get` returns a document by source id or None.
- `Ingester.chunks_for` returns the chunks for a given source id (empty if absent).
- `Ingester.chunk_count` reports the number of chunks currently stored.
- `Ingester.character_count` reports the total number of characters across all documents.
- `Ingester.chunk_mean_chars` reports the mean character length of current chunks (0.0 if none).
- `Ingester.chunk_max_chars` reports the maximum character length of current chunks (0 if none).
- `Ingester.chunk_min_chars` reports the minimum character length of current chunks (0 if none).
- `Ingester.chunk_total_chars` reports the total number of characters across all current chunks (0 if none).
- `Ingester.chunk_median_chars` reports the median character length of current chunks (0.0 if none).
- `Ingester.chunk_stdev_chars` reports the population standard deviation of chunk character lengths (0.0 if fewer than 2 chunks).
- `Ingester.chunk_variance_chars` reports the population variance of chunk character lengths (0.0 if fewer than 2 chunks).
- `Ingester.chunk_range_chars` reports the range (max - min) of chunk character lengths (0 if fewer than 2 chunks).
- `Ingester.chunk_mode_chars` reports the most common chunk character length (smallest if multimodal; 0 if none).
- `Ingester.chunk_mode_count` reports the number of chunks that have the mode character length (0 if none).
- `Ingester.chunk_length_histogram` reports the distribution of chunk character lengths as a dict of length → count (empty if none; keys sorted).
- `Ingester.chunk_unique_length_count` reports the number of distinct chunk character lengths (0 if none).
- `Ingester.chunk_iqr_chars` reports the interquartile range (Q3 − Q1) of chunk character lengths via `statistics.quantiles` (0.0 if fewer than 2 chunks).
- `Ingester.chunk_mad_chars` reports the median absolute deviation of chunk character lengths (0.0 if none).
- `Ingester.chunk_cv_chars` reports the coefficient of variation (population stdev / mean) of chunk character lengths (0.0 if fewer than 2 chunks or mean is 0).
- `Ingester.chunk_skewness_chars` reports the population skewness (third central moment / stdev³) of chunk character lengths (0.0 if fewer than 3 chunks or variance is 0).
- `Ingester.chunk_kurtosis_chars` reports the population kurtosis (fourth central moment / variance²) of chunk character lengths (0.0 if fewer than 4 chunks or variance is 0).

### Notes
- Alpha reference implementation. No network calls or real LLM keys required for unit tests.
- Side effects stay behind the approval gate.
