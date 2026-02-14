---
title: "JSON Input Support + Debug Step-by-Step Toolkit"
description: "Add JSON/JSONL input for indexing and comprehensive debug mode for index/retrieve pipelines"
status: complete
priority: P2
effort: 4h
branch: main
tags: [feature, debug, indexing, retrieval]
created: 2026-02-14
---

# JSON Input Support + Debug Step-by-Step Toolkit

## Summary

Two features for HyP-DLM:

1. **JSON List Input** -- Enable `index_corpus.py` to accept a JSON/JSONL file as input instead of only a directory of `.txt`/`.md` files. Adds `--input_json` flag, schema validation, and integrates with incremental indexing.

2. **Debug Step-by-Step Toolkit** -- Add `--debug` flag to both `index_corpus.py` and `query.py` that prints detailed intermediate results after each pipeline step, saves checkpoints, and produces a JSON debug report with timing.

## Phases

| Phase | Title | Status | Effort | Files Modified |
|-------|-------|--------|--------|----------------|
| 01 | [JSON Input Support](./phase-01-json-input-support.md) | complete | 1.5h | `index_corpus.py`, `semantic_chunker.py` |
| 02 | [Debug Config & Logger](./phase-02-debug-config-and-logger.md) | complete | 1h | `logger.py`, `default.yaml`, `index_corpus.py`, `query.py` |
| 03 | [Indexing Debug](./phase-03-indexing-debug.md) | complete | 0.75h | `index_corpus.py` |
| 04 | [Retrieval Debug](./phase-04-retrieval-debug.md) | complete | 0.75h | `query.py` |

## Dependencies

- Phase 02 must complete before Phase 03 and 04 (logger methods needed)
- Phase 01 is independent; can be done in parallel with Phase 02

## Key Decisions

- **Metadata on Chunk**: Add optional `metadata: dict` field to `Chunk` dataclass. Default `None`, backward compatible.
- **Debug output location**: `data/debug/` directory (configurable via `debug.output_dir`)
- **No new files**: All changes go into existing files. Debug helpers added to `logger.py`.
- **JSONL support**: Read line-by-line for streaming large datasets. JSON read all at once.

## Constraints

- Must not break existing `--input_dir` workflow
- Debug mode must not affect non-debug performance (guard all debug code behind `if debug:`)
- Checkpoint serialization: use JSON for inspectable data, pickle only for non-serializable objects
- All new config keys have sensible defaults so existing `default.yaml` works without changes

## Testing

- Unit test for `load_documents_json()` with valid/invalid JSON
- Unit test for `load_documents_jsonl()` with valid/invalid JSONL
- Integration test: JSON input -> full index pipeline
- Debug mode smoke test: verify checkpoint files created, no crashes
