# Code Review: JSON/JSONL Input Support + Debug Step-by-Step Toolkit

**Rating: 7/10**

## Scope

- Files reviewed: 6
- LOC added: ~550 (estimated across all files)
- Focus: Two features -- JSON input loading and debug instrumentation
- Scout findings: metadata field disconnect, no file-size guard, logger state sharing

## Overall Assessment

Solid, well-structured implementation. JSON input validation is thorough with good error messages. Debug toolkit is cleanly gated behind `is_debug` checks so zero overhead in production. Backward compatibility is preserved -- all 43 relevant tests pass. A few medium-severity issues around metadata propagation, memory safety with large files, and logger state isolation need attention.

---

## Critical Issues

None found.

---

## High Priority

### H1. Metadata from JSON input is loaded but never propagated to Chunk objects

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/scripts/index_corpus.py` (lines 97-102)
**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/src/indexing/semantic_chunker.py` (lines 154, 238-266)

The `_validate_document_entry()` correctly extracts `metadata` from JSON docs (line 101), but `chunk_document()` and `chunk_documents()` never pass `doc["metadata"]` into the `Chunk` dataclass. The `Chunk.metadata` field (line 35) was added but always remains `None` during actual indexing because chunkers only access `doc["text"]` and `doc["id"]`.

**Impact:** Users who supply metadata in JSON input will silently lose it. The metadata field on `Chunk` is dead code in practice.

**Fix:** Either:
- (a) Thread metadata through the chunking pipeline: pass `doc.get("metadata")` into `chunk_document()` and set it on each `Chunk`, or
- (b) Remove `Chunk.metadata` until it is actually needed (YAGNI).

### H2. No file-size guard on JSON loading -- potential OOM

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/scripts/index_corpus.py` (lines 107-108)

`_load_json()` calls `json.load(f)` which reads the entire file into memory. A multi-GB JSON file would cause OOM without warning. The JSONL loader is naturally streaming-friendly, but the JSON loader is not.

**Impact:** Operational risk when users supply large datasets via `--input_json`.

**Fix:** Add a file-size check before loading:
```python
file_size = path.stat().st_size
MAX_JSON_SIZE = 500 * 1024 * 1024  # 500 MB
if file_size > MAX_JSON_SIZE:
    raise ValueError(f"JSON file too large ({file_size / 1e6:.0f} MB). Use JSONL format for large datasets.")
```

---

## Medium Priority

### M1. Logger instances share debug state across modules due to `logging.getLogger` caching

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/src/utils/logger.py` (lines 261-289)

`get_logger()` calls `logging.getLogger(name)` which returns cached instances. If `enable_debug()` is called on one logger (e.g., in `index_corpus.py`), other loggers obtained via `get_logger("different_name")` will NOT have debug enabled. This means `logger.is_debug` checks inside library code (like propagation) would return `False` even when the user passes `--debug`.

However, currently the code only checks `logger.is_debug` in the two script files (`index_corpus.py`, `query.py`) which each call `enable_debug()` on their own logger, so the issue is latent rather than active. It will surface if debug blocks are added to library files.

**Impact:** Latent bug. Debug instrumentation in future library code would silently not fire.

**Fix:** Consider a module-level debug flag or a singleton pattern:
```python
_GLOBAL_DEBUG = False

def enable_global_debug(output_dir: str, max_samples: int = 3):
    global _GLOBAL_DEBUG
    _GLOBAL_DEBUG = True
    # ... configure all loggers
```

### M2. `json as json_mod` aliasing reduces readability

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/src/utils/logger.py` (line 11)

The import `import json as json_mod` is unusual. The comment-free alias makes it non-obvious why standard `json` needs renaming. This was likely done to avoid shadowing with a local variable or parameter named `json`, but no such conflict exists in the file.

**Fix:** Use `import json` directly. If there is a specific reason for the alias, add a comment.

### M3. Debug checkpoint timing parsing is fragile

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/src/utils/logger.py` (lines 195-201)

The timing extraction from `stop_timer()` result string uses string parsing (`step_time.split(":")[-1].strip().rstrip("s")`). This relies on the exact format of `stop_timer()`'s return value. If `stop_timer()` changes its format, checkpoint timing silently breaks.

**Fix:** Return a float from `stop_timer()` instead of a formatted string, or have `debug_checkpoint` accept `step_time_s: Optional[float]` directly.

### M4. Duplicate doc IDs in JSON input are silently accepted

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/scripts/index_corpus.py` (lines 105-119)

`_load_json()` validates individual entries but does not check for duplicate `id` values. Duplicate IDs would cause manifest hash collisions and potentially corrupt incremental indexing state.

**Fix:** Add a uniqueness check:
```python
seen_ids = set()
for i, entry in enumerate(raw):
    doc = _validate_document_entry(entry, i)
    if doc["id"] in seen_ids:
        raise ValueError(f"Duplicate document id '{doc['id']}' at index {i}")
    seen_ids.add(doc["id"])
    documents.append(doc)
```

### M5. Temp files in tests not cleaned up

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/tests/test_json_input.py` (lines 79, 92, etc.)

Tests use `tempfile.NamedTemporaryFile(delete=False)` but never clean up the files. This leaks temp files on each test run.

**Fix:** Use `pytest`'s `tmp_path` fixture or wrap in `try/finally` with `os.unlink()`.

---

## Low Priority

### L1. Debug blocks in `index_corpus.py` add ~100 lines of inline code

The debug instrumentation is well-gated but makes `index_corpus()` quite long (~250 lines). Consider extracting debug blocks into a helper:
```python
def _debug_step_documents(logger, documents, elapsed):
    # ... all the debug logic
```

### L2. Inconsistent error message indexing in JSONL

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/scripts/index_corpus.py` (line 134)

`_load_jsonl()` passes `line_num - 1` as the `index` to `_validate_document_entry()`, making it 0-based after accounting for the 1-based `line_num`. But `_validate_document_entry` error messages say "Document at index N" -- this could confuse users who think in terms of line numbers.

### L3. `route_result["scores"]` access in `query.py` debug is unguarded

**File:** `/Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm/scripts/query.py` (lines 127-129)

The code checks `hasattr(route_result["scores"], '__len__')` but `route_result` may not have a `"scores"` key at all depending on the router implementation. The `"scores" in route_result` check on line 127 handles this, but the pattern is somewhat fragile if the router returns `scores=None`.

---

## Edge Cases Found by Scout

1. **Metadata disconnect**: JSON docs carry metadata, but chunks never receive it -- silent data loss
2. **Logger per-instance debug state**: Each `get_logger("name")` call returns a separate instance; `enable_debug()` on one does not affect others
3. **Pickle serialization of Chunk**: Adding `metadata: Optional[dict]` to the `Chunk` dataclass does not break existing pickled indexes (pickle handles new `None` defaults), but old indexes loaded and re-saved will now carry the `metadata=None` field -- safe but worth noting
4. **Duplicate doc IDs**: JSON input with duplicate IDs silently overwrites manifest entries

---

## Positive Observations

1. Clean validation with clear error messages in `_validate_document_entry()`
2. Proper no-op pattern for debug methods (early return when disabled) -- zero overhead in production
3. Good test coverage: 19 new tests covering validation edge cases, file loading, and debug lifecycle
4. Mutually exclusive `--input_dir` / `--input_json` CLI group prevents ambiguous input
5. JSONL loader correctly handles empty lines (line 129)
6. Backward-compatible: `metadata` field defaults to `None`, existing code unaffected
7. Debug checkpoints save both JSON-serializable data and numpy/sparse arrays intelligently

---

## Recommended Actions

1. **[H1]** Decide: propagate metadata through chunks or remove the field (YAGNI)
2. **[H2]** Add file-size guard for JSON loading or recommend JSONL in docs
3. **[M4]** Add duplicate ID detection in JSON/JSONL loaders
4. **[M5]** Fix temp file cleanup in tests
5. **[M1]** Document logger debug state scope; consider global debug flag for future use
6. **[M3]** Refactor `stop_timer()` to return float alongside formatted string

---

## Metrics

- Type Coverage: N/A (Python, no mypy configured)
- Test Coverage: 19 new tests, all passing
- Linting Issues: 0 syntax errors, builds clean
- Pre-existing failures: 2 NER tests (missing spacy in env), 1 KMeans segfault (sklearn issue)

---

## Unresolved Questions

1. Is `Chunk.metadata` intended to be used downstream (e.g., in generation or passage ranking)? If yes, the propagation path needs to be built. If no, the field should be removed per YAGNI.
2. Should debug mode be a global flag rather than per-logger-instance? This matters if debug instrumentation is planned for library modules beyond the two scripts.
3. Should the JSON loader enforce a maximum document count or total text size to prevent resource exhaustion?
