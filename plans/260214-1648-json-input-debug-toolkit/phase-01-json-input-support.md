# Phase 01: JSON List Input Support

## Context Links

- [plan.md](./plan.md)
- [index_corpus.py](../../hyp_dlm/scripts/index_corpus.py) -- main file to modify
- [semantic_chunker.py](../../hyp_dlm/src/indexing/semantic_chunker.py) -- Chunk dataclass
- [CLAUDE.md](../../CLAUDE.md) -- architecture reference

## Overview

- **Priority**: P2
- **Status**: complete
- **Effort**: 1.5h
- **Description**: Add JSON and JSONL file input support to `index_corpus.py` so users can load documents from structured files instead of only from a directory of `.txt`/`.md` files.

## Key Insights

- Current `load_documents()` returns `list[dict[str, str]]` with keys `id`, `text`, `path`. New loaders must return the same shape.
- `build_manifest()` already uses `doc["id"]` and `doc["text"]` -- JSON input just needs to conform to that dict shape.
- `Chunk` dataclass has no `metadata` field. Adding one (optional, default `None`) enables passing through user-provided metadata from JSON without breaking existing code.
- JSONL is preferred for large datasets (streaming, line-by-line) while JSON is simpler for small datasets.

## Requirements

### Functional
1. New CLI flag `--input_json <path>` mutually exclusive with `--input_dir`
2. Support `.json` files containing a JSON array of document objects
3. Support `.jsonl` files with one document object per line
4. Auto-detect format by file extension (`.json` vs `.jsonl`)
5. Validate required fields: `id` (str), `text` (str)
6. Accept optional fields: `metadata` (dict), `path` (str)
7. Propagate `metadata` through to `Chunk` dataclass and `metadata.pkl`
8. Work with both full-rebuild and incremental indexing

### Non-Functional
- Schema validation should produce clear error messages with line/index numbers
- Loading 10k+ documents from JSONL should not load entire file into memory at once

## Architecture

```
CLI args
  |
  v
--input_json path.json    --input_dir ./data/raw
  |                              |
  v                              v
load_documents_json()      load_documents()
  |                              |
  +--------- same shape --------+
  |
  v
list[dict]  ->  rest of pipeline unchanged
```

No new files. All changes in `index_corpus.py` + minor `Chunk` dataclass update.

## Related Code Files

### Files to Modify
- `hyp_dlm/scripts/index_corpus.py` -- Add `load_documents_json()`, `load_documents_jsonl()`, update CLI args, update `main()`
- `hyp_dlm/src/indexing/semantic_chunker.py` -- Add `metadata: Optional[dict] = None` to `Chunk` dataclass

### Files NOT Modified
- `config/default.yaml` -- No config needed; this is a CLI-level feature
- `manifest.json` -- Already works with `doc["id"]` and `doc["text"]`; no schema change

## Implementation Steps

### Step 1: Update Chunk dataclass (semantic_chunker.py)

Add optional metadata field to `Chunk`:

```python
@dataclass
class Chunk:
    """A semantic chunk = one hyperedge in the hypergraph."""
    text: str
    doc_id: str
    chunk_id: int
    start_char: int
    end_char: int
    entities: list = field(default_factory=list)
    metadata: Optional[dict] = field(default=None)
```

This is backward-compatible -- existing code that doesn't pass `metadata` gets `None`.

### Step 2: Add JSON schema validation (index_corpus.py)

Add a validation function:

```python
def _validate_document_entry(entry: dict, index: int) -> dict:
    """Validate a single document entry from JSON input.

    Args:
        entry: Raw dict from JSON
        index: Position in file (for error messages)

    Returns:
        Normalized document dict with keys: id, text, path, metadata

    Raises:
        ValueError: If required fields missing or wrong type
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Document at index {index}: expected dict, got {type(entry).__name__}")
    if "id" not in entry:
        raise ValueError(f"Document at index {index}: missing required field 'id'")
    if "text" not in entry:
        raise ValueError(f"Document at index {index}: missing required field 'text'")
    if not isinstance(entry["id"], str):
        raise ValueError(f"Document at index {index}: 'id' must be string")
    if not isinstance(entry["text"], str):
        raise ValueError(f"Document at index {index}: 'text' must be string")
    if not entry["text"].strip():
        raise ValueError(f"Document at index {index} (id='{entry['id']}'): 'text' is empty")

    return {
        "id": entry["id"],
        "text": entry["text"],
        "path": entry.get("path", ""),
        "metadata": entry.get("metadata"),
    }
```

### Step 3: Add load_documents_json() (index_corpus.py)

```python
def load_documents_json(json_path: str) -> list[dict]:
    """Load documents from a JSON file (array of objects) or JSONL file.

    Auto-detects format by file extension:
      - .json  -> parse as JSON array
      - .jsonl -> parse line-by-line

    Args:
        json_path: Path to .json or .jsonl file

    Returns:
        List of document dicts with keys: id, text, path, metadata
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON input file not found: {json_path}")

    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    else:
        return _load_json(path)
```

### Step 4: Implement _load_json() and _load_jsonl()

```python
def _load_json(path: Path) -> list[dict]:
    """Load from a .json file containing an array of document objects."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"JSON file must contain a top-level array, got {type(raw).__name__}")

    documents = []
    for i, entry in enumerate(raw):
        doc = _validate_document_entry(entry, i)
        documents.append(doc)

    logger.step("LoadDocuments", f"Loaded {len(documents)} documents from {path}")
    return documents


def _load_jsonl(path: Path) -> list[dict]:
    """Load from a .jsonl file with one document object per line."""
    documents = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {line_num}: {e}") from e
            doc = _validate_document_entry(entry, line_num - 1)
            documents.append(doc)

    logger.step("LoadDocuments", f"Loaded {len(documents)} documents from {path} (JSONL)")
    return documents
```

### Step 5: Update CLI argument parser (index_corpus.py)

In `main()`, update argparse:

```python
# Create mutually exclusive group for input source
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input_dir", help="Directory with raw documents (.txt, .md)")
input_group.add_argument("--input_json", help="JSON or JSONL file with document objects")
```

### Step 6: Update main() dispatch logic

```python
# Determine input source
if args.input_json:
    documents = load_documents_json(args.input_json)
    # Pass documents directly to pipeline (bypass load_documents in index_corpus)
else:
    documents = None  # Let index_corpus/index_corpus_incremental handle loading

# Update index_corpus() and index_corpus_incremental() signatures:
# Add optional `documents` param so they can accept pre-loaded docs
```

### Step 7: Refactor index_corpus() to accept pre-loaded documents

Modify `index_corpus()` signature:

```python
def index_corpus(input_dir: str, output_dir: str, config: dict,
                 documents: Optional[list[dict]] = None) -> None:
    # ...
    if documents is None:
        documents = load_documents(input_dir)
    # rest unchanged
```

Same for `index_corpus_incremental()`.

### Step 8: Update main() to pass documents through

```python
if args.input_json:
    documents = load_documents_json(args.input_json)
    input_source = args.input_json  # for logging
else:
    documents = None
    input_source = args.input_dir

if incremental_enabled and not args.full_rebuild:
    index_corpus_incremental(args.input_dir or "", args.output_dir, config, documents=documents)
else:
    index_corpus(args.input_dir or "", args.output_dir, config, documents=documents)
```

## Todo List

- [ ] Add `metadata: Optional[dict] = None` to `Chunk` dataclass
- [ ] Add `_validate_document_entry()` function
- [ ] Add `load_documents_json()` function
- [ ] Add `_load_json()` helper
- [ ] Add `_load_jsonl()` helper
- [ ] Update CLI with `--input_json` flag (mutually exclusive with `--input_dir`)
- [ ] Refactor `index_corpus()` to accept optional `documents` param
- [ ] Refactor `index_corpus_incremental()` to accept optional `documents` param
- [ ] Update `main()` dispatch logic
- [ ] Write unit tests for JSON/JSONL loading
- [ ] Write integration test: JSON -> full index pipeline

## Success Criteria

1. `python scripts/index_corpus.py --input_json data/test.json --output_dir data/indexed` works
2. `python scripts/index_corpus.py --input_json data/test.jsonl --output_dir data/indexed` works
3. `--input_dir` and `--input_json` cannot be used together (argparse error)
4. Invalid JSON schema produces clear error with index/line number
5. Empty text documents are rejected with clear error
6. Metadata field flows through to `metadata.pkl`
7. Incremental indexing works with JSON input
8. All existing tests still pass

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Breaking existing `--input_dir` CLI | High | Low | `mutually_exclusive_group(required=True)` preserves old behavior |
| Chunk metadata breaking pickle compat | Medium | Low | Default `None`, old pickles won't have field but dataclass handles missing |
| Large JSON OOM | Medium | Low | JSONL alternative for large datasets; document in help text |
| Duplicate document IDs in JSON | Low | Medium | Add warning log but don't error (last-write-wins via manifest) |

## Security Considerations

- Validate JSON input before processing (no arbitrary code execution)
- File path in `metadata` is user-provided; do not use it for file I/O operations
- Limit JSON file size? Not needed -- OS handles this; JSONL streams anyway

## Next Steps

- After this phase: debug toolkit phases can proceed
- Future: Consider adding CSV input support if requested
- Future: Consider `--input_url` for remote JSON endpoints
