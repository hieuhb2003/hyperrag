# Phase 03: Indexing Pipeline Debug Output

## Context Links

- [plan.md](./plan.md)
- [phase-02-debug-config-and-logger.md](./phase-02-debug-config-and-logger.md) -- prerequisite (logger methods)
- [index_corpus.py](../../hyp_dlm/scripts/index_corpus.py) -- main file to modify
- [logger.py](../../hyp_dlm/src/utils/logger.py) -- debug methods from Phase 02

## Overview

- **Priority**: P2
- **Status**: complete
- **Effort**: 0.75h
- **Description**: Add debug output after each step of `index_corpus()` and `index_corpus_incremental()`. When `--debug` is active, the pipeline prints detailed intermediate results and saves checkpoints.

## Key Insights

- The `index_corpus()` function has 10 clearly numbered steps. Debug output goes between existing steps -- no structural changes needed.
- All debug calls are guarded by `if logger.is_debug:` so zero overhead when debug is off.
- Checkpoints use `logger.debug_checkpoint()` from Phase 02 which handles JSON/numpy/sparse serialization.
- Timing: wrap each step with `logger.start_timer()` / `logger.stop_timer()` and pass elapsed to checkpoint.
- Incremental pipeline (`index_corpus_incremental()`) gets the same debug calls for its steps, reusing the same patterns.

## Requirements

### Functional

Debug output after each indexing step:

| Step | What to Show | Checkpoint Data |
|------|-------------|-----------------|
| 1. Load Documents | Doc count, sample IDs, char distribution | `{doc_count, doc_ids, char_stats}` |
| 2. Chunking | Chunk count, sample chunks (first 3), token distribution | `{chunk_count, token_stats, samples}` |
| 3. NER | Sample entities per chunk (first 3 chunks), entity type distribution | `{entity_count, type_distribution, samples}` |
| 4. Dedup | Unique entity count, dedup ratio, sample entities | `{unique_count, dedup_ratio}` |
| 5. Embeddings | Shape, norm stats, sample norms | `{entity_emb_shape, hyperedge_emb_shape, norm_stats}` |
| 6. Entity Linking | S matrix detail, synonym pairs found (sample), link count | `{S_stats, sample_synonyms, synonym_count}` |
| 7. Hypergraph Build | H matrix detail, isolated entities, empty hyperedges | `{H_stats, isolated, empty}` |
| 8. Masking | Strategy name, cluster count/distribution, coverage stats | `{strategy, cluster_stats}` |
| 9. Bipartite | Node/edge counts | `{entity_nodes, hyperedge_nodes, edges}` |
| 10. Save | File list, total disk size | `{files, total_bytes}` |

### Non-Functional
- Debug blocks should be visually distinct (Rich panels with yellow/magenta borders)
- Each checkpoint produces a separate JSON file in `data/debug/`
- Total pipeline timing reported at the end via `logger.debug_report()`

## Architecture

No new files. All changes are inline in `index_corpus()` function body, adding debug blocks between existing steps.

```
index_corpus()
  |
  Step 1: load_documents()
  |-- if debug: logger.debug_sample("Documents", docs)
  |             logger.debug_distribution("Doc Sizes", ...)
  |             logger.debug_checkpoint("step01_documents", ...)
  |
  Step 2: chunking
  |-- if debug: logger.debug_sample("Chunks", chunks)
  |             logger.debug_distribution("Token Distribution", ...)
  |             logger.debug_checkpoint("step02_chunking", ...)
  |
  ... (same pattern for steps 3-10)
  |
  End: logger.debug_report()
```

## Related Code Files

### Files to Modify
- `hyp_dlm/scripts/index_corpus.py` -- Add debug blocks after each step in `index_corpus()` and `index_corpus_incremental()`

### Files NOT Modified
- `logger.py` -- Methods already added in Phase 02
- `config/default.yaml` -- Debug section already added in Phase 02

## Implementation Steps

### Step 1: Add debug helper for document stats

Add a small helper function at module level (DRY -- reused by both full and incremental):

```python
def _debug_documents(documents: list[dict]) -> None:
    """Debug output for loaded documents."""
    if not logger.is_debug:
        return

    logger.debug_sample("Loaded Documents", [
        f"id={d['id']}, chars={len(d['text'])}, path={d.get('path', 'N/A')}"
        for d in documents
    ])

    # Character count distribution
    char_counts = [len(d["text"]) for d in documents]
    logger.debug_distribution("Document Sizes (chars)", {
        "< 1K": sum(1 for c in char_counts if c < 1000),
        "1K-5K": sum(1 for c in char_counts if 1000 <= c < 5000),
        "5K-10K": sum(1 for c in char_counts if 5000 <= c < 10000),
        "10K+": sum(1 for c in char_counts if c >= 10000),
    })
```

### Step 2: Add debug block after Step 2 (Chunking)

Insert after the existing `logger.step("IndexCorpus", f"Step 2 complete: ...")`:

```python
if logger.is_debug:
    logger.start_timer("debug_step02")

    logger.debug_sample("Chunks", [
        f"doc={c.doc_id}, chunk_id={c.chunk_id}, tokens={c.token_count}, text='{c.text[:100]}...'"
        for c in chunks
    ])

    token_counts = [c.token_count for c in chunks]
    logger.debug_distribution("Chunk Token Distribution", {
        f"< {config['chunking']['min_chunk_tokens']}": sum(1 for t in token_counts if t < config["chunking"]["min_chunk_tokens"]),
        f"{config['chunking']['min_chunk_tokens']}-{config['chunking']['max_chunk_tokens']}": sum(1 for t in token_counts if config["chunking"]["min_chunk_tokens"] <= t <= config["chunking"]["max_chunk_tokens"]),
        f"> {config['chunking']['max_chunk_tokens']}": sum(1 for t in token_counts if t > config["chunking"]["max_chunk_tokens"]),
    })

    # Chunks per document
    from collections import Counter
    chunks_per_doc = Counter(c.doc_id for c in chunks)
    logger.debug_distribution("Chunks per Document", dict(chunks_per_doc))

    elapsed = logger.stop_timer("debug_step02")
    logger.debug_checkpoint("step02_chunking", {
        "chunk_count": len(chunks),
        "token_stats": {"min": min(token_counts), "max": max(token_counts), "mean": sum(token_counts) / len(token_counts)},
        "chunks_per_doc": dict(chunks_per_doc),
    }, step_time=elapsed)
```

### Step 3: Add debug block after Step 3 (NER)

```python
if logger.is_debug:
    logger.start_timer("debug_step03")

    # Sample: show entities for first 3 chunks
    for chunk in chunks[:logger._debug_max_samples]:
        entity_strs = [f"{e.name} ({e.entity_type})" for e in chunk.entities]
        logger.debug_sample(f"Entities in chunk '{chunk.doc_id}:{chunk.chunk_id}'", entity_strs)

    # Entity type distribution across all chunks
    from collections import Counter
    type_counts = Counter()
    total_entities = 0
    for chunk in chunks:
        for e in chunk.entities:
            type_counts[e.entity_type] += 1
            total_entities += 1
    logger.debug_distribution("Entity Type Distribution", dict(type_counts))

    elapsed = logger.stop_timer("debug_step03")
    logger.debug_checkpoint("step03_ner", {
        "total_entity_mentions": total_entities,
        "type_distribution": dict(type_counts),
        "avg_entities_per_chunk": total_entities / len(chunks) if chunks else 0,
    }, step_time=elapsed)
```

### Step 4: Add debug block after Step 4 (Dedup)

```python
if logger.is_debug:
    total_mentions = sum(len(c.entities) for c in chunks)
    dedup_ratio = 1 - len(global_entities) / total_mentions if total_mentions > 0 else 0

    logger.debug_sample("Global Entities", [e.name for e in global_entities])
    console.print(Panel(
        f"Total mentions: {total_mentions}\n"
        f"Unique entities: {len(global_entities)}\n"
        f"Dedup ratio: {dedup_ratio:.1%}",
        title="[DEBUG] Entity Deduplication",
        border_style="yellow",
    ))

    logger.debug_checkpoint("step04_dedup", {
        "total_mentions": total_mentions,
        "unique_entities": len(global_entities),
        "dedup_ratio": round(dedup_ratio, 4),
    })
```

### Step 5: Add debug block after Step 5 (Embeddings)

```python
if logger.is_debug:
    entity_norms = np.linalg.norm(entity_embs, axis=1)
    hyperedge_norms = np.linalg.norm(hyperedge_embs, axis=1)

    logger.debug_matrix_detail("Entity Embeddings", entity_embs)
    logger.debug_matrix_detail("Hyperedge Embeddings", hyperedge_embs)

    logger.debug_checkpoint("step05_embeddings", {
        "entity_emb_shape": list(entity_embs.shape),
        "hyperedge_emb_shape": list(hyperedge_embs.shape),
        "entity_norm_stats": {"min": float(entity_norms.min()), "max": float(entity_norms.max()), "mean": float(entity_norms.mean())},
        "hyperedge_norm_stats": {"min": float(hyperedge_norms.min()), "max": float(hyperedge_norms.max()), "mean": float(hyperedge_norms.mean())},
    })
```

### Step 6: Add debug block after Step 6 (Entity Linking)

```python
if logger.is_debug:
    logger.debug_matrix_detail("Synonym Matrix S", S)

    # Extract sample synonym pairs
    S_coo = S.tocoo()
    synonym_pairs = []
    for i, j, v in zip(S_coo.row, S_coo.col, S_coo.data):
        if i < j:  # upper triangle only, skip diagonal
            synonym_pairs.append((global_entities[i].name, global_entities[j].name, float(v)))
    synonym_pairs.sort(key=lambda x: -x[2])

    logger.debug_sample("Synonym Pairs (by score)", [
        f"{a} <-> {b} (score={s:.3f})" for a, b, s in synonym_pairs
    ])

    logger.debug_checkpoint("step06_linking", {
        "S_shape": list(S.shape),
        "S_nnz": int(S.nnz),
        "synonym_pair_count": len(synonym_pairs),
        "top_pairs": [{"a": a, "b": b, "score": round(s, 4)} for a, b, s in synonym_pairs[:10]],
    })
```

### Step 7: Add debug blocks after Steps 7-10

Same pattern: `debug_matrix_detail()` for H, `debug_checkpoint()` for stats. For masking (Step 8), show cluster distribution if available. For bipartite (Step 9), show graph node/edge counts. For save (Step 10), list files and sizes.

### Step 8: Add debug_report() at end of pipeline

At the very end of `index_corpus()`, before the final summary table:

```python
if logger.is_debug:
    logger.debug_report(extra={
        "pipeline": "indexing",
        "documents": len(documents),
        "chunks": len(chunks),
        "entities": len(global_entities),
        "H_shape": list(H.shape),
        "S_nnz": int(S.nnz),
    })
```

### Step 9: Add same debug blocks to index_corpus_incremental()

Apply equivalent debug blocks to the incremental pipeline steps, reusing the same helper functions where possible.

## Todo List

- [ ] Add `_debug_documents()` helper function
- [ ] Add debug block after Step 1 (Load Documents) in `index_corpus()`
- [ ] Add debug block after Step 2 (Chunking)
- [ ] Add debug block after Step 3 (NER)
- [ ] Add debug block after Step 4 (Dedup)
- [ ] Add debug block after Step 5 (Embeddings)
- [ ] Add debug block after Step 6 (Entity Linking)
- [ ] Add debug block after Step 7 (Hypergraph Build)
- [ ] Add debug block after Step 8 (Masking)
- [ ] Add debug block after Step 9 (Bipartite)
- [ ] Add debug block after Step 10 (Save)
- [ ] Add `debug_report()` call at end
- [ ] Add equivalent debug blocks to `index_corpus_incremental()`
- [ ] Smoke test: run pipeline with `--debug` and verify output

## Success Criteria

1. `python scripts/index_corpus.py --input_dir data/raw --output_dir data/indexed --debug` produces Rich debug panels after each step
2. Checkpoint files appear in `data/debug/` (step01_documents.json, step02_chunking.json, etc.)
3. `debug_report.json` produced at end with all timings
4. Without `--debug`, pipeline runs identically to before (no output changes, no performance impact)
5. Incremental pipeline also produces debug output
6. All existing tests pass

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| index_corpus.py exceeds 200 LOC limit | Medium | High | Extract debug blocks into helper functions (`_debug_after_chunking()`, etc.) to keep main function clean |
| Checkpoint I/O slows pipeline | Low | Low | Checkpoint writes are tiny JSON files; numpy saves are fast for embedding-size arrays |
| S.tocoo() conversion for synonym pairs | Low | Low | Only in debug mode; S is typically small |

## Security Considerations

- Checkpoint data is pipeline-internal (entities, stats); no user secrets
- Debug output directory is created with default permissions

## Next Steps

- Phase 04 adds the same pattern for the retrieval pipeline in `query.py`
- Consider adding a `--debug-steps 2,3,6` flag to selectively enable debug for specific steps (future, YAGNI for now)
