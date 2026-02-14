# Phase 04: Retrieval Pipeline Debug Output

## Context Links

- [plan.md](./plan.md)
- [phase-02-debug-config-and-logger.md](./phase-02-debug-config-and-logger.md) -- prerequisite (logger methods)
- [query.py](../../hyp_dlm/scripts/query.py) -- main file to modify
- [propagation.py](../../hyp_dlm/src/retrieval/propagation.py) -- sub-question propagation details
- [router.py](../../hyp_dlm/src/retrieval/router.py) -- route decision details
- [query_decomposer.py](../../hyp_dlm/src/retrieval/query_decomposer.py) -- DAG structure

## Overview

- **Priority**: P2
- **Status**: complete
- **Effort**: 0.75h
- **Description**: Add debug output after each step of `run_query()` in `query.py`. When `--debug` is active, the pipeline prints route decisions, DAG structure, per-sub-question propagation details, passage rankings, and generation context.

## Key Insights

- `run_query()` has 7 clearly numbered steps. Steps 4-6 only execute for `graph`/`hybrid` routes.
- Propagation (Step 5) is the richest debug target -- it iterates over sub-questions in DAG order, computing guidance vectors, masks, modulation matrices, and running PPR convergence. The `prop_result` dict contains per-sub-question scores.
- Route decision (Step 2) returns `route_result` with scores and entropy -- already computed, just needs display.
- Decomposition (Step 4) returns a `QueryDAG` with `nodes` (list of `SubQuestion`). Display topology.
- Generation (Step 7) has internal passage merging -- show the assembled context before LLM call.

## Requirements

### Functional

Debug output after each retrieval step:

| Step | What to Show | Checkpoint Data |
|------|-------------|-----------------|
| 1. Encode Query | Query embedding norm, dimension | `{query, emb_shape, emb_norm}` |
| 2. Route | Route decision, mean_score, entropy, top-K similarity scores | `{route, mean_score, entropy, top_k_scores}` |
| 3. Dense Retrieval | Top chunks with scores, chunk texts (truncated) | `{chunk_count, top_scores, samples}` |
| 4. Decomposition | DAG nodes, dependencies, topological order | `{node_count, dag_structure, topo_order}` |
| 5. Propagation | Per sub-question: guidance norm, mask coverage, D_i stats, convergence curve, top-K entities | `{per_question: [{question, mask_coverage, convergence_steps, top_entities}]}` |
| 6. Passage Ranking | Ranked passages with PPR scores, passage texts (truncated) | `{passage_count, top_passages}` |
| 7. Generation | Merged passage count, context token estimate, raw LLM prompt (truncated), answer | `{merged_count, context_tokens, answer}` |

### Non-Functional
- All debug calls guarded by `if logger.is_debug:`
- Per-sub-question propagation debug should not be excessively verbose -- cap at max_samples sub-questions for detailed output
- Convergence curves stored as lists of delta values (small data)

## Architecture

No new files. All changes inline in `run_query()` function body.

```
run_query()
  |
  Step 1: encode query
  |-- if debug: show embedding stats
  |
  Step 2: route
  |-- if debug: show route decision panel
  |
  Step 3: dense retrieval
  |-- if debug: show top chunk passages
  |
  [if graph/hybrid]
  |
  Step 4: decompose
  |-- if debug: show DAG structure
  |
  Step 5: propagate
  |-- if debug: per-sub-question detail panel
  |
  Step 6: rank passages
  |-- if debug: show ranked passages
  |
  Step 7: generate
  |-- if debug: show merged context, answer
  |
  End: logger.debug_report()
```

## Related Code Files

### Files to Modify
- `hyp_dlm/scripts/query.py` -- Add debug blocks after each step in `run_query()`

### Files NOT Modified
- `logger.py` -- Methods already added in Phase 02
- `propagation.py` -- Read `prop_result` dict, don't modify propagation internals
- `router.py` -- Read `route_result` dict, don't modify router

## Implementation Steps

### Step 1: Add debug block after query encoding (Step 1)

After `query_emb = embedder.encode_single(query)`:

```python
if logger.is_debug:
    emb_norm = float(np.linalg.norm(query_emb))
    console.print(Panel(
        f"Query: '{query}'\n"
        f"Embedding shape: {query_emb.shape}\n"
        f"Embedding norm (L2): {emb_norm:.4f}",
        title="[DEBUG] Step 1: Query Encoding",
        border_style="yellow",
    ))
    logger.debug_checkpoint("query_step01_encode", {
        "query": query,
        "emb_shape": list(query_emb.shape),
        "emb_norm": round(emb_norm, 4),
    })
```

### Step 2: Add debug block after routing (Step 2)

After `route = route_result["route"]`:

```python
if logger.is_debug:
    logger.start_timer("debug_route")

    # route_result typically has: route, mean_score, entropy, scores
    route_details = {k: v for k, v in route_result.items() if k != "scores"}
    lines = [f"Route: {route}"]
    for k, v in route_details.items():
        if k != "route":
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Show top-K similarity scores if available
    if "scores" in route_result:
        scores = route_result["scores"]
        if hasattr(scores, '__len__') and len(scores) > 0:
            top_scores = sorted(scores, reverse=True)[:5]
            lines.append(f"  Top-5 similarity scores: {[round(s, 4) for s in top_scores]}")

    console.print(Panel("\n".join(lines), title="[DEBUG] Step 2: Routing Decision", border_style="yellow"))

    elapsed = logger.stop_timer("debug_route")
    logger.debug_checkpoint("query_step02_route", {
        "route": route,
        "details": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in route_result.items() if k != "scores"},
    }, step_time=elapsed)
```

### Step 3: Add debug block after dense retrieval (Step 3)

After `chunk_passages = chunk_retriever.retrieve(...)`:

```python
if logger.is_debug:
    logger.debug_sample("Dense Retrieved Chunks", [
        f"idx={p.get('hyperedge_id', '?')}, score={p.get('score', 0):.4f}, "
        f"text='{chunks[p['hyperedge_id']].text[:80]}...'" if p.get('hyperedge_id') is not None and p['hyperedge_id'] < len(chunks) else f"score={p.get('score', 0):.4f}"
        for p in chunk_passages
    ])

    logger.debug_checkpoint("query_step03_dense", {
        "chunk_count": len(chunk_passages),
        "top_scores": [round(p.get("score", 0), 4) for p in chunk_passages[:5]],
    })
```

### Step 4: Add debug block after decomposition (Step 4)

After `dag = decomposer.decompose(query)`:

```python
if logger.is_debug:
    dag_lines = []
    for node in dag.nodes:
        deps = ", ".join(node.depends_on) if node.depends_on else "none"
        dag_lines.append(f"  [{node.id}] {node.question}  (depends: {deps})")

    topo_order = [n.id for n in dag.topological_order()]
    dag_lines.append(f"\n  Topological order: {' -> '.join(topo_order)}")

    console.print(Panel(
        "\n".join(dag_lines),
        title=f"[DEBUG] Step 4: Query DAG ({len(dag.nodes)} sub-questions)",
        border_style="yellow",
    ))

    logger.debug_checkpoint("query_step04_decompose", {
        "node_count": len(dag.nodes),
        "dag_structure": [
            {"id": n.id, "question": n.question, "depends_on": n.depends_on}
            for n in dag.nodes
        ],
        "topological_order": topo_order,
    })
```

### Step 5: Add debug block after propagation (Step 5)

After `prop_result = propagator.propagate(...)`:

```python
if logger.is_debug:
    logger.start_timer("debug_propagation")

    # prop_result contains entity_scores (dict), hyperedge_scores, per-question info
    entity_scores_dict = prop_result.get("entity_scores", {})
    top_entities_by_score = sorted(entity_scores_dict.items(), key=lambda x: -x[1])[:10]

    entity_lines = []
    for idx, score in top_entities_by_score:
        name = global_entities[idx].name if idx < len(global_entities) else f"entity_{idx}"
        entity_lines.append(f"  [{idx}] {name}: {score:.4f}")

    console.print(Panel(
        f"Total activated entities: {len(entity_scores_dict)}\n"
        f"Top-10 entities by score:\n" + "\n".join(entity_lines),
        title="[DEBUG] Step 5: Propagation Results",
        border_style="magenta",
    ))

    # Per sub-question details if available
    per_q = prop_result.get("per_question", [])
    for q_info in per_q[:logger._debug_max_samples]:
        q_id = q_info.get("id", "?")
        q_text = q_info.get("question", "?")
        hops = q_info.get("hops_used", "?")
        mask_cov = q_info.get("mask_coverage", "?")
        console.print(Panel(
            f"Question: {q_text}\n"
            f"Hops used: {hops}\n"
            f"Mask coverage: {mask_cov}",
            title=f"[DEBUG] Sub-question {q_id}",
            border_style="magenta",
        ))

    elapsed = logger.stop_timer("debug_propagation")
    logger.debug_checkpoint("query_step05_propagation", {
        "activated_entities": len(entity_scores_dict),
        "top_entities": [
            {"idx": int(idx), "name": global_entities[idx].name if idx < len(global_entities) else f"entity_{idx}", "score": round(score, 4)}
            for idx, score in top_entities_by_score
        ],
        "per_question_summary": [
            {"id": q.get("id"), "hops": q.get("hops_used"), "mask_coverage": q.get("mask_coverage")}
            for q in per_q
        ],
    }, step_time=elapsed)
```

### Step 6: Add debug block after passage ranking (Step 6)

After `graph_passages = ranker.rank(...)`:

```python
if logger.is_debug:
    passage_lines = []
    for i, p in enumerate(graph_passages[:5]):
        h_id = p.get("hyperedge_id", "?")
        score = p.get("score", 0)
        text_preview = chunks[h_id].text[:100] if isinstance(h_id, int) and h_id < len(chunks) else "N/A"
        passage_lines.append(f"  [{i+1}] hyperedge={h_id}, score={score:.4f}")
        passage_lines.append(f"       '{text_preview}...'")

    console.print(Panel(
        f"Total ranked passages: {len(graph_passages)}\n\n" + "\n".join(passage_lines),
        title="[DEBUG] Step 6: PPR Passage Ranking",
        border_style="yellow",
    ))

    logger.debug_checkpoint("query_step06_ranking", {
        "passage_count": len(graph_passages),
        "top_passages": [
            {"hyperedge_id": p.get("hyperedge_id"), "score": round(p.get("score", 0), 4)}
            for p in graph_passages[:10]
        ],
    })
```

### Step 7: Add debug block after generation (Step 7)

After `result = generator.generate(...)`:

```python
if logger.is_debug:
    retrieved = result.get("retrieved_passages", [])
    raw_response = result.get("raw_response", "")

    # Estimate context tokens (rough: split by whitespace)
    context_tokens = sum(len(p.get("text", "").split()) for p in retrieved) if retrieved else 0

    console.print(Panel(
        f"Merged passages used: {len(retrieved)}\n"
        f"Estimated context tokens: {context_tokens}\n"
        f"Answer: {result.get('answer', 'N/A')}\n"
        f"\nRaw LLM response (first 300 chars):\n{raw_response[:300]}...",
        title="[DEBUG] Step 7: Generation",
        border_style="green",
    ))

    logger.debug_checkpoint("query_step07_generation", {
        "merged_passage_count": len(retrieved),
        "context_tokens_estimate": context_tokens,
        "answer": result.get("answer", ""),
        "route": route,
    })
```

### Step 8: Add debug_report() at end

Before the final output in `run_query()`:

```python
if logger.is_debug:
    logger.debug_report(extra={
        "pipeline": "retrieval",
        "query": query,
        "route": route,
        "answer": result.get("answer", ""),
        "graph_passages": len(graph_passages),
        "chunk_passages": len(chunk_passages),
    })
```

### Step 9: Import console at top of query.py

Add `from rich.console import Console` and `console = Console()` near the top, or import from logger module:

```python
from src.utils.logger import get_logger, console  # reuse existing console instance
```

Note: `console` is already a module-level variable in `logger.py`. Just need to export it.

## Todo List

- [ ] Import `console` from `logger.py` in `query.py`
- [ ] Add debug block after Step 1 (Query Encoding)
- [ ] Add debug block after Step 2 (Routing)
- [ ] Add debug block after Step 3 (Dense Retrieval)
- [ ] Add debug block after Step 4 (Decomposition)
- [ ] Add debug block after Step 5 (Propagation)
- [ ] Add debug block after Step 6 (Passage Ranking)
- [ ] Add debug block after Step 7 (Generation)
- [ ] Add `debug_report()` call at end
- [ ] Smoke test: run query with `--debug` and verify output

## Success Criteria

1. `python scripts/query.py --index_dir data/indexed --query "..." --debug` produces Rich debug panels after each step
2. Checkpoint files appear in `data/debug/` (query_step01_encode.json through query_step07_generation.json)
3. `debug_report.json` produced with pipeline=retrieval and all timings
4. For `direct` route, steps 4-6 debug blocks are skipped (no crash, no empty panels)
5. Without `--debug`, query pipeline runs identically to before
6. All existing tests pass

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `prop_result` dict structure varies | Medium | Medium | Use `.get()` with defaults for all prop_result keys; test with both direct and graph routes |
| `route_result` dict keys vary by router version | Low | Low | Use `.get()` pattern; only display keys that exist |
| `query.py` exceeds 200 LOC | Medium | High | Extract debug blocks into helper functions (`_debug_after_routing()`, etc.) |
| `console` import from logger.py | Low | Low | `console` is module-level in logger.py; just import it |

## Security Considerations

- Debug checkpoint saves the query text and answer to disk; no secrets involved
- Raw LLM response is truncated to 300 chars in display to avoid flooding terminal
- Debug output directory permissions follow OS defaults

## Next Steps

- After all 4 phases: write unit tests for JSON loading (Phase 01) and smoke test for debug mode
- Future: Add `--debug-json` flag to output debug panels as JSON instead of Rich (for CI/CD parsing)
- Future: Add `--profile` flag for detailed per-function CPU profiling (separate from debug)
