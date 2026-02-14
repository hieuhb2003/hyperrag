# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run from `hyp_dlm/` directory.

```bash
# Install
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# Run pipeline
python scripts/index_corpus.py --input_dir data/raw --output_dir data/indexed --config config/default.yaml
python scripts/index_corpus.py --input_dir data/raw --output_dir data/indexed --full-rebuild
python scripts/query.py --index_dir data/indexed --query "question" --config config/default.yaml
python scripts/evaluate.py --index_dir data/indexed --benchmark data/benchmarks/hotpotqa_dev.json --config config/default.yaml --output results/eval.json

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_propagation.py -v              # single file
python -m pytest tests/test_e2e.py::test_masking_kmeans -v  # single test
python -m pytest tests/test_chunker.py -v                  # chunker + anchor tests

# Convenience script
./run.sh setup|demo|index|query|eval|test
```

## Architecture

**HyP-DLM** is a training-free, zero-shot GraphRAG system for multi-hop QA. It runs as a 3-phase pipeline:

### Phase 1: Indexing (offline)
Documents → `create_chunker(config)` → `SemanticChunker` or `AnchorChunker` → chunks (= hyperedges) → `NERExtractor` → entities → `deduplicate_entities()` → `EntityLinker` → synonym matrix S → `HypergraphBuilder` → incidence matrix H (N entities × M hyperedges) → `MaskingStrategy.fit()` → saved to `data/indexed/` + `manifest.json`

**Chunking methods** (selected via `config.chunking.method`):
- `"similarity"` — `SemanticChunker`: embedding cosine similarity breakpoint detection (no LLM cost)
- `"llm_anchor"` — `AnchorChunker`: LLM-based knowledge boundary detection with near-zero output tokens. Inserts `[1]`, `[2]`, ... markers at sentence boundaries, LLM returns which markers start new segments.

**Indexing modes** (selected via `config.indexing.incremental` + `--full-rebuild` flag):
- Full rebuild: processes all documents from scratch, saves manifest
- Incremental: compares SHA-256 content hashes with manifest, only processes new documents. Changed/deleted docs trigger full rebuild recommendation.

**Parallel support**: `AnchorChunker.chunk_documents_parallel()` uses `ThreadPoolExecutor` for I/O-bound LLM calls. `NERExtractor.extract_from_chunks_batch()` uses SpaCy `nlp.pipe()` for batch processing.

### Phase 2: Retrieval (online, 1 LLM call for decomposition)
Query → `FamiliarityRouter.route()` → decides `"direct"|"graph"|"hybrid"` → `QueryDecomposer.decompose()` → `QueryDAG` (topologically ordered sub-questions) → `DAGPropagation.propagate()` (core algorithm: damped PPR on hypergraph per sub-question) → `PassageRanker.rank()` (PPR on bipartite)

### Phase 3: Generation (1 LLM call)
Graph passages + dense passages → `HybridRAGGenerator.generate()` → merges by `hyperedge_id`, weighted fusion → LLM generates answer inside `<answer>` tags

### Core algorithm (propagation.py)
For each sub-question in DAG topological order:
1. Compute guidance vector `g_i = embed(resolved_question)` — **fixed** per sub-question
2. Compute mask via `MaskingStrategy.compute_mask(g_i, top_p)`
3. Compute modulation `D_i = diag(ReLU(cosine(g_i, hyperedge_embs)) * mask)` — **fixed** per sub-question. ReLU zeros out anti-correlated hyperedges (cosine < 0) to prevent negative propagation weights.
4. Build propagation matrix `A_i = H @ D_i @ H^T + w_syn * S` — **fixed**, not per-step
5. Run damped PPR: `s_{t+1} = (1 - α) * s_0 + α * normalize(A_i @ s_t)` until convergence

Key invariant: `A_i` is computed **once per sub-question** and reused across all propagation steps. The "dynamic" in Dynamic Logic Modulation refers to `D_i` changing across sub-questions in the DAG, not across propagation steps.

### Entity linking (entity_linker.py)
Synonym matrix S uses adaptive two-path similarity:
- **Short-entity path** (`min(len(v_i), len(v_j)) <= 5`): Bypasses Jaro-Winkler lexical pre-filter, uses semantic-only threshold (0.85). Catches acronyms (e.g., WHO ↔ World Health Organization) where JW similarity is too low.
- **Standard path** (`min(len(v_i), len(v_j)) > 5`): Jaro-Winkler pre-filter (0.7) + combined score (0.3 × JW + 0.7 × cosine > 0.9).

## Key Conventions

**Import pattern**: Scripts and tests use `sys.path.insert(0, parent_dir)` then absolute imports from `src.*`. All commands must run from `hyp_dlm/` root.

**Shared instances**: `EmbeddingModel` is expensive to load. Create once, pass via `.set_embedder()` or constructor. Same for `LLMClient` — share across decomposer, generator, and `AnchorChunker` (via `.set_llm()`).

**Config-driven**: All hyperparameters in `config/default.yaml`. Components accept `config: dict` subsections at init. No hardcoded values.

**Factory Pattern**: Chunkers created via `create_chunker(config)` factory (`"similarity"` → `SemanticChunker`, `"llm_anchor"` → `AnchorChunker`). Masking strategies created via `create_masking_strategy(config)`.

**Strategy Pattern**: Masking strategies implement `MaskingStrategy` ABC (`fit`, `compute_mask`, `supports_incremental`, `add_hyperedges`). Four implementations: `HDBSCANMasking` (primary), `KMeansMasking` (ablation), `FAISSDirectMasking` (production), `NoMasking` (ablation).

**Chunk = Hyperedge**: These terms are interchangeable throughout. `hyperedge_id` in retrieval results is the chunk index.

**Index artifacts** (in `data/indexed/`): `incidence_matrix_H.npz`, `synonym_matrix_S.npz` (scipy sparse), `entity_embeddings.npy`, `hyperedge_embeddings.npy` (numpy), `metadata.pkl` (chunks, entities, indices), `masking_strategy.pkl` (fitted strategy object — must be importable to unpickle), `bipartite_graph.pkl`, `manifest.json` (document hashes for incremental updates).

**Incremental updates**: `EntityLinker.expand_synonym_matrix()` grows S incrementally. `HypergraphBuilder.expand()` grows H incrementally. `BipartiteStorage.add_document()` adds new chunks/entities. All masking strategies support `add_hyperedges()`.

**Logging**: Custom `HypDLMLogger` extends `logging.Logger`. Use `logger.step()` for pipeline progress panels, `logger.matrix()` for sparse matrix stats, `logger.convergence()` for propagation tracking. Level set via `config.logging.level`.

**Sparse matrices**: All graph operations use `scipy.sparse` CSR format. Build with `lil_matrix`, convert to `csr_matrix` for operations. Utility functions in `src/utils/sparse_ops.py`.
