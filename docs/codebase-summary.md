# HyP-DLM Codebase Summary

## Overview

HyP-DLM (Hypergraph Propagation with Dynamic Logic Modulation) is a training-free, hypergraph-based RAG framework for multi-hop question answering. Core: 29 Python modules organized across indexing, retrieval, generation, and evaluation pipelines.

## Architecture

29-file architecture organized in 5 main modules:

### Core Infrastructure
- `config.py` — Centralized hyperparameter config (dataclass with YAML support)
- `main.py` — CLI entry point with subcommands: index, retrieve, generate, evaluate
- `logging-setup.py` — Thread-safe logging framework with console + file handlers
- `token-tracker.py` — LLM token usage recording (per phase, per doc/query)
- `time-tracker.py` — Performance timing (context manager API)
- `data-loader.py` — JSON document/query loader with validation

### Indexing Module (9 files)
Transforms raw documents into a hypergraph index. Sequential stages executed in parallel where possible.

- `coreference-resolver.py` — FastCoref (LingMessCoref) anaphora resolution
- `semantic-chunker.py` — Embedding-based semantic chunking via sentence similarity
- `entity-extractor-base.py` — Abstract NER interface
- `entity-extractor-gliner.py` — GLiNER2 zero-shot NER backend
- `entity-extractor-spacy.py` — spaCy traditional NER backend
- `incidence-matrix-builder.py` — Hypergraph H matrix construction + normalization
- `synonym-matrix-builder.py` — 8-stage entity-entity synonym pipeline (TF-IDF + dense + RRF + context validation)
- `semantic-masker.py` — HDBSCAN clustering of chunks for coarse filtering
- `indexing-pipeline.py` — Orchestrator: parallelizes per-doc stages, serializes global stages

### Retrieval Module (5 files)
Query-to-evidence pipeline with DAG decomposition and dynamic modulation.

- `query-decomposer.py` — LLM-based multi-hop query decomposition → DAG structure
- `state-initializer.py` — NER + embedding similarity for entity seed vectors
- `dag-propagator.py` — Damped PPR with dynamic modulation matrix per sub-question
- `multi-granular-scorer.py` — Composite chunk + entity scoring (w_h=1.0, w_e=0.3)
- `retrieval-pipeline.py` — End-to-end orchestrator

### Generation Module (2 files)
Answer synthesis with reasoning traces.

- `answer-generator.py` — Trace-augmented LLM prompting for final answers
- `generation-pipeline.py` — Batch generation + output formatting

### Evaluation Module (2 files)
Metrics aggregation and ablation support.

- `metrics-aggregator.py` — Recall@K, EM, F1 computation + cross-run comparison
- `output-saver.py` — JSON serialization of retrieval + generation results

## File Count

- **Total:** 29 modules
- **Indexing:** 9 files
- **Retrieval:** 5 files
- **Generation:** 2 files
- **Evaluation:** 2 files
- **Infrastructure:** 6 files + 5 __init__.py

## Storage Layout

```
storage/
├── {embedding_model}/
    ├── chunks.json                 # Chunk metadata
    ├── chunk_embeddings.npy        # Chunk embeddings [M x 768]
    ├── entities.json               # Entity registry
    ├── h_norm.npz                  # Normalized incidence matrix [N x M]
    ├── synonym_conf.npz            # Synonym confidence matrix [N x N]
    ├── entity_map.json             # Entity ID → row index
    ├── chunk_map.json              # Chunk ID → col index
    ├── cluster_centroids.npy       # Cluster centroids [K x 768]
    ├── chunk_cluster_map.json      # Chunk → cluster assignments
    └── cluster_chunk_map.json      # Cluster → chunk memberships

output/
├── {dataset_name}/
    ├── retrieval-results.json      # Top-10 chunks per query
    └── generation-results.json     # Generated answers + scores

metrics/
├── {dataset_name}/
    ├── token-usage.json            # LLM token counts per phase
    └── timing.json                 # Pipeline timing per phase
```

## Key Design Patterns

1. **Pluggable NER** — Abstract interface supports GLiNER2 or spaCy
2. **Sparse Matrices** — scipy.sparse for memory efficiency (H, H_norm, S_conf)
3. **Embedding Reuse** — All-mpnet-base-v2 shared across chunking, NER matching, synonym building
4. **Thread-Safe Tracking** — Token/time metrics safe for concurrent indexing
5. **DAG Processing** — Topological sort of sub-questions with parent seeding
6. **Dynamic Modulation** — Per-sub-question chunk masking via cosine relevance

## Configuration

Central `Config` dataclass covers:
- Device (CUDA/MPS/CPU)
- Embedding model (all-mpnet-base-v2)
- NER backend (gliner/spacy)
- Chunking (min/max tokens, breakpoint percentile)
- Hyperparameters from paper (PPR α, damping, pruning δ, etc.)
- LLM settings (model, temperature, max_tokens)
- Worker count, batch sizes, logging paths
