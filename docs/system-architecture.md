# HyP-DLM System Architecture

## Three-Stage Pipeline

HyP-DLM implements a training-free multi-hop QA system via three main stages:

### Stage 1: Offline Indexing (36.5h total effort)
Transforms raw documents into a indexed hypergraph with semantic masking.

**Pipeline Flow:**
```
Raw Documents
  ↓
[Phase 04] Coreference Resolution (FastCoref)
  ↓
[Phase 05] Semantic Chunking (embeddings + cosine breakpoints)
  ↓
[Phase 06] Entity Extraction (NER: GLiNER2 or spaCy)
  ↓
[Phase 07] Incidence Matrix (H[N×M], normalized)
  ↓
[Phase 08] Synonym Matrix (8-stage TF-IDF + dense + RRF + context)
  ↓
[Phase 09] Semantic Masking (HDBSCAN clustering)
  ↓
[Phase 12] Multithreaded Orchestration (parallelizes per-doc stages)
  ↓
Storage artifacts (matrices, embeddings, clusters)
```

**Key Components:**
- **Coreference:** Resolves pronouns → canonical entity forms
- **Chunking:** Splits docs at semantic breakpoints (cosine similarity between consecutive sentence embeddings)
- **NER:** Extracts entities with types; global deduplication
- **Matrices:**
  - H_norm: normalized incidence matrix for propagation
  - S_conf: entity synonym confidence (supports entity merging)
- **Masking:** Clusters chunks; coarse filter during retrieval

### Stage 2: Online Retrieval (6h total effort)
Decomposes queries into DAG of sub-questions, propagates through hypergraph with dynamic modulation.

**Pipeline Flow:**
```
Query
  ↓
[Phase 10.1] Query Decomposition (LLM → DAG of sub-questions)
  ↓
Topological Sort (process in dependency order)
  ↓
For each sub-question:
  [Phase 10.2] State Initialization (NER + embedding similarity)
  [Phase 10.3] Semantic Masking (top-P active clusters)
  [Phase 10.4] Dynamic Modulation (D_i = cos(sub_q, chunk) * mask)
  [Phase 10.5] Damped PPR (A_i = H_norm @ D_i @ H_norm.T + S_conf)
  → Converges in T≤6 iterations, dynamic pruning at δ=0.05
  [Phase 10.6] Intermediate Answer (if non-leaf node)
  ↓
[Phase 10.7] Multi-Granular Scoring (composite: w_h * s_hyper + w_e * s_entity)
  ↓
Top-10 Ranked Chunks
```

**Key Algorithms:**
- **Query Decomposition:** One LLM call produces DAG structure with dependencies
- **State Init:** Query NER → entity embedding similarity (τ_init=0.5) → fallback to query embedding
- **Propagation:** Damped PPR with dynamic per-sub-question chunk relevance
- **Parent Seeding:** Non-root sub-questions inherit parent state (α_parent=0.5)
- **Scoring:** Combines hyperedge score (chunk relevance) + entity score (top entity within chunk)

### Stage 3: Online Generation (3h total effort)
Synthesizes final answer using evidence and reasoning trace.

**Pipeline Flow:**
```
Retrieval Results + DAG Traversal
  ↓
[Phase 11.1] Trace Building (reasoning path from DAG)
[Phase 11.2] Evidence Assembly (top-10 chunks with scores)
[Phase 11.3] Prompt Construction (trace + evidence + query)
  ↓
[Phase 11.4] LLM Answer Generation (gpt-4o-mini or configurable)
  ↓
[Phase 11.5] Output Serialization (JSON with trace, scores, tokens)
```

**Key Features:**
- Trace-augmented prompting reveals sub-question decomposition
- Evidence ranked by HyP-DLM composite score
- Token tracking for all LLM calls

### Stage 4: Evaluation & Ablation (3h total effort)
Metrics aggregation and cross-run comparison.

**Metrics:**
- **Retrieval:** Recall@K (K=10) vs ground truth evidence
- **Generation:** Exact Match (EM) and token-level F1 (SQuAD-style)
- **Efficiency:** Token usage per phase, timing per component
- **Ablation:** Compare runs across embedding models, NER backends, hyperparams

## Data Structures

### Hypergraph Representation

**Incidence Matrix H [N × M]:**
- N = number of entities (nodes)
- M = number of chunks (hyperedges)
- H[i,j] = 1 if entity i appears in chunk j
- Normalized: H_norm = Dv^(-1/2) @ H @ De^(-1/2)

**Synonym Matrix S_conf [N × N]:**
- Symmetric confidence scores for entity pairs
- Built via 8-stage pipeline (sparse + dense hybrid search)
- Used during propagation: A_i = H_norm @ D_i @ H_norm.T + S_conf

**Semantic Mask:**
- K cluster centroids [K × 768]
- Chunk-to-cluster multi-label assignments
- Runtime: select top-P clusters by query similarity → active chunks

### Query DAG

**SubQuestion:**
- id, text, depends_on (parent IDs)

**QueryDAG:**
- root_query, list of sub_questions, topological_order

**State Vectors:**
- Per sub-question: entity relevance [N] from PPR
- Per query: aggregate via max across all sub-questions

## Parallelization Strategy

**Embarrassingly Parallel (Phase 12):**
- Per-document: coref, chunking, sentence splitting
- Per-chunk: NER extraction
- Uses ThreadPoolExecutor with configurable max_workers

**Global Operations (single-threaded):**
- Incidence matrix construction (sparse ops)
- Synonym matrix building (8-stage pipeline)
- Semantic masking (HDBSCAN)

**GPU-friendly:**
- Batch embeddings (sentence-transformers)
- Batch NER (GLiNER2 or spaCy)
- No per-thread GPU access; batched inference

## Hyperparameters (from Paper, Appendix B)

| Component | Param | Value | Purpose |
|-----------|-------|-------|---------|
| General | device | cuda/mps/cpu | Execution device |
| Chunking | breakpoint_percentile | 90 | Cosine similarity breakpoint threshold |
| NER | backend | gliner/spacy | Entity extraction |
| | min_entity_len | 3 | Min entity text length |
| Synonym | rrf_k | 60 | RRF denominator |
| | char_ngram_range | (2,4) | TF-IDF features |
| Masking | hdbscan_min_cluster_size | 15 | HDBSCAN density |
| | multi_label_threshold | 0.5 (τ_multi) | Multi-label cosine |
| Propagation | ppr_alpha | 0.85 | Damping factor |
| | ppr_epsilon_conv | 0.01 | Convergence threshold |
| | ppr_max_iterations | 6 (T) | Max PPR iterations |
| | ppr_tau_init | 0.5 | State init threshold |
| | ppr_delta_pruning | 0.05 | Dynamic pruning |
| | ppr_alpha_parent | 0.5 | Parent seeding |
| Scoring | score_weight_hyper | 1.0 (w_h) | Chunk weight |
| | score_weight_entity | 0.3 (w_e) | Entity weight |
| | score_top_n_final | 10 | Final top-N results |
| LLM | model | gpt-4o-mini | Generation model |
| | temperature | 0.0 | Deterministic |
| | max_tokens | 1024 | Max output length |

## Extensibility Points

1. **NER Backends** — Add factory in entity-extractor-base.py
2. **Embedding Models** — Config entry, load via sentence-transformers
3. **LLM Providers** — litellm wrapper handles OpenAI, Anthropic, etc.
4. **Metrics** — Extend metrics-aggregator.py with custom evaluation functions
5. **Propagation** — Replace PPR with alternative diffusion algorithms
