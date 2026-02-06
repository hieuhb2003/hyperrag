# HyP-DLM Implementation Plan

## Overview

**Mục tiêu:** Nâng cấp hệ thống LinearRAG hiện tại thành **HyP-DLM (Hypergraph Propagation with Dynamic Logic Modulation)** để cải thiện đáng kể khả năng suy luận đa bước (5+ hops).

**Điểm thay đổi chính:**
| Hiện tại (LinearRAG) | Mục tiêu (HyP-DLM) |
|---|---|
| Graph đơn giản (Entity-Passage) | **Hypergraph Bipartite** (Entity-Proposition) |
| PageRank tĩnh (PPR) | **Dynamic Logic Modulation** (Attention theo câu hỏi) |
| BFS Iteration | **Sparse Matrix Multiplication + Semantic Masking** |

---

## Project Type

**BACKEND** - Thư viện Python cho GraphRAG.

---

## Success Criteria

- [ ] Multi-hop 5+ hops không bị drift điểm số.
- [ ] Query latency < 500ms cho 100K propositions.
- [ ] Indexing cost thấp hơn GraphRAG (Microsoft) nhờ không dùng LLM để extract relation.
- [ ] API endpoint `/query` trả về câu trả lời với đầy đủ citations.

---

## Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| LLM | `GPT-4o-mini` / `Qwen-2.5-7B` | Zero-shot proposition extraction |
| NER | `SpaCy` (`en_core_web_trf`) | Đã có sẵn trong project |
| Embedding | `all-mpnet-base-v2` / `bge-m3` | Đã có sẵn |
| Vector DB | `LanceDB` (Embedded) | Serverless, dễ tích hợp |
| Math | `SciPy` (Sparse), `NumPy`, `PyTorch` | Đã có trong project |
| Server | `FastAPI` | Production-ready |

---

## File Structure (Proposed)

```
hyperrag/
├── src/
│   ├── HyPDLM.py              # [NEW] Core HyP-DLM class
│   ├── propositions.py        # [NEW] Atomic Proposition Extraction
│   ├── hypergraph.py          # [NEW] Hypergraph Matrix Builder (H matrix)
│   ├── semantic_affinity.py   # [NEW] A_sem Matrix Builder (Entity Soft-Linking)
│   ├── dynamic_modulation.py  # [NEW] Attention-weighted Propagation
│   ├── semantic_masking.py    # [NEW] Cluster-based Filtering
│   ├── LinearRAG.py           # [MODIFY] Refactor base methods
│   ├── config.py              # [MODIFY] Add HyPDLMConfig
│   ├── ner.py                 # [KEEP] Reuse SpaCy NER
│   ├── embedding_store.py     # [KEEP]
│   └── utils.py               # [MODIFY] Add helper functions
├── run_hypdlm.py              # [NEW] Entry point for HyP-DLM
├── server.py                  # [NEW] FastAPI Server
└── tests/
    ├── test_propositions.py   # [NEW]
    ├── test_hypergraph.py     # [NEW]
    ├── test_affinity.py       # [NEW] Tests for A_sem
    └── test_modulation.py     # [NEW]
```

---

## Task Breakdown

### Phase 1: Foundation (Offline Indexing)

#### Task 1.1: Atomic Proposition Extraction
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns`, `clean-code` |
| **Priority** | P0 |
| **Dependencies** | None |
| **INPUT** | Clean Chunks (text) |
| **OUTPUT** | List of `Proposition` objects with embedding |
| **VERIFY** | Unit test: 1 complex sentence → 2+ atomic propositions |

#### Task 1.2: Semantic Structured Compression
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns` |
| **Priority** | P0 |
| **Dependencies** | None |
| **INPUT** | Raw Chunks |
| **OUTPUT** | Filtered Clean Chunks (entropy > threshold) |
| **VERIFY** | Unit test: Low-entropy chunks filtered out |

#### Task 1.3: Hypergraph Matrix Builder
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns` |
| **Priority** | P1 |
| **Dependencies** | Task 1.1, Task 1.2 |
| **INPUT** | Entities, Propositions |
| **OUTPUT** | Incidence Matrix `H` (sparse CSR), Affinity Matrix `S` |
| **VERIFY** | `H.shape == (num_entities, num_propositions)` |

#### Task 1.4: Cluster Centroids for Semantic Masking
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns` |
| **Priority** | P1 |
| **Dependencies** | Task 1.1 |
| **INPUT** | Proposition embeddings |
| **OUTPUT** | K cluster centroids (numpy array) |
| **VERIFY** | K-Means converges, centroids stored in `.npz` |

#### Task 1.5: Semantic Affinity Matrix (`A_sem`) - Entity Soft-Linking
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns`, `performance-profiling` |
| **Priority** | P1 |
| **Dependencies** | Task 1.2 (NER entities) |
| **INPUT** | List of Entity names + embeddings |
| **OUTPUT** | Sparse Matrix `A_sem` (n_entities × n_entities) |
| **VERIFY** | `A_sem["L.Messi", "Leo Messi"] > 0.8` |

**Chi tiết xây dựng `A_sem`:**

1. **Blocking (Faiss/LSH):** Tìm nhanh candidate pairs có khả năng giống nhau → Giảm từ O(n²) xuống O(n×k).

2. **Scoring (Hybrid):**
   - **Lexical:** Jaro-Winkler distance (khớp mặt chữ)
   - **Embedding:** Cosine similarity (khớp ngữ nghĩa)
   - **Công thức:** `S(i,j) = α × Lexical + (1-α) × Embedding` với `α = 0.5`

3. **Sparsification:** Chỉ giữ các cặp có `S(i,j) > τ` (mặc định `τ = 0.7`)

**Vai trò trong Retrieval:**
- Công thức cập nhật: `x_{t+1} = H^T × D_t × H × x_t + β × A_sem × x_t`
- `β = 0.3-0.5`: Hệ số "rò rỉ" sang entity tương đương
- Giải quyết vấn đề "L.Messi" không nối với "Inter Miami" nhưng "Leo Messi" thì có

---

### Phase 2: Online Retrieval

#### Task 2.1: Router & Familiarity Check
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns` |
| **Priority** | P2 |
| **Dependencies** | Phase 1 complete |
| **INPUT** | User query |
| **OUTPUT** | Decision: Bypass Graph OR Graph Reasoning |
| **VERIFY** | High-confidence queries bypass graph |

#### Task 2.2: Logic Decomposition (Guidance Vectors)
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `api-patterns` |
| **Priority** | P2 |
| **Dependencies** | Task 2.1 |
| **INPUT** | Complex query |
| **OUTPUT** | List of Guidance Vectors `g_1, g_2, ..., g_T` |
| **VERIFY** | "Wife of Microsoft founder's birth year" → 3 vectors |

#### Task 2.3: Dynamic Modulation Propagation
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `python-patterns`, `performance-profiling` |
| **Priority** | P2 |
| **Dependencies** | Task 1.3, Task 2.2 |
| **INPUT** | Seed entities, Guidance vector, Matrices |
| **OUTPUT** | Updated entity scores after T hops |
| **VERIFY** | Correct entities activated after 3+ hops |

#### Task 2.4: Semantic Masking Integration
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `performance-profiling` |
| **Priority** | P2 |
| **Dependencies** | Task 1.4, Task 2.3 |
| **INPUT** | Guidance vector, Cluster centroids |
| **OUTPUT** | Binary mask reducing hyperedge search space |
| **VERIFY** | Search space reduced by 90%+ |

---

### Phase 3: Integration & API

#### Task 3.1: HyPDLM Core Class
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `clean-code` |
| **Priority** | P3 |
| **Dependencies** | Phase 1 + 2 |
| **INPUT** | All modules from Phase 1, 2 |
| **OUTPUT** | `HyPDLM` class with `index()` and `retrieve()` methods |
| **VERIFY** | Integration test passes |

#### Task 3.2: FastAPI Server
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `api-patterns` |
| **Priority** | P3 |
| **Dependencies** | Task 3.1 |
| **INPUT** | HyPDLM class |
| **OUTPUT** | `POST /query` endpoint |
| **VERIFY** | `curl -X POST /query` returns JSON response |

#### Task 3.3: Config Update
| Field | Value |
|---|---|
| **Agent** | `backend-specialist` |
| **Skill** | `clean-code` |
| **Priority** | P1 |
| **Dependencies** | None |
| **INPUT** | Current `LinearRAGConfig` |
| **OUTPUT** | `HyPDLMConfig` dataclass with new parameters |
| **VERIFY** | All HyP-DLM params configurable |

---

## Phase X: Verification

- [ ] **Lint**: `ruff check src/`
- [ ] **Type Check**: `pyright src/`
- [ ] **Unit Tests**: `pytest tests/ -v`
- [ ] **Integration Test**: Run full pipeline on sample dataset
- [ ] **Benchmark**: Compare latency vs LinearRAG on 2wikimultihop
- [ ] **Security**: `python .agent/skills/vulnerability-scanner/scripts/security_scan.py .`

---

## Agent Assignments Summary

| Phase | Tasks | Agent |
|---|---|---|
| 1 | 1.1, 1.2, 1.3, 1.4, 1.5 | `backend-specialist` |
| 2 | 2.1, 2.2, 2.3, 2.4 | `backend-specialist` |
| 3 | 3.1, 3.2, 3.3 | `backend-specialist` |
| X | Verification | `security-auditor`, `backend-specialist` |

---

## Next Steps

1. Review this plan
2. Run `/create` to start implementation
3. Or modify plan manually
