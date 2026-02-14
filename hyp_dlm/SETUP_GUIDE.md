# HyP-DLM Setup & Usage Guide

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Configuration](#4-configuration)
5. [Prepare Your Data](#5-prepare-your-data)
6. [Run the Pipeline](#6-run-the-pipeline)
7. [Understanding the Logs](#7-understanding-the-logs)
8. [Run Tests](#8-run-tests)
9. [Incremental Updates](#9-incremental-updates)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | >= 3.10 | `python3 --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| OpenAI API Key | - | Required for query decomposition & generation |

**Hardware**: 8 GB RAM minimum. GPU optional (only used by FAISS if `use_gpu: true`).

> **Note**: If using `llm_anchor` chunking method, an OpenAI API key is also required during indexing.

---

## 2. Installation

### Step 1 — Clone & enter the project

```bash
cd /path/to/Hypdlm_rag/hyp_dlm
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
# Core dependencies
pip install -e ".[dev]"

# Download SpaCy English NER model (required)
python -m spacy download en_core_web_sm
```

**Optional extras:**

```bash
# If you want GLiNER (zero-shot NER) instead of SpaCy
pip install -e ".[gliner]"

# If you have an NVIDIA GPU for FAISS
pip install -e ".[gpu]"
```

### Step 4 — Set your OpenAI API key

The LLM is used during **retrieval** (query decomposition) and **generation**.
Indexing is fully local and free unless using `llm_anchor` chunking.

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or create a `.env` file (not committed):

```
OPENAI_API_KEY=sk-your-key-here
```

> **Tip**: If you use a local LLM (Ollama, vLLM, etc.), set `OPENAI_BASE_URL` instead:
> ```bash
> export OPENAI_BASE_URL="http://localhost:11434/v1"
> ```
> And change `llm_model` in `config/default.yaml` to your model name.

### Step 5 — Verify installation

```bash
python -c "
from src.utils.logger import get_logger
from src.utils.embedding import EmbeddingModel
import spacy, scipy, numpy, sklearn, hdbscan, faiss, networkx, yaml, rich
print('All imports OK')
logger = get_logger('test')
logger.step('Setup', 'Installation verified successfully')
"
```

You should see a green panel: `[STEP 1] [Setup] Installation verified successfully`

---

## 3. Project Structure

```
hyp_dlm/
├── config/
│   └── default.yaml               # All hyperparameters (edit this)
├── src/
│   ├── utils/                      # Shared utilities
│   │   ├── logger.py               # Debug logging system
│   │   ├── embedding.py            # Sentence-transformer wrapper
│   │   ├── llm_client.py           # OpenAI-compatible LLM client
│   │   └── sparse_ops.py           # Sparse matrix operations
│   ├── indexing/                    # Phase 1: Offline Indexing
│   │   ├── semantic_chunker.py     # SemanticChunker + AnchorChunker + factory
│   │   ├── ner_extractor.py        # Chunk → named entities (batch support)
│   │   ├── entity_linker.py        # Entity synonym detection (short-entity bypass)
│   │   ├── hypergraph_builder.py   # Build/expand incidence matrix H
│   │   ├── masking_strategy.py     # HDBSCAN / KMeans / FAISS / None
│   │   ├── embedding_store.py      # Compute & store embeddings
│   │   └── bipartite_storage.py    # Bipartite graph for incremental updates
│   ├── retrieval/                   # Phase 2: Online Retrieval
│   │   ├── router.py               # Familiarity-based routing
│   │   ├── query_decomposer.py     # LLM-based query → DAG
│   │   ├── propagation.py          # Core: DAG-guided matrix propagation
│   │   ├── passage_ranker.py       # PPR passage ranking
│   │   └── chunk_retriever.py      # Dense retrieval fallback
│   └── generation/                  # Phase 3: Answer Generation
│       └── generator.py            # Hybrid RAG generator
├── scripts/
│   ├── index_corpus.py             # CLI: index documents
│   ├── query.py                    # CLI: ask a question
│   └── evaluate.py                 # CLI: benchmark evaluation
├── tests/                          # Unit + integration tests
├── data/
│   ├── raw/                        # Put your .txt / .md documents here
│   ├── indexed/                    # Output: hypergraph artifacts + manifest.json
│   └── benchmarks/                 # Evaluation datasets (HotpotQA, etc.)
└── run.sh                          # One-command convenience script
```

---

## 4. Configuration

Edit `config/default.yaml` to tune the system. Key settings:

### Logging (see every step)

```yaml
logging:
  level: "DEBUG"           # DEBUG = see everything, INFO = key steps only
```

### Chunking Method

```yaml
chunking:
  method: "similarity"     # "similarity" (embedding cosine, no LLM cost)
                           # "llm_anchor" (LLM boundary detection, near-zero output tokens)
  llm_model: "gpt-4o-mini" # Model for llm_anchor method
```

- `"similarity"` — Uses sentence embedding cosine similarity to detect breakpoints. Fast, free, no API key needed.
- `"llm_anchor"` — Inserts `[1]`, `[2]`, ... markers at sentence boundaries, LLM identifies which markers start new knowledge segments. Better boundary detection but requires API key during indexing.

### NER Backend

```yaml
ner:
  model: "spacy"           # "spacy" (fast, default) or "gliner" (zero-shot)
```

### Entity Linking

```yaml
entity_linking:
  lexical_prefilter: 0.7          # JW threshold for standard path
  synonym_threshold: 0.9          # Combined threshold for standard path
  short_entity_max_len: 5         # Entities <= this use short-entity bypass
  short_entity_threshold: 0.85    # Semantic-only threshold for short entities
```

The **short-entity bypass** catches acronym synonyms (e.g., WHO ↔ World Health Organization) that fail Jaro-Winkler pre-filtering. Entities with `min(len(v_i), len(v_j)) <= 5` skip lexical comparison and rely on embedding similarity alone with a higher threshold.

### Masking Strategy

```yaml
masking:
  strategy: "hdbscan"      # Options: "hdbscan", "kmeans", "faiss_direct", "none"
```

### Propagation

```yaml
propagation:
  strategy: "damped_ppr"   # "damped_ppr" (recommended) or "max_update" (ablation)
  alpha: 0.85              # Damping factor (0.5 = more query, 0.95 = more graph)
```

The modulation matrix D_i uses **ReLU gating**: `D_i[j,j] = max(0, cosine(g_i, f(c_j))) * mask_j`. This zeros out anti-correlated hyperedges instead of allowing negative propagation weights.

### Indexing (parallelism & incremental)

```yaml
indexing:
  max_workers: 4             # Thread pool size for parallel chunking
  parallel_chunking: true    # Parallel LLM calls (llm_anchor only, I/O-bound)
  parallel_ner: true         # Batch NER via SpaCy nlp.pipe()
  ner_batch_size: 50         # Batch size for SpaCy pipe
  incremental: true          # Enable incremental updates (SHA-256 manifest)
```

### LLM Model

```yaml
decomposition:
  llm_model: "gpt-4o-mini"  # Used for query decomposition
generation:
  llm_model: "gpt-4o-mini"  # Used for answer generation
```

---

## 5. Prepare Your Data

Place your documents as `.txt` or `.md` files in `data/raw/`:

```bash
mkdir -p data/raw

# Example: create a sample document
cat > data/raw/sample_science.txt << 'EOF'
Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.

Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences. Curie discovered the elements polonium and radium. She founded the Curie Institutes in Paris and Warsaw.

Isaac Newton was an English mathematician and physicist who is widely recognized as one of the most influential scientists of all time. He formulated the laws of motion and universal gravitation. Newton also made contributions to optics and shares credit with Gottfried Wilhelm Leibniz for developing infinitesimal calculus. He was born in 1643 in Woolsthorpe, England.
EOF

cat > data/raw/sample_tech.txt << 'EOF'
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975. The company is headquartered in Redmond, Washington. Microsoft is best known for its Windows operating system and Office productivity suite. Satya Nadella has been the CEO of Microsoft since 2014.

Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976. The company is headquartered in Cupertino, California. Apple revolutionized the smartphone industry with the iPhone in 2007. Tim Cook succeeded Steve Jobs as CEO in 2011.

Google was founded by Larry Page and Sergey Brin in September 1998 while they were Ph.D. students at Stanford University. The company is now a subsidiary of Alphabet Inc. Google dominates the search engine market and also develops the Android mobile operating system. Sundar Pichai serves as the CEO of both Google and Alphabet.
EOF
```

---

## 6. Run the Pipeline

### Option A: Use the convenience script

```bash
chmod +x run.sh
./run.sh
```

This runs all 3 phases interactively. See [the script](#convenience-script) below.

### Option B: Run each step manually

#### Phase 1 — Index your documents

```bash
# Full build (default on first run, or when incremental is disabled)
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed \
  --config config/default.yaml

# Force full rebuild (ignores manifest, re-indexes everything)
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed \
  --config config/default.yaml \
  --full-rebuild
```

> **Note**: Indexing is fully local and free with `method: "similarity"`.
> With `method: "llm_anchor"`, each document costs ~1 LLM call with near-zero output tokens.

**What you will see in the logs:**
```
[STEP 1]  [Embedding]          Loaded model 'all-MiniLM-L6-v2'
[STEP 2]  [SemanticChunker]    Initialized (threshold=0.5)
[STEP 3]  [SemanticChunker]    Document 'sample_science' -> 6 chunks
[STEP 4]  [NERExtractor]       Extracted entities from 12 chunks
[STEP 5]  [Entity Dedup]       Global unique entities: 28
[STEP 6]  [EntityLinker]       Synonym matrix built: 3 synonym pairs
[STEP 7]  [HypergraphBuilder]  Incidence matrix H: (28, 12)
[STEP 8]  [HDBSCAN Masking]    Found 4 clusters, 2 noise hyperedges
[STEP 9]  [BipartiteStorage]   Built bipartite graph: 40 nodes, 45 edges
[STEP 10] [SaveIndex]          All artifacts saved to data/indexed
```

#### Phase 2+3 — Ask a question (uses LLM)

```bash
python scripts/query.py \
  --index_dir data/indexed \
  --query "Who founded Microsoft and where is it headquartered?" \
  --config config/default.yaml
```

**What you will see in the logs:**
```
[STEP 1]  [Query]             Query encoded
[STEP 2]  [Router]            Route decision: 'graph' (mean_score=0.42, entropy=2.1)
[STEP 3]  [QueryDecomposer]   Decomposed into 2 sub-questions
              q1: 'Who founded Microsoft?' (depends_on=[])
              q2: 'Where is Microsoft headquartered?' (depends_on=[])
[STEP 4]  [DAGPropagation]    Starting propagation for 2 sub-questions
              PPR step 1: delta=0.324, active=8, max=0.91
              PPR step 2: delta=0.089, active=12, max=0.87
              PPR converged at step 3
[STEP 5]  [PassageRanker]     Ranked 10 passages via PPR
[STEP 6]  [Generator]         Generated answer (85 chars)

ANSWER: Microsoft was founded by Bill Gates and Paul Allen. It is headquartered in Redmond, Washington.
```

#### Save output to JSON

```bash
python scripts/query.py \
  --index_dir data/indexed \
  --query "Who is older, Einstein or Newton?" \
  --config config/default.yaml \
  --output results/output.json
```

#### Evaluate on a benchmark

```bash
python scripts/evaluate.py \
  --index_dir data/indexed \
  --benchmark data/benchmarks/hotpotqa_dev.json \
  --config config/default.yaml \
  --output results/eval_hotpotqa.json \
  --max_samples 50
```

---

## 7. Understanding the Logs

The logging system has 4 levels of detail:

| Level | What you see | When to use |
|-------|--------------|-------------|
| `DEBUG` | Everything: matrix stats, convergence per step, entity matches | Debugging / development |
| `INFO` | Key pipeline steps only | Normal usage |
| `WARNING` | Only problems and fallbacks | Production |
| `ERROR` | Only failures | Production |

### Log output types

| Log Type | Example | What it tells you |
|----------|---------|-------------------|
| **STEP panel** | `[STEP 3] [NERExtractor] Extracted 150 entities` | Pipeline progress |
| **Matrix panel** | `Matrix: H, Shape: (200, 50), NNZ: 380` | Sparse matrix health |
| **Convergence** | `PPR step 2: delta=0.03, active=15` | Propagation is converging |
| **Summary table** | Tabular stats at end of indexing/evaluation | Final results |

### Change log level

In `config/default.yaml`:
```yaml
logging:
  level: "INFO"    # Change from "DEBUG" to see less output
```

---

## 8. Run Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_propagation.py -v
python -m pytest tests/test_chunker.py -v              # Includes AnchorChunker tests

# Run a single test
python -m pytest tests/test_e2e.py::test_hypergraph_builder -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

**Expected test output (36 tests):**
```
tests/test_chunker.py::test_sentence_tokenize                  PASSED
tests/test_chunker.py::test_chunk_document_basic                PASSED
tests/test_chunker.py::test_robust_sentence_split_basic         PASSED
tests/test_chunker.py::test_robust_sentence_split_abbreviations PASSED
tests/test_chunker.py::test_robust_sentence_split_decimals      PASSED
tests/test_chunker.py::test_insert_markers                      PASSED
tests/test_chunker.py::test_parse_llm_response_normal           PASSED
tests/test_chunker.py::test_parse_llm_response_bracket_wrapped  PASSED
tests/test_chunker.py::test_parse_llm_response_missing_one      PASSED
tests/test_chunker.py::test_create_chunker_similarity           PASSED
tests/test_chunker.py::test_create_chunker_llm_anchor           PASSED
tests/test_chunker.py::test_create_chunker_invalid              PASSED
tests/test_ner.py::test_spacy_extraction                        PASSED
tests/test_ner.py::test_entity_normalization                    PASSED
tests/test_ner.py::test_deduplication                           PASSED
tests/test_propagation.py::test_damped_ppr_convergence          PASSED
tests/test_propagation.py::test_max_update_propagation          PASSED
tests/test_propagation.py::test_sparse_ops                      PASSED
tests/test_propagation.py::test_build_propagation_matrix        PASSED
tests/test_e2e.py::test_hypergraph_builder                      PASSED
tests/test_e2e.py::test_masking_kmeans                          PASSED
tests/test_e2e.py::test_router                                  PASSED
tests/test_e2e.py::test_passage_ranker                          PASSED
```

---

## 9. Incremental Updates

When `indexing.incremental: true` (default), the pipeline tracks document content via SHA-256 hashes in `manifest.json`. On subsequent runs, only new documents are processed.

### How it works

1. **First run**: Full indexing, creates `manifest.json` alongside index artifacts
2. **Subsequent runs**: Compares current documents against manifest
   - **New docs**: Processed incrementally (chunk, NER, expand H/S, update masking)
   - **Changed docs**: Triggers full rebuild recommendation
   - **Deleted docs**: Triggers full rebuild recommendation
   - **No changes**: Prints "Index is up to date" and exits

### Usage

```bash
# First run — full build, creates manifest
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed

# Add a new document
cp new_article.txt data/raw/

# Second run — only processes new_article.txt
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed

# Third run — no changes detected
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed
# Output: "No new or changed documents. Index is up to date."

# Force full rebuild (ignores manifest)
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed \
  --full-rebuild
```

### What happens incrementally

| Component | Incremental method |
|-----------|-------------------|
| Chunking | Only new documents are chunked |
| NER | Only new chunks are processed |
| Entity embeddings | New embeddings computed and concatenated |
| Synonym matrix S | `expand_synonym_matrix()` — old S preserved, new links added |
| Incidence matrix H | `expand()` — old H preserved, new columns added |
| Masking strategy | `add_hyperedges()` — strategy updated with new embeddings |
| Bipartite graph | `add_document()` — new nodes/edges added |

### When to use `--full-rebuild`

- After editing or deleting existing documents
- After changing chunking config (e.g., `method`, `min_chunk_tokens`)
- After changing NER config (e.g., switching from `spacy` to `gliner`)
- After changing entity linking thresholds
- If index seems corrupted or inconsistent

---

## 10. Troubleshooting

### "No module named 'src'"

You must run commands from inside `hyp_dlm/`:
```bash
cd /path/to/Hypdlm_rag/hyp_dlm
```

### "SpaCy model not found"

```bash
python -m spacy download en_core_web_sm
```

### "OPENAI_API_KEY not set"

Indexing works without it. Only query/generation needs it:
```bash
export OPENAI_API_KEY="sk-..."
```

### "HDBSCAN: Found 0 clusters"

Your corpus is too small for the default `min_cluster_size: 15`.
Lower it in config:
```yaml
masking:
  hdbscan:
    min_cluster_size: 5   # Lower for small corpora
    min_samples: 2
```

Or switch to a simpler strategy:
```yaml
masking:
  strategy: "kmeans"      # Works better on small datasets
```

### Memory issues with large corpora

- Lower `embedding.batch_size` to `64` or `32`
- Use `faiss_direct` masking strategy (no clustering overhead)
- Set `logging.level: "WARNING"` to reduce output

### Using a local LLM instead of OpenAI

```bash
# Example with Ollama
export OPENAI_BASE_URL="http://localhost:11434/v1"
```

Then in `config/default.yaml`:
```yaml
decomposition:
  llm_model: "llama3"
generation:
  llm_model: "llama3"
chunking:
  llm_model: "llama3"    # Only needed if method: "llm_anchor"
```

### LLM anchor chunking fails or produces bad boundaries

If using `method: "llm_anchor"` and getting poor chunk boundaries:

- Ensure `OPENAI_API_KEY` (or `OPENAI_BASE_URL` for local LLMs) is set
- Some local models may not follow the "comma-separated numbers only" instruction well. Try a more capable model or fall back to `method: "similarity"`
- Check logs for `LLM anchor response:` — the response should be just numbers like `1, 3, 5, 8`

### Incremental indexing falls back to full rebuild

This happens when documents are changed or deleted. The incremental path only handles **new** documents. To handle edits/deletions:

```bash
python scripts/index_corpus.py \
  --input_dir data/raw \
  --output_dir data/indexed \
  --full-rebuild
```
