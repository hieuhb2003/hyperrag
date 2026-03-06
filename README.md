# HyP-DLM: Hypergraph Propagation with Dynamic Logic Modulation

Training-free hypergraph-based RAG system for multi-hop question answering.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Format

### Documents (`documents.json`)
```json
[
  {"id": "doc_001", "text": "Full passage text here..."},
  {"id": "doc_002", "text": "Another passage..."}
]
```

### Queries (`queries.json`)
```json
[
  {
    "id": "q_001",
    "query": "What is X?",
    "answer": "X is...",
    "evidence": ["doc_001", "doc_002"]
  }
]
```

## Usage

### Step 1: Index (Build Hypergraph)

```bash
python -m src.main index --data data/documents.json
```

This runs: Coreference Resolution -> Semantic Chunking -> NER -> Incidence Matrix -> Synonym Matrix -> Semantic Masking.

Artifacts saved to `storage/{dataset_name}/{embedding_model}/`.

### Step 2: Retrieve (Query Decomposition + Propagation)

```bash
python -m src.main retrieve --queries data/queries.json
```

This runs: Query Decomposition (DAG) -> State Initialization -> Semantic Masking -> Dynamic Modulation -> Damped PPR -> Multi-Granular Scoring.

Results saved to `output/{dataset_name}/retrieval-results.json` (top-10 chunks per query).

### Step 3: Generate (Answer Synthesis)

```bash
python -m src.main generate --queries data/queries.json
```

This runs: Trace Building -> Evidence Assembly -> LLM Answer Generation.

Results saved to `output/{dataset_name}/generation-results.json`.

### Step 4: Evaluate

```bash
python -m src.main evaluate
```

Computes Recall@K, Exact Match, F1. Report saved to `output/{dataset_name}/evaluation-report.json`.

### Run All (Steps 1-3)

```bash
python -m src.main run --data data/documents.json --queries data/queries.json
```

## CLI Options

```bash
python -m src.main --help
```

| Flag | Description | Example |
|------|-------------|---------|
| `--config` | Config YAML path | `--config my_config.yaml` |
| `--device` | Compute device | `--device cuda` |
| `--embedding-model` | Embedding model | `--embedding-model all-MiniLM-L6-v2` |
| `--ner-backend` | NER backend | `--ner-backend spacy` |
| `--max-workers` | Thread workers | `--max-workers 8` |
| `--dataset-name` | Dataset name | `--dataset-name musique` |
| `--log-level` | Log level | `--log-level DEBUG` |

## Configuration

Edit `config.yaml` to customize all hyperparameters. Key settings:

```yaml
device: "cuda"                    # cuda / cpu / mps
embedding_model: "all-mpnet-base-v2"
ner_backend: "gliner"             # gliner / spacy
chunk_breakpoint_percentile: 90.0 # semantic chunking threshold
ppr_alpha: 0.85                   # PPR damping factor
llm_model: "gpt-4o-mini"         # LLM for decomposition & generation
max_workers: 4                    # indexing threads
```

## Ablation

Change embedding model to compare results. Storage is separated by model name:

```bash
# Run with default embedding
python -m src.main run --data data/docs.json --queries data/queries.json

# Run with different embedding
python -m src.main run --data data/docs.json --queries data/queries.json \
  --embedding-model all-MiniLM-L6-v2

# Compare evaluations
python -m src.main evaluate --dataset-name default
```

## Output Structure

```
storage/{dataset_name}/{embedding_model}/  # Index artifacts
output/{dataset_name}/           # Retrieval & generation results
metrics/{dataset_name}/          # Token usage & timing JSON
logs/                            # Log files
```

## Metrics Tracked

- **Token usage**: input/output tokens per LLM call, per document (indexing), per query (retrieval + generation)
- **Timing**: avg time per document (indexing), per query (retrieval + generation)
- **Retrieval**: Recall@K (K=10) vs ground truth evidence
- **Generation**: Exact Match (EM), token-level F1 (SQuAD-style)

## Environment Variables

Set `OPENAI_API_KEY` for LLM calls (query decomposition + answer generation):

```bash
export OPENAI_API_KEY="sk-..."
```

## Project Structure

```
src/
  main.py                          # CLI entry point
  config.py                        # Central configuration
  logging-setup.py                 # Logging (console + file)
  token-tracker.py                 # LLM token usage tracking
  time-tracker.py                  # Phase timing tracking
  data-loader.py                   # Document/query JSON loader
  indexing/
    coreference-resolver.py        # FastCoref LingMessCoref
    semantic-chunker.py            # Embedding cosine breakpoints
    entity-extractor-base.py       # NER interface + factory
    entity-extractor-gliner.py     # GLiNER2 NER
    entity-extractor-spacy.py      # spaCy NER
    incidence-matrix-builder.py    # Sparse H[N x M] + normalization
    synonym-matrix-builder.py      # 8-stage TF-IDF + dense + RRF
    semantic-masker.py             # HDBSCAN clustering
    indexing-pipeline.py           # Orchestrator
  retrieval/
    query-decomposer.py            # LLM DAG decomposition
    state-initializer.py           # NER-based state seeding
    dag-propagator.py              # Damped PPR propagation
    multi-granular-scorer.py       # Composite scoring
    retrieval-pipeline.py          # Orchestrator
  generation/
    answer-generator.py            # Trace-augmented prompting
    generation-pipeline.py         # Orchestrator
  evaluation/
    metrics-aggregator.py          # Recall, EM, F1, aggregation
    output-saver.py                # JSON output persistence
```
