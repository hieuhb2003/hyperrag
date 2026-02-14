# Debug Toolkit Quick Start Guide

## Overview

HyP-DLM now supports:
1. **JSON/JSONL input** — Load documents from structured files instead of directories
2. **Debug mode** — See detailed intermediate results and save checkpoints after each pipeline step

## Files Provided

- `data/sample_docs.json` — 10 sample documents about famous people, places, and concepts
- `test-debug-index.sh` — Test JSON input + debug indexing
- `test-debug-query.sh` — Test debug query (requires index first)

## Quick Start

### Step 1: Verify Setup
```bash
cd /Users/hieunguyenmanh/Desktop/Hypdlm_rag/hyp_dlm
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### Step 2: Index with Debug Mode
```bash
chmod +x test-debug-index.sh
./test-debug-index.sh
```

This will:
- Load 10 documents from `data/sample_docs.json`
- Run the full indexing pipeline
- Print Rich panels after each step
- Save checkpoints to `data/debug/`
- Create `data/debug/debug_report.json` with timing info

**What you'll see:**
- Chunking stats (chunks created, token distribution)
- NER results (entities extracted by type)
- Entity linking (synonym pairs)
- Masking strategy details
- Final summary table

### Step 3: Query with Debug Mode
```bash
chmod +x test-debug-query.sh
./test-debug-query.sh
```

This will:
- Ask 4 sample questions
- Run retrieval pipeline with debug output
- Show routing decision, decomposition DAG, propagation results
- Save checkpoints for each step

**What you'll see:**
- Route decision (direct/graph/hybrid)
- Query decomposition DAG
- Entity activation from propagation
- Ranked passages with scores
- Final answer

## JSON Input Format

```json
[
  {
    "id": "unique_id",
    "text": "Document content here",
    "path": "optional/source/path",
    "metadata": { "optional": "custom fields" }
  },
  ...
]
```

- **id**: Required, string, unique identifier
- **text**: Required, string, document content (non-empty)
- **path**: Optional, source file path
- **metadata**: Optional, dict for custom metadata (flows through to chunks)

## CLI Flags

### Indexing
```bash
python scripts/index_corpus.py \
  --input_json data/sample_docs.json \    # OR --input_dir data/raw
  --output_dir data/indexed \
  --config config/default.yaml \
  --debug \                               # Enable debug mode
  --full-rebuild                          # Skip incremental, rebuild from scratch
```

### Querying
```bash
python scripts/query.py \
  --index_dir data/indexed \
  --query "Your question here?" \
  --config config/default.yaml \
  --debug \                               # Enable debug mode
  --output results.json                   # Save result
```

## Debug Output

### Indexing Debug Files
```
data/debug/
├── step01_documents.json       # Loaded documents
├── step02_chunking.json        # Chunks created
├── step03_ner.json             # Entities extracted
├── step04_dedup.json           # Entity deduplication stats
├── step05_embeddings.json      # Embedding shapes
├── step06_linking.json         # Synonym pairs
├── step07_hypergraph.json      # H matrix stats
├── step08_masking.json         # Masking strategy
├── step09_bipartite.json       # Bipartite graph
├── step10_save.json            # Saved files
└── debug_report.json           # Overall timing + summary
```

### Query Debug Files
```
data/debug/
├── query_step01_encode.json    # Query embedding
├── query_step02_route.json     # Route decision
├── query_step03_dense.json     # Dense retrieval
├── query_step04_decompose.json # Query DAG (if graph/hybrid)
├── query_step05_propagation.json # Entity scores
├── query_step06_ranking.json   # Ranked passages
├── query_step07_generation.json # Generated answer
└── debug_report.json           # Overall timing
```

## Understanding Debug Output

### Rich Panels
Terminal output shows colored panels:
- **Yellow border**: Indexing steps, routing, encoding
- **Magenta border**: Advanced analysis (matrices, propagation)
- **Green border**: Final results (generation)

### Checkpoints (JSON files)
Each checkpoint contains:
- **Counts**: How many items, totals
- **Stats**: Min/max/mean values
- **Distribution**: Breakdown by category
- **Samples**: Examples (first 3 by default)

### debug_report.json
Final JSON with:
```json
{
  "pipeline": "indexing" or "retrieval",
  "timings": [
    {"step": "step01_documents", "time_s": 0.123},
    ...
  ],
  "total_time_s": 5.678,
  "steps": 10,
  "extra": { ... }
}
```

## Troubleshooting

**Q: "JSON file must contain a top-level array"**
- Your JSON must be an array `[...]`, not an object `{...}`

**Q: "missing required field 'id'" or "'text'"**
- Each document needs at least `id` and `text` fields

**Q: Debug files not created**
- Check that `data/debug/` directory was created
- Verify `--debug` flag is passed
- Check `config/default.yaml` has `debug.enabled: false` (or true, then --debug overrides)

**Q: Query fails with "index not found"**
- Run indexing first: `python scripts/index_corpus.py ...`
- Use same `--output_dir` as `--index_dir` in query

## Next Steps

1. **Explore checkpoints**: Open `data/debug/*.json` to inspect intermediate results
2. **Modify queries**: Edit `test-debug-query.sh` to ask different questions
3. **Try JSONL**: Create `data/sample_docs.jsonl` with one doc per line
4. **Production use**: Use `--debug` in your own data pipeline

## Config Options

In `config/default.yaml`:

```yaml
debug:
  enabled: false              # Set true to always enable debug
  output_dir: "data/debug"   # Where to save checkpoints
  save_checkpoints: true      # Save JSON files
  save_report: true           # Save final report
  max_sample_items: 3         # How many samples to show
```

## Logger API (for advanced use)

If you're adding debug to custom code:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Enable debug mode
logger.enable_debug("data/debug", max_samples=5)

# Show samples in panel
logger.debug_sample("My Items", items_list)

# Show distribution
logger.debug_distribution("Category Counts", {"A": 10, "B": 5})

# Save checkpoint
logger.debug_checkpoint("my_step", {
    "count": 100,
    "matrix": scipy_sparse_matrix,
    "embedding": numpy_array,
})

# Final report
logger.debug_report(extra={"custom": "metadata"})
```

All debug calls are no-ops when debug is disabled — zero overhead!
