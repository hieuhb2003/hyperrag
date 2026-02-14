#!/bin/bash
# Test JSON input + debug mode for indexing

echo "=== Test 1: Index JSON input WITH debug mode ==="
python scripts/index_corpus.py \
  --input_json data/sample_docs.json \
  --output_dir data/indexed_test \
  --config config/default.yaml \
  --debug

echo ""
echo "=== Debug output saved to data/debug/ ==="
echo "Files created:"
ls -lh data/debug/ 2>/dev/null | grep -E "\.(json|npy|npz)" || echo "No debug files yet"

echo ""
echo "=== Quick peek at debug_report.json ==="
if [ -f "data/debug/debug_report.json" ]; then
  python -m json.tool data/debug/debug_report.json | head -20
fi

echo ""
echo "=== Test 2: Index WITHOUT debug mode (faster) ==="
python scripts/index_corpus.py \
  --input_json data/sample_docs.json \
  --output_dir data/indexed_test_no_debug \
  --config config/default.yaml

echo ""
echo "=== Done! ==="
