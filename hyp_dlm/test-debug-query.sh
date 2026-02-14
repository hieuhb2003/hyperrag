#!/bin/bash
# Test query with debug mode (requires indexed data first)

echo "=== Test: Query WITH debug mode ==="
echo ""
echo "Make sure you've run test-debug-index.sh first!"
echo ""

# Sample queries
QUERIES=(
  "Who is Albert Einstein?"
  "Where is the Eiffel Tower?"
  "What is Python?"
  "Who wrote Romeo and Juliet?"
)

for query in "${QUERIES[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Query: $query"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  python scripts/query.py \
    --index_dir data/indexed_test \
    --query "$query" \
    --config config/default.yaml \
    --debug \
    --output /tmp/query_result.json

  echo ""
  echo "=== Query result saved to /tmp/query_result.json ==="
  if [ -f "/tmp/query_result.json" ]; then
    python -m json.tool /tmp/query_result.json
  fi
  echo ""
done

echo "=== All debug checkpoints saved to data/debug/ ==="
ls -1 data/debug/query_step*.json 2>/dev/null | head -10
