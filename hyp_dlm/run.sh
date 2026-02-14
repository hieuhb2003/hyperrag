#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# HyP-DLM — One-command runner
# ═══════════════════════════════════════════════════════════
#
# Usage:
#   ./run.sh                     # Interactive menu
#   ./run.sh setup               # Install everything
#   ./run.sh index               # Index documents in data/raw/
#   ./run.sh query "Your question here"
#   ./run.sh eval                # Run benchmark evaluation
#   ./run.sh test                # Run all tests
#   ./run.sh demo                # Create sample data + index + query
#
# ═══════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/default.yaml"
DATA_RAW="data/raw"
DATA_INDEXED="data/indexed"
RESULTS_DIR="results"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  HyP-DLM: Hypergraph Propagation with Dynamic Logic Modulation${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ─────────────────────────────────────────────────────────
# setup: Install all dependencies
# ─────────────────────────────────────────────────────────
cmd_setup() {
    print_header
    print_step "Setting up HyP-DLM..."

    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_step "Python version: $PYTHON_VERSION"

    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        print_step "Python >= 3.10 OK"
    else
        print_error "Python >= 3.10 is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    # Create venv if not exists
    if [ ! -d ".venv" ]; then
        print_step "Creating virtual environment..."
        python3 -m venv .venv
    fi

    print_step "Activating virtual environment..."
    source .venv/bin/activate

    # Install dependencies
    print_step "Installing core dependencies..."
    pip install --upgrade pip -q
    pip install -e ".[dev]" -q

    # Download SpaCy model
    print_step "Downloading SpaCy English NER model..."
    python -m spacy download en_core_web_sm -q

    # Create data directories
    mkdir -p "$DATA_RAW" "$DATA_INDEXED" "$RESULTS_DIR" data/benchmarks

    # Verify
    print_step "Verifying installation..."
    python3 -c "
from src.utils.logger import get_logger
import spacy, scipy, numpy, sklearn, hdbscan, faiss, networkx, yaml, rich
logger = get_logger('setup')
logger.step('Setup', 'All dependencies installed and verified')
"

    echo ""
    echo -e "${GREEN}Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Set your OpenAI API key:  export OPENAI_API_KEY='sk-...'"
    echo "  2. Put .txt/.md files in:    data/raw/"
    echo "  3. Run indexing:             ./run.sh index"
    echo "  4. Ask a question:           ./run.sh query \"Your question\""
    echo ""
}

# ─────────────────────────────────────────────────────────
# index: Run the indexing pipeline
# ─────────────────────────────────────────────────────────
cmd_index() {
    print_header
    local input_dir="${1:-$DATA_RAW}"
    local output_dir="${2:-$DATA_INDEXED}"

    # Check for documents
    file_count=$(find "$input_dir" -maxdepth 1 \( -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$file_count" -eq 0 ]; then
        print_error "No .txt or .md files found in $input_dir"
        echo "  Put your documents in $input_dir/ and try again."
        echo "  Or run: ./run.sh demo"
        exit 1
    fi

    print_step "Indexing $file_count document(s) from $input_dir"
    print_step "Output: $output_dir"
    print_step "Config: $CONFIG"
    echo ""

    python3 scripts/index_corpus.py \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --config "$CONFIG"

    echo ""
    echo -e "${GREEN}Indexing complete!${NC} Artifacts saved to $output_dir"
    echo "  Run: ./run.sh query \"Your question here\""
}

# ─────────────────────────────────────────────────────────
# query: Run a single query
# ─────────────────────────────────────────────────────────
cmd_query() {
    print_header

    if [ -z "${1:-}" ]; then
        echo -e "${YELLOW}Usage:${NC} ./run.sh query \"Your question here\""
        echo ""
        read -rp "Enter your question: " QUERY
    else
        QUERY="$1"
    fi

    local index_dir="${2:-$DATA_INDEXED}"

    # Check index exists
    if [ ! -f "$index_dir/incidence_matrix_H.npz" ]; then
        print_error "No index found at $index_dir"
        echo "  Run indexing first: ./run.sh index"
        exit 1
    fi

    # Check API key
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        print_warn "OPENAI_API_KEY not set. Query decomposition & generation will fail."
        print_warn "Set it with: export OPENAI_API_KEY='sk-...'"
    fi

    print_step "Query: '$QUERY'"
    print_step "Index: $index_dir"
    echo ""

    mkdir -p "$RESULTS_DIR"

    python3 scripts/query.py \
        --index_dir "$index_dir" \
        --query "$QUERY" \
        --config "$CONFIG" \
        --output "$RESULTS_DIR/last_query.json"
}

# ─────────────────────────────────────────────────────────
# eval: Run benchmark evaluation
# ─────────────────────────────────────────────────────────
cmd_eval() {
    print_header

    local benchmark="${1:-data/benchmarks/hotpotqa_dev.json}"
    local index_dir="${2:-$DATA_INDEXED}"
    local max_samples="${3:-10}"

    if [ ! -f "$benchmark" ]; then
        print_error "Benchmark file not found: $benchmark"
        echo "  Place benchmark JSON in data/benchmarks/"
        exit 1
    fi

    print_step "Evaluating on: $benchmark"
    print_step "Index: $index_dir"
    print_step "Max samples: $max_samples"
    echo ""

    mkdir -p "$RESULTS_DIR"

    python3 scripts/evaluate.py \
        --index_dir "$index_dir" \
        --benchmark "$benchmark" \
        --config "$CONFIG" \
        --output "$RESULTS_DIR/eval_results.json" \
        --max_samples "$max_samples"
}

# ─────────────────────────────────────────────────────────
# test: Run all tests
# ─────────────────────────────────────────────────────────
cmd_test() {
    print_header
    print_step "Running all tests..."
    echo ""

    python3 -m pytest tests/ -v --tb=short

    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
}

# ─────────────────────────────────────────────────────────
# demo: Create sample data, index it, then query
# ─────────────────────────────────────────────────────────
cmd_demo() {
    print_header
    print_step "Running demo pipeline..."

    # Create sample documents
    mkdir -p "$DATA_RAW"

    cat > "$DATA_RAW/sample_science.txt" << 'SAMPLE_EOF'
Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. He later moved to the United States and worked at the Institute for Advanced Study in Princeton, New Jersey.

Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences. Curie discovered the elements polonium and radium. She founded the Curie Institutes in Paris and Warsaw. Marie Curie was born on November 7, 1867, in Warsaw, Poland.

Isaac Newton was an English mathematician and physicist who is widely recognized as one of the most influential scientists of all time. He formulated the laws of motion and universal gravitation. Newton also made contributions to optics and shares credit with Gottfried Wilhelm Leibniz for developing infinitesimal calculus. He was born on January 4, 1643, in Woolsthorpe, England. Newton served as Lucasian Professor of Mathematics at the University of Cambridge.
SAMPLE_EOF

    cat > "$DATA_RAW/sample_tech.txt" << 'SAMPLE_EOF'
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975, in Albuquerque, New Mexico. The company later moved its headquarters to Redmond, Washington. Microsoft is best known for its Windows operating system and Office productivity suite. Satya Nadella has been the CEO of Microsoft since 2014. The company also owns LinkedIn, GitHub, and the Xbox gaming brand.

Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, in a garage in Los Altos, California. The company is headquartered in Cupertino, California. Apple revolutionized the smartphone industry with the iPhone in 2007. Tim Cook succeeded Steve Jobs as CEO in August 2011 after Jobs resigned due to health issues. Steve Jobs passed away on October 5, 2011.

Google was founded by Larry Page and Sergey Brin in September 1998 while they were Ph.D. students at Stanford University in California. The company is now a subsidiary of Alphabet Inc., which was created in 2015 as a restructuring of Google. Google dominates the search engine market and also develops the Android mobile operating system. Sundar Pichai serves as the CEO of both Google and Alphabet since December 2019.
SAMPLE_EOF

    print_step "Created sample documents in $DATA_RAW/"
    echo ""

    # Index
    print_step "Phase 1: Indexing documents..."
    echo ""
    python3 scripts/index_corpus.py \
        --input_dir "$DATA_RAW" \
        --output_dir "$DATA_INDEXED" \
        --config "$CONFIG"

    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Demo indexing complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Your index is ready at: $DATA_INDEXED/"
    echo ""
    echo "Now try querying (requires OPENAI_API_KEY):"
    echo "  ./run.sh query \"Who founded Microsoft?\""
    echo "  ./run.sh query \"Where was Einstein born?\""
    echo "  ./run.sh query \"Who is older, the founder of Microsoft or the founder of Apple?\""
    echo ""
}

# ─────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────
cmd_menu() {
    print_header

    echo "What would you like to do?"
    echo ""
    echo "  1) setup    — Install all dependencies"
    echo "  2) demo     — Create sample data + index (no API key needed)"
    echo "  3) index    — Index documents from data/raw/"
    echo "  4) query    — Ask a question"
    echo "  5) eval     — Run benchmark evaluation"
    echo "  6) test     — Run unit tests"
    echo "  7) exit"
    echo ""
    read -rp "Choose [1-7]: " choice

    case "$choice" in
        1) cmd_setup ;;
        2) cmd_demo ;;
        3) cmd_index ;;
        4)
            read -rp "Enter your question: " q
            cmd_query "$q"
            ;;
        5) cmd_eval ;;
        6) cmd_test ;;
        7) echo "Bye!" ; exit 0 ;;
        *) print_error "Invalid choice: $choice" ; exit 1 ;;
    esac
}

# ─────────────────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────────────────
case "${1:-menu}" in
    setup)   cmd_setup ;;
    index)   cmd_index "${2:-}" "${3:-}" ;;
    query)   cmd_query "${2:-}" "${3:-}" ;;
    eval)    cmd_eval "${2:-}" "${3:-}" "${4:-}" ;;
    test)    cmd_test ;;
    demo)    cmd_demo ;;
    menu)    cmd_menu ;;
    help|-h|--help)
        echo "Usage: ./run.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  setup                          Install dependencies + SpaCy model"
        echo "  demo                           Create sample data, index, ready to query"
        echo "  index [input_dir] [output_dir] Index documents"
        echo "  query \"question\" [index_dir]   Ask a question"
        echo "  eval [benchmark] [index_dir] [max_samples]  Run evaluation"
        echo "  test                           Run all unit tests"
        echo "  help                           Show this help"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run ./run.sh help for usage."
        exit 1
        ;;
esac
