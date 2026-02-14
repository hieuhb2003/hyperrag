#!/bin/bash
# Setup virtual environment and install dependencies

set -e

echo "📦 Setting up HyP-DLM environment..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
    echo "✓ venv created"
else
    echo "✓ venv already exists"
fi

# Activate venv
source venv/bin/activate
echo "✓ venv activated"

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -e ".[dev]" > /dev/null 2>&1
echo "✓ Package installed in editable mode"

# Download spaCy model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm > /dev/null 2>&1
echo "✓ en_core_web_sm downloaded"

# Verify setup
echo ""
echo "✅ Setup complete! To activate:"
echo "   source venv/bin/activate"
echo ""
echo "Then run:"
echo "   ./test-debug-index.sh"
echo "   ./test-debug-query.sh"
