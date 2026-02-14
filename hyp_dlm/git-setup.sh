#!/bin/bash
# Setup git repo and push to GitHub

set -e

REPO_URL="https://github.com/hieuhb2003/hyperrag.git"

echo "🔧 Setting up git repository..."
cd /Users/hieunguyenmanh/Desktop/Hypdlm_rag

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repo..."
    git init
    git config user.name "Developer"
    git config user.email "dev@example.com"
fi

# Add remote
echo "Adding remote: $REPO_URL"
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

# Create .gitignore
cat > .gitignore << 'EOF'
venv/
*.pyc
__pycache__/
.pytest_cache/
.coverage
*.egg-info/
dist/
build/
.env
.DS_Store
data/indexed*/
data/debug/
*.npy
*.npz
.idea/
.vscode/
EOF

# Stage files
git add -A
git commit -m "Initial commit: JSON input + debug toolkit for HyP-DLM" || echo "Nothing to commit"

# Create/checkout dev branch
git checkout -b dev 2>/dev/null || git checkout dev

# Push to dev
echo "Pushing to dev branch..."
git push -u origin dev

# Create master from dev
git checkout -b master 2>/dev/null || git checkout master
git push -u origin master

echo "✅ Git setup complete!"
echo "Branches created:"
git branch -a
