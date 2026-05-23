#!/usr/bin/env bash
# Run from project root. Shows what would be committed (respecting .gitignore).
# Use before: git add . && git status
set -e
cd "$(dirname "$0")/.."
echo "=== Files tracked / to be committed (excluding .gitignore) ==="
git status --short 2>/dev/null || echo "Not a git repo yet. Run: git init"
echo ""
echo "=== Ignored patterns (from .gitignore) ==="
grep -v '^#' .gitignore | grep -v '^$' | sed 's/^/  /'
