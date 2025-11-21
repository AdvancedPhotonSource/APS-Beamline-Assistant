#!/bin/bash
# Build APEXA documentation to static HTML
# Output will be in the site/ directory

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")"

echo "======================================================================="
echo "  APEXA Documentation Builder"
echo "======================================================================="
echo ""
echo "Building documentation..."

uv run mkdocs build

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Documentation built successfully!"
    echo ""
    echo "Output directory: site/"
    echo ""
    echo "To view locally, open: site/index.html"
else
    echo ""
    echo "✗ Build failed. Check the errors above."
    exit 1
fi
