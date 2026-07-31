#!/bin/bash
# APEXA Desktop UI Startup Script
# Launches the web UI in a native desktop window (pywebview) — the same React
# frontend + FastAPI backend as the web server, no browser required.

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Prevent a stale VIRTUAL_ENV from another project causing uv warnings
unset VIRTUAL_ENV

# Change to script directory
cd "$(dirname "$0")"

echo "🖥  Starting APEXA Desktop UI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build the React frontend if the bundle is missing or any source file is newer.
if [ -d "frontend" ]; then
    NEED_BUILD=0
    if [ ! -f "frontend/dist/index.html" ]; then
        NEED_BUILD=1
    elif [ -n "$(find frontend/src -type f -newer frontend/dist/index.html 2>/dev/null | head -1)" ]; then
        NEED_BUILD=1
    fi
    if [ "$NEED_BUILD" = "1" ]; then
        echo "🔨 Building React frontend..."
        (cd frontend && npm install --silent && npm run build) 2>&1 | tail -3
        echo "✓ Frontend built"
    fi
fi

echo "🚀 Opening desktop window (close the window to stop)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python apexa_desktop.py
