#!/bin/bash
# Start Web Server with Image Viewer
# Uses existing .venv managed by UV

cd "$(dirname "$0")"

echo "🔬 Starting Beamline Assistant Web Server + Image Viewer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: .venv not found!"
    echo "   Run: uv sync"
    exit 1
fi

echo "✓ Using virtual environment: .venv"

# Build React frontend if dist is missing or stale
if [ -d "frontend" ]; then
    if [ ! -f "frontend/dist/index.html" ] || \
       [ "frontend/src/components/viz/VizLauncher.tsx" -nt "frontend/dist/index.html" ] || \
       [ "frontend/src/components/viz/VizPanel.tsx" -nt "frontend/dist/index.html" ]; then
        echo ""
        echo "🔨 Building React frontend..."
        (cd frontend && npm install --silent && npm run build) 2>&1 | tail -3
        echo "✓ Frontend built"
    fi
fi

echo ""
echo "🚀 Starting server..."
echo "   Web UI:         http://localhost:8001"
echo "   MIDAS Viewers:  Sidebar → Viewers icon → Scan a directory"
echo "   Image Viewer:   /api/viewer/* endpoints"
echo "   Viz API:        /api/viz/* endpoints"
echo ""
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run with venv Python directly
.venv/bin/python3 web_server.py
