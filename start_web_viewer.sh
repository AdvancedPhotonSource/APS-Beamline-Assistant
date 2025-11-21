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
echo ""
echo "🚀 Starting server..."
echo "   Web UI: http://localhost:8001"
echo "   Image Viewer: Available at /api/viewer/* endpoints"
echo ""
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run with venv Python directly
.venv/bin/python3 web_server.py
