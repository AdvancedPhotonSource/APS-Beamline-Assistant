#!/bin/bash
# APEXA Trame Web UI Startup Script
# Pure-Python web interface — no Node.js required

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Change to script directory
cd "$(dirname "$0")"

echo "================================================================"
echo "  APEXA - Advanced Photon EXperiment Assistant"
echo "  Trame Web UI"
echo "================================================================"
echo ""

# Check if servers.config exists
if [ ! -f "servers.config" ]; then
    echo "Error: servers.config not found!"
    echo "Please create servers.config with MCP server definitions."
    exit 1
fi

echo "Loading MCP servers from servers.config..."
echo ""
echo "Launching Trame UI on http://localhost:8002"
echo "Press Ctrl+C to stop"
echo ""

uv run python trame_ui.py --port 8002

echo ""
echo "APEXA Trame UI stopped."
