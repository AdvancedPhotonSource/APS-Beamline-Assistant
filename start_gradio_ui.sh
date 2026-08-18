#!/bin/bash
# APEXA Gradio UI Startup Script
# Launches conversational interface for beamline analysis

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Change to script directory
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  APEXA - Advanced Photon EXperiment Assistant                       ║"
echo "║  Gradio Conversational UI                                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if servers.config exists
if [ ! -f "servers.config" ]; then
    echo "Error: servers.config not found!"
    echo "Creating default configuration..."
    cat > servers.config << 'EOF'
# Core servers (must match the checked-in servers.config)
core:beamline_core_server.py
midas:midas_comprehensive_server.py
motor:epics_motor_server.py
EOF
fi

echo "Loading MCP servers from servers.config..."
echo ""

# Start Gradio UI
echo "🚀 Launching Gradio UI on http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

uv run python gradio_ui.py

echo ""
echo "APEXA Gradio UI stopped."
