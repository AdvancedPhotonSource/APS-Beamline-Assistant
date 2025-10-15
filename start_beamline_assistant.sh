#!/bin/bash
# APEXA Startup Script
# Advanced Photon EXperiment Assistant
# Loads analysis servers from servers.config

# Add uv to PATH (installed in ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Change to script directory
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  APEXA - Advanced Photon EXperiment Assistant                       ║"
echo "║  Your AI Scientist at the Beamline                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if servers.config exists
if [ ! -f "servers.config" ]; then
    echo "Error: servers.config not found!"
    echo "Creating default configuration..."
    cat > servers.config << 'EOF'
# Core servers
filesystem:filesystem_server.py
executor:command_executor_server.py
midas:midas_comprehensive_server.py
EOF
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Run ./setup_user.sh first."
    echo ""
fi

# Parse servers.config and build server list
SERVER_ARGS=""
ENABLED_SERVERS=()

echo "Loading analysis servers:"
while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    # Extract server name and path
    if [[ $line =~ ^([^:]+):(.+)$ ]]; then
        server_name="${BASH_REMATCH[1]}"
        server_path="${BASH_REMATCH[2]}"

        # Check if server file exists
        if [ -f "$server_path" ]; then
            SERVER_ARGS="$SERVER_ARGS ${server_name}:${server_path}"
            ENABLED_SERVERS+=("$server_name")
            echo "  ✓ $server_name ($server_path)"
        else
            echo "  ✗ $server_name (not found: $server_path)"
        fi
    fi
done < servers.config

echo ""
echo "Active Servers: ${ENABLED_SERVERS[@]}"
echo ""
echo "AI Models available via Argo Gateway:"
echo "  OpenAI: gpt4o (default), gpt4turbo, gpt4, gpt35"
echo "  Anthropic: claudesonnet4, claudeopus4"
echo "  Google: gemini25pro, gemini25flash"
echo ""
echo "======================================================================="
echo ""

# Check if we have any servers to start
if [ -z "$SERVER_ARGS" ]; then
    echo "Error: No valid servers found in servers.config"
    exit 1
fi

# Start the assistant with dynamically loaded servers
uv run argo_mcp_client.py $SERVER_ARGS

echo ""
echo "Beamline Assistant stopped."
