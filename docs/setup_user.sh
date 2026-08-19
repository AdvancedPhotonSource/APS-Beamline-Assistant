#!/bin/bash
# Beamline Assistant - User Setup Script
# Sets up configuration for a new user

set -e

echo "======================================================================="
echo "  Beamline Assistant - User Setup"
echo "======================================================================="
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  Warning: .env file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Your existing .env was not modified."
        exit 0
    fi
fi

# Get ANL username
echo "Step 1: ANL Authentication"
echo "-------------------------"
read -p "Enter your ANL username: " ANL_USERNAME

if [ -z "$ANL_USERNAME" ]; then
    echo "Error: ANL username cannot be empty"
    exit 1
fi

# Select AI model
echo ""
echo "Step 2: AI Model Selection"
echo "-------------------------"
echo "Available models:"
echo "  1) gpt4o (GPT-4o - Fast, recommended)"
echo "  2) claudesonnet4 (Claude Sonnet 4)"
echo "  3) gemini25pro (Gemini 2.5 Pro)"
echo "  4) gpt4turbo (GPT-4 Turbo)"
echo ""
read -p "Select model [1]: " model_choice
model_choice=${model_choice:-1}

case $model_choice in
    1) ARGO_MODEL="gpt4o" ;;
    2) ARGO_MODEL="claudesonnet4" ;;
    3) ARGO_MODEL="gemini25pro" ;;
    4) ARGO_MODEL="gpt4turbo" ;;
    *) ARGO_MODEL="gpt4o" ;;
esac

# MIDAS path
echo ""
echo "Step 3: MIDAS Installation"
echo "-------------------------"
echo "The system will automatically search for MIDAS in this order:"
echo "  1. ~/Git/MIDAS"
echo "  2. ~/opt/MIDAS"
echo "  3. /home/beams/S*USER/opt/MIDAS (beamline systems)"
echo "  4. ~/MIDAS"
echo "  5. /opt/MIDAS"
echo "  6. ~/.MIDAS"
echo ""
read -p "Do you want to specify a custom MIDAS path? (y/N): " -n 1 -r
echo

MIDAS_PATH=""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter MIDAS path: " MIDAS_PATH
    # Expand ~ to home directory
    MIDAS_PATH="${MIDAS_PATH/#\~/$HOME}"

    # Check if path exists
    if [ ! -d "$MIDAS_PATH" ]; then
        echo "⚠️  Warning: Directory $MIDAS_PATH does not exist"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup cancelled."
            exit 0
        fi
    fi
fi

# Detect network reachability
echo ""
echo "Step 3: Network Environment"
echo "-------------------------"
echo "Probing what this host can reach (3s timeout each)..."

APEXA_NETWORK=$(python3 - <<'PYEOF'
import socket
def up(h, p=443, t=3.0):
    try:
        with socket.create_connection((h, p), timeout=t):
            return True
    except OSError:
        return False
web = up("huggingface.co")
anl = up("apps.inside.anl.gov") or up("inference-api.alcf.anl.gov")
print("web" if web else ("internal" if anl else "data"))
PYEOF
)

case "$APEXA_NETWORK" in
  web)
    echo "  ✓ public internet reachable  -> APEXA_NETWORK=web"
    echo "    All tools available, including Materials Project and DOI lookup."
    ;;
  internal)
    echo "  ✓ ANL internal reachable, no public internet -> APEXA_NETWORK=internal"
    echo "    Argo / ALCF / SSH / MIDAS all work. Web-only tools are disabled and"
    echo "    HuggingFace is forced offline, so startup cannot hang on a model"
    echo "    download (this is what stalled APEXA on copland)."
    echo ""
    echo "    IMPORTANT: pre-stage the RAG embedder cache on a web-connected"
    echo "    machine and copy ~/.cache/huggingface across, or query_hedm_knowledge"
    echo "    stays unavailable. See docs/OFFLINE_DEPLOYMENT.md."
    ;;
  *)
    echo "  ⚠ no network reachable -> APEXA_NETWORK=data"
    echo "    APEXA needs an LLM endpoint; check the VPN/host networking."
    ;;
esac

# Create .env file
echo ""
echo "Creating .env file..."

cat > .env << EOF
# Beamline Assistant Configuration
# Generated: $(date)

# ANL Authentication
ANL_USERNAME=$ANL_USERNAME

# AI Model
ARGO_MODEL=$ARGO_MODEL

# Network tier: web | internal | data  (detected at setup; edit if the host moves)
#   internal = ANL-only. Web-only tools are disabled and HuggingFace is forced
#   offline so the server cannot hang at startup on a model download.
APEXA_NETWORK=$APEXA_NETWORK

# MIDAS Installation Path
EOF

if [ -n "$MIDAS_PATH" ]; then
    echo "MIDAS_PATH=$MIDAS_PATH" >> .env
else
    echo "# MIDAS_PATH will be auto-detected" >> .env
fi

# Set secure permissions
chmod 600 .env

echo ""
echo "✓ Setup complete!"
echo ""
echo "Configuration saved to .env:"
echo "  - ANL Username: $ANL_USERNAME"
echo "  - AI Model: $ARGO_MODEL"
echo "  - Network tier: $APEXA_NETWORK"
if [ -n "$MIDAS_PATH" ]; then
    echo "  - MIDAS Path: $MIDAS_PATH"
else
    echo "  - MIDAS Path: Auto-detect"
fi
echo ""
echo "File permissions set to 600 (owner read/write only)"
echo ""
echo "To start the assistant, run:"
echo "  ./start_beamline_assistant.sh"
echo ""
echo "======================================================================="
