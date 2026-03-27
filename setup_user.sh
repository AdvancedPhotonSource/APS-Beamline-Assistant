#!/bin/bash
# APEXA - User Setup Script
# Creates .env configuration for a new user

set -e

echo ""
echo "  APEXA - Advanced Photon EXperiment Assistant"
echo "  User Setup"
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "  .env file already exists."
    read -p "  Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Setup cancelled."
        exit 0
    fi
fi

# Step 1: ANL username
echo "  1. ANL Username (domain username, not email)"
read -p "     Username: " ANL_USERNAME
if [ -z "$ANL_USERNAME" ]; then
    echo "  Error: username cannot be empty"
    exit 1
fi

# Step 2: AI Model
echo ""
echo "  2. Default AI Model"
echo "     1) gpt4o       - GPT-4o (fastest, recommended)"
echo "     2) gpt41mini   - GPT-4.1 Mini (1M context, fast)"
echo "     3) gpt54       - GPT-5.4 (1M context, most capable)"
echo "     4) claudesonnet45 - Claude Sonnet 4.5"
echo "     5) gemini25pro - Gemini 2.5 Pro (1M context)"
read -p "     Select [1]: " model_choice
model_choice=${model_choice:-1}

case $model_choice in
    1) ARGO_MODEL="gpt4o" ;;
    2) ARGO_MODEL="gpt41mini" ;;
    3) ARGO_MODEL="gpt54" ;;
    4) ARGO_MODEL="claudesonnet45" ;;
    5) ARGO_MODEL="gemini25pro" ;;
    *) ARGO_MODEL="gpt4o" ;;
esac

# Step 3: MIDAS path (optional)
echo ""
echo "  3. MIDAS Installation (auto-detected from ~/Git/MIDAS, ~/opt/MIDAS, etc.)"
read -p "     Custom path? (y/N): " -n 1 -r
echo

MIDAS_LINE="# MIDAS_PATH auto-detected"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "     Path: " MIDAS_PATH
    MIDAS_PATH="${MIDAS_PATH/#\~/$HOME}"
    if [ -d "$MIDAS_PATH" ]; then
        MIDAS_LINE="MIDAS_PATH=$MIDAS_PATH"
    else
        echo "     Warning: $MIDAS_PATH not found"
        read -p "     Use anyway? (y/N): " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] && MIDAS_LINE="MIDAS_PATH=$MIDAS_PATH"
    fi
fi

# Write .env
cat > .env << EOF
ANL_USERNAME=$ANL_USERNAME
ARGO_MODEL=$ARGO_MODEL
$MIDAS_LINE
EOF

chmod 600 .env

echo ""
echo "  Setup complete!"
echo "    Username: $ANL_USERNAME"
echo "    Model:    $ARGO_MODEL"
echo "    MIDAS:    ${MIDAS_PATH:-auto-detect}"
echo ""
echo "  Start APEXA:"
echo "    ./start_beamline_assistant.sh    # CLI"
echo "    ./start_gradio_ui.sh             # Gradio UI"
echo ""
