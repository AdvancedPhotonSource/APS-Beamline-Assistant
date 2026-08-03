#!/bin/bash
# APEXA - User Setup Script
# Creates .env configuration for a new user

set -e

echo ""
echo "  APEXA - Advanced Photon EXperiment Assistant"
echo "  User Setup"
echo ""

# Check if .env already exists
# NOTE: use full-line reads (no `-n 1`) for every y/N prompt. `read -n 1` returns
# after one character WITHOUT consuming the Enter keystroke, and that leftover
# newline is then swallowed by the next full-line read — which silently blanked
# the MIDAS path prompt and made a typed "~/..." path leak into later fields.
if [ -f ".env" ]; then
    echo "  .env file already exists."
    read -p "  Overwrite? (y/N): " -r REPLY
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
echo "     1) gpt55        - GPT-5.5 (1M context, DEFAULT, most reliable tool calls)"
echo "     2) gpt54        - GPT-5.4 (1M context, strong all-round, lower cost)"
echo "     3) claudeopus48 - Claude Opus 4.8 (1M context, best planning)"
echo "     4) gpt4o        - GPT-4o (fastest, cheapest)"
echo "     5) gemini35flash - Gemini 3.5 Flash"
echo ""
echo "     Enter a number, or type any model name directly (e.g. gpt51, claudesonnet46)."
read -p "     Select [1]: " model_choice
model_choice=${model_choice:-1}

case $model_choice in
    1) ARGO_MODEL="gpt55" ;;
    2) ARGO_MODEL="gpt54" ;;
    3) ARGO_MODEL="claudeopus48" ;;
    4) ARGO_MODEL="gpt4o" ;;
    5) ARGO_MODEL="gemini35flash" ;;
    ""|*[!a-zA-Z0-9]*)
        # empty or contains punctuation → not a valid model token; fall back
        echo "     Unrecognized selection '$model_choice' — using default gpt55"
        ARGO_MODEL="gpt55" ;;
    *)
        # any other bare alphanumeric token → treat as a typed model name
        ARGO_MODEL="$model_choice"
        _KNOWN_MODELS=" gpt4o gpt41 gpt41mini gpt41nano gpto3mini gpto4mini gpt5 gpt5mini gpt5nano gpt51 gpt52 gpt54 gpt55 claudeopus48 claudeopus47 claudeopus46 claudeopus45 claudeopus41 claudesonnet46 claudesonnet45 claudehaiku45 gemini35flash gemini31flashlite gemini25pro gemini25flash "
        if [[ "$_KNOWN_MODELS" != *" $ARGO_MODEL "* ]]; then
            echo "     Note: '$ARGO_MODEL' is not in the known model list — saving it anyway."
        fi
        ;;
esac

# Step 3: MIDAS path
# Single full-line prompt with a default (Enter to accept). A leading ~ is
# expanded here so a literal tilde never reaches .env.
echo ""
echo "  3. MIDAS Installation"
DEFAULT_MIDAS="$HOME/opt/MIDAS_canonical"
echo "     Default: ~/opt/MIDAS_canonical"
echo "     (Enter to accept the default, type another path, or '-' to auto-detect)"
read -p "     Path: " -r MIDAS_INPUT
MIDAS_INPUT="${MIDAS_INPUT:-$DEFAULT_MIDAS}"

if [ "$MIDAS_INPUT" = "-" ] || [ "$MIDAS_INPUT" = "auto" ]; then
    MIDAS_LINE="# MIDAS_PATH auto-detected"
    MIDAS_PATH=""
else
    MIDAS_PATH="${MIDAS_INPUT/#\~/$HOME}"   # expand a leading ~
    MIDAS_LINE="MIDAS_PATH=$MIDAS_PATH"
    if [ ! -d "$MIDAS_PATH" ]; then
        echo "     Note: $MIDAS_PATH does not exist yet — saved anyway (create it or edit .env later)."
    fi
fi

# Step 4: Materials Project API key (optional)
echo ""
echo "  4. Materials Project API Key (optional, for CIF file fetching)"
echo "     Get one at: https://next-gen.materialsproject.org/api"
read -p "     API Key (Enter to skip): " MP_API_KEY

MP_LINE="# MP_API_KEY not set (optional, for fetch_cif_from_mp)"
if [ -n "$MP_API_KEY" ]; then
    MP_LINE="MP_API_KEY=$MP_API_KEY"
fi

# Write .env
cat > .env << EOF
ANL_USERNAME=$ANL_USERNAME
ARGO_MODEL=$ARGO_MODEL
$MIDAS_LINE
$MP_LINE
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
