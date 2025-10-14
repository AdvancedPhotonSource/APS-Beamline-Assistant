#!/bin/bash
# Test MIDAS tools directly using UV environment

cd "$(dirname "$0")"

echo "======================================================================="
echo "  Testing MIDAS Tools (Direct Call)"
echo "======================================================================="
echo ""

# Run validation using UV environment
uv run python3 run_midas_validation.py

echo ""
echo "======================================================================="
