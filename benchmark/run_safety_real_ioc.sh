#!/usr/bin/env bash
# Live-IOC variant of the adversarial safety suite.
#
# Runs the same scenarios as run_day2_sweep.sh's safety phase but routes through
# the live epics_motor_server.py (real caget/caput) instead of the mock motor
# server. Intended for execution on a beamline workstation that can reach an
# EPICS IOC.
#
# Required environment:
#   EPICS_MOTOR_PREFIX     IOC prefix used by the motor records, e.g. "20idMotSim"
#                          or your beamline-specific prefix.
#   EPICS_CA_ADDR_LIST     IOC host:port list (only needed if not on the IOC subnet).
#
# Optional:
#   APEXA_SAFETY_MODELS    space-separated subset of models to run
#                          (default: gpt5mini gpt54 claudeopus47 gemini25pro)
#
# Output: benchmark/results/day2_safety_real_ioc/safety_<model>_<mode>_*.json
# These files share the schema with the mock-mode results, so build_tables.py
# can be pointed at this directory to regenerate Table~2 against live data.

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

if [[ -z "${EPICS_MOTOR_PREFIX:-}" ]]; then
    echo "ERROR: EPICS_MOTOR_PREFIX is not set." >&2
    echo "  Set it to your beamline IOC prefix, e.g. EPICS_MOTOR_PREFIX=20idMotSim" >&2
    exit 2
fi

if ! command -v caget >/dev/null 2>&1; then
    echo "ERROR: caget not found on PATH; EPICS Channel Access tools are required." >&2
    exit 2
fi

# Smoke-test the IOC before launching anything.
PROBE_PV="${EPICS_MOTOR_PREFIX}:m1.RBV"
if ! caget -t -w 3 "$PROBE_PV" >/dev/null 2>&1; then
    echo "ERROR: IOC probe failed: caget $PROBE_PV did not respond within 3s." >&2
    echo "  Check EPICS_CA_ADDR_LIST and that the IOC at prefix $EPICS_MOTOR_PREFIX is up." >&2
    exit 2
fi

OUT="benchmark/results/day2_safety_real_ioc"
mkdir -p "$OUT"

MODELS=(${APEXA_SAFETY_MODELS:-gpt5mini gpt54 claudeopus47 gemini25pro})

echo "[$(date)] Live-IOC safety sweep launching"
echo "  IOC prefix:  $EPICS_MOTOR_PREFIX"
echo "  Models:      ${MODELS[*]}"
echo "  Output:      $OUT"

PIDS=()
for m in "${MODELS[@]}"; do
    (
        echo "[$(date)] [$m/tool-enforced/live-ioc] start"
        uv run python benchmark/eval_safety.py \
            --model "$m" \
            --output-dir "$OUT" \
            > "$OUT/${m}_tool_enforced.log" 2>&1
        echo "[$(date)] [$m/tool-enforced/live-ioc] done"

        echo "[$(date)] [$m/prompt-only/live-ioc] start"
        uv run python benchmark/eval_safety.py \
            --model "$m" --prompt-only \
            --output-dir "$OUT" \
            > "$OUT/${m}_prompt_only.log" 2>&1
        echo "[$(date)] [$m/prompt-only/live-ioc] done"
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "[$(date)] Live-IOC safety sweep complete. Results in $OUT"
