#!/usr/bin/env bash
# Day 2 sweep: 4 models × 4 configs × 50 standard tasks + 4 models × 2 safety modes × 50 scenarios.
#
# Parallelism strategy: one process per model (4 parallel workers), each
# stepping through all 4 configs sequentially. Argo Gateway tolerates ~4
# concurrent users from a single ANL account; going wider risks rate limiting.
#
# Safety sweep runs after the standard sweep finishes (avoid colliding on
# Argo concurrency limits). Each safety mode is also one process per model.
#
# Outputs go to benchmark/results/day2/<model>_<config>_*.json and
# benchmark/results/day2_safety/safety_<model>_<mode>_*.json
#
# Estimated wall-clock:
#   Standard: 50 tasks × 4 configs × ~15s avg = ~50 min per model worker
#   Safety:   50 scenarios × ~5s avg per LLM call = ~5 min per (model, mode)
#   Total parallel:  ~50 + 10 = ~60 min wall clock

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

OUT_STD="benchmark/results/day2"
OUT_SAFE="benchmark/results/day2_safety"
mkdir -p "$OUT_STD" "$OUT_SAFE"

MODELS=(gpt5mini gpt54 claudeopus47 gemini25pro)

echo "[$(date)] Standard sweep launching: 4 models × 4 configs × 50 tasks"
PIDS=()
for m in "${MODELS[@]}"; do
    (
        for c in single keyword dspy autogen; do
            echo "[$(date)] [$m/$c] start"
            uv run python benchmark/eval_harness.py \
                --model "$m" --config "$c" \
                --output-dir "$OUT_STD" \
                > "$OUT_STD/${m}_${c}.log" 2>&1
            echo "[$(date)] [$m/$c] done"
        done
    ) &
    PIDS+=($!)
done

echo "Standard sweep PIDs: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "[$(date)] Standard sweep complete"

echo "[$(date)] Safety sweep launching: 4 models × {tool-enforced, prompt-only} × 50 scenarios"
PIDS=()
for m in "${MODELS[@]}"; do
    (
        echo "[$(date)] [$m/tool-enforced] start"
        uv run python benchmark/eval_safety.py \
            --model "$m" --mock \
            --output-dir "$OUT_SAFE" \
            > "$OUT_SAFE/${m}_tool_enforced.log" 2>&1
        echo "[$(date)] [$m/tool-enforced] done"

        echo "[$(date)] [$m/prompt-only] start"
        uv run python benchmark/eval_safety.py \
            --model "$m" --mock --prompt-only \
            --output-dir "$OUT_SAFE" \
            > "$OUT_SAFE/${m}_prompt_only.log" 2>&1
        echo "[$(date)] [$m/prompt-only] done"
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "[$(date)] Safety sweep complete"

echo "[$(date)] Day 2 sweep complete. Results in $OUT_STD and $OUT_SAFE"
