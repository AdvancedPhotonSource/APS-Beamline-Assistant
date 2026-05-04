# APEXA-Bench

Reference harness, task suite, and aggregation scripts for the APEXA-Bench
paper. The benchmark evaluates LLM agents on synchrotron beamline workflows
spanning real-time data analysis (calibration, integration, Rietveld
refinement, HEDM) and safety-critical motor control.

This directory is the public artifact for the NeurIPS 2026 E&D Track
submission. Everything required to reproduce Tables 1–4 and Figures 1–3 of
the paper lives here.

## Layout

```
benchmark/
├── benchmark_tasks.json          50-task standard suite (with ground truth)
├── safety_suite.json             50-scenario adversarial safety suite
├── eval_harness.py               runs standard suite on (model, config)
├── eval_safety.py                runs safety suite (mock or live IOC)
├── build_tables.py               aggregates JSON → LaTeX table fragments
├── regen_nfig2.py                rebuilds Figure 2 from real data
├── run_day2_sweep.sh             full 4×4 standard sweep + safety
├── run_safety_real_ioc.sh        live-IOC variant of the safety sweep
├── detector_zoo/                 detector geometry presets used by tasks
└── results/
    ├── day2/                     standard sweep JSON outputs
    └── day2_safety/              safety sweep JSON outputs
```

## Prerequisites

```bash
uv sync                                  # installs runtime + benchmark deps
echo "ANL_USERNAME=<your_anl_id>" >> .env  # required for Argo Gateway
```

The harness assumes the parent project (`apexa_agents.py`,
`argo_mcp_client.py`, MCP servers) is installed and importable — `uv sync`
from the repo root handles this.

For the live-IOC safety sweep, EPICS Channel Access tools and a reachable
motor IOC are required:

```bash
export EPICS_MOTOR_PREFIX=20idMotSim   # or your beamline prefix
export EPICS_CA_ADDR_LIST=<ioc_host>:<port>
caget -t -w 3 ${EPICS_MOTOR_PREFIX}:m1.RBV   # smoke-test
```

## Reproducing the paper

### One-command full sweep (~60 min wall, model-level parallel)

```bash
./benchmark/run_day2_sweep.sh
```

This runs all 4 models × 4 configurations on the 50-task standard suite,
plus all 4 models × 2 modes (tool-enforced, prompt-only) on the safety
suite, writing JSON outputs into `benchmark/results/day2/` and
`benchmark/results/day2_safety/`.

### Individual cells

```bash
# Single (model, config) on the standard suite
uv run python benchmark/eval_harness.py --model gpt5mini --config dspy

# Safety suite — mock-IOC (no EPICS needed)
uv run python benchmark/eval_safety.py --model claudeopus47
uv run python benchmark/eval_safety.py --model claudeopus47 --prompt-only

# Safety suite — live IOC (paranoid: refuses to start without EPICS_MOTOR_PREFIX)
./benchmark/run_safety_real_ioc.sh
```

### Regenerating tables and figures

```bash
# LaTeX fragments for Tables 1–3
uv run python benchmark/build_tables.py > /tmp/tables.tex

# Figure 2 (4-panel results overview) from current sweep data
uv run python benchmark/regen_nfig2.py \
    --out manuscripts/1-CS-APEXA-Bench/figures
```

Figures 1 and 3 are static and produced by their scripts in
`manuscripts/1-CS-APEXA-Bench/figures/`.

## Models and configurations

| Field      | Values                                               |
|------------|------------------------------------------------------|
| `--model`  | `gpt5mini`, `gpt54`, `claudeopus47`, `gemini25pro`   |
| `--config` | `single`, `keyword`, `dspy`, `autogen`               |

Models are accessed via Argonne's Argo Gateway. Per-agent temperatures
(0.2–0.6 by specialist role) and the prompting protocol are described in
Section 5 of the paper and implemented in `apexa_agents.py`.

## Result schema

Each JSON output has the same top-level structure:

```json
{
  "metadata": { "model": "...", "config": "...", "started_at": "...", ... },
  "aggregate": {
    "overall":         { "success_rate": ..., "avg_apexa_score": ..., ... },
    "per_category":    { "calibration": { ... }, "integration": { ... }, ... },
    "per_difficulty":  { "L1": { ... }, "L2": { ... }, "L3": { ... } }
  },
  "per_task": [ { "task_id": "...", "correctness": true/false, "tools_called": [...], ... }, ... ]
}
```

`build_tables.py` reads these files and emits the exact LaTeX fragments
used in the paper, so any change to the suite or sweep can be propagated
to the manuscript with one command.

## Known issues

- **GPT-5-family prompt-only safety runs (fixed in this revision).** The
  earlier safety harness sent `temperature` to GPT-5-family models, which
  Argo rejects with HTTP 400. `apexa_agents.ArgoProvider._build_payload`
  now treats `gpt5*` like `gpto*` (no temperature, no top_p). The original
  failure mode is documented in the paper for transparency.

- **Single-run-per-cell.** Closed API endpoints make seed control
  infeasible and per-call cost is nontrivial, so each (model, config) cell
  is a single deterministic run of the 50-task suite. We report
  per-category breakdowns to expose within-suite variance; the central
  separation result (tool-enforced 0/200 vs. prompt-only 11/100, 75% on
  the large-slew category) is well outside any plausible single-run noise
  band.

## License

Code: MIT. Task suite definitions (`benchmark_tasks.json`,
`safety_suite.json`): CC-BY 4.0.
