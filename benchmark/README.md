# APEXA-Bench

Evaluation harness for AI agents on synchrotron beamline operations, released with the
APEXA framework it drives. **58 facility tasks** (`benchmark_tasks.json`) across six
categories — Calibration, Integration, HEDM Analysis, Motor Control, GSAS-II Refinement,
Domain Knowledge — plus a **50-scenario adversarial motor-safety suite**
(`safety_suite.json`). Tasks carry a four-class physical-consequence label and are graded
on tool selection + parameters within tolerance (not on adjudicating scientific unknowns).

See the paper's Artifact Description appendix for the mapping from these commands to
reported results.

## Setup

From the repository root (installs the framework + harness deps):

```bash
uv sync
```

Frontier-model access is via an institutional LLM gateway (`.env`: `ANL_USERNAME`,
`ARGO_MODEL`). The harness runs without credentials in `--dry-run` mode.

## Reproduce the paper's deterministic results

```bash
# 1. Validate all task / scenario definitions — no API calls
uv run python benchmark/eval_harness.py --dry-run
uv run python benchmark/eval_safety.py  --dry-run

# 2. Motor-safety gate, mock EPICS (no IOC needed).
#    Tool-enforced -> 0/200 violations (deterministic, holds for any model);
#    prompt-only    -> 15/200 (the baseline contrast, Fig. 4).
uv run python benchmark/eval_safety.py --mock
uv run python benchmark/eval_safety.py --mock --prompt-only
```

The five safety checks (soft-limit, limit-switch, slew-magnitude, velocity, jog-duration)
run in server-side code before any `caput`, so the 0/200 result is a deterministic
property of the tool layer, not a sampled quantity.

## Run the task benchmark (needs gateway credentials)

```bash
uv run python benchmark/eval_harness.py --model gpt54                    # full 58-task suite
uv run python benchmark/eval_harness.py --model gpt54 --category calibration
uv run python benchmark/eval_harness.py --model gpt54 --difficulty L3
```

Models used in the paper's safety sweep: `gpt5mini`, `gpt54`, `claudeopus47`,
`gemini25pro` (`--all-models`). Per-task JSON is written under `benchmark/results/`;
`build_tables.py` aggregates it. Interaction traces from the reported safety runs are in
`benchmark/results/day2_safety/` (4 models × {tool-enforced, prompt-only}).

## Cross-detector CI slice (`ref_01`–`ref_08`)

Grades one calibrate→integrate→refine pipeline across four detector geometries (Varex
aero, Varex distortion, Pilatus, GE) against a NIST-traceable CeO₂ lattice constant.
Shipped here: the reference structure (`detector_zoo/CeO2_NIST_5p41165.cif`), ground
truth (`detector_zoo/ground_truth.json`), scoring (`detector_zoo/score_crossengine.py`),
the fixed refinement path (`detector_zoo/refine_v2.py`), and the cross-validation report
(`detector_zoo/crossvalidation_report.md`).

The per-detector raw/intermediate frames (~5 GB of `Map.bin` / `.caked.hdf`) are **not**
committed — they regenerate from MIDAS's bundled CeO₂ (NIST SRM 674b) example data.
`ground_truth.json` documents the four geometries and the two pipeline bugs the
cross-detector grading surfaced (NaN-unsafe residual extraction; a DFT-relaxed starting
cell).

## Real-IOC safety variant

`run_safety_real_ioc.sh` routes the same scenarios through a live EPICS IOC (real
`caget`/`caput`) instead of the mock layer — for a beamline workstation. Requires
`EPICS_MOTOR_PREFIX` and the Channel Access tools on `PATH`.
