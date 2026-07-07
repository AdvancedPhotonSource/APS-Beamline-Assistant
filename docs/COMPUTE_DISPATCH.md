# APEXA Compute Dispatch — standard practice

How APEXA should place data-reduction work on the right compute: **local CPU**,
**local GPU**, or a **remote GPU endpoint inside ANL**. The goal is that small /
interactive tasks stay fast and local, and large batch/HEDM jobs stop grinding on a
CPU-only beamline host.

## The problem this solves
Beamline analysis hosts are often **CPU-only**. APEXA's native PyTorch engines
auto-use a GPU but *refuse to run on CPU* past a pixel budget and fall back to the
CPU OpenMP path (`integrator.py` → `IntegratorZarrOMP`), one 2880² frame at a time.
An overnight scan (hundreds–thousands of frames) then takes hours because nothing
is on a GPU and nothing fans out to a cluster.

## Compute tiers
| Tier | Where | Engines | Use for |
|---|---|---|---|
| **0 — local CPU** | the APEXA host | `integrator.py`/`IntegratorZarrOMP`, `IndexerOMP`, native PyTorch on CPU (small only) | single-frame integrate/calibrate, viewers, validation, knowledge |
| **1 — local GPU** | same host, CUDA/MPS present | native `midas_integrate`/`midas_calibrate`, `midas-integrate-v2`, `IntegratorFitPeaksGPUStream`, `IndexerGPU`/`FitPosOrStrainsGPU` | batch integration, FF/PF/NF index+refine when a local GPU exists |
| **2 — remote GPU endpoint** | ANL GPU node / ALCF Polaris | `midas-pipeline --machine … --device cuda --shard-gpus auto` (parsl), or the same tool run on the GPU host | large batch/HEDM on a CPU-only host |

## Decision rule (what `_pick_compute_target()` implements)
`cost = n_frames × megapixels` (2880² ≈ 8.3 MP).
1. `prefer` overrides: `cpu` / `cuda` / `mps` / `local` / `remote` force the choice.
2. Local accelerator present → **Tier 1** (local GPU).
3. Else `cost ≤ APEXA_COST_LOCAL_CPU` (default 40 ≈ 5 frames) → **Tier 0** (local CPU).
4. Else (large job, CPU-only host):
   - a GPU endpoint is configured (`APEXA_GPU_MACHINE` or `APEXA_GPU_ENDPOINT`) → **Tier 2** (recommend/offload).
   - none configured → Tier 0 **with a "this will be slow" warning**.

## Configuration (env)
| Var | Meaning |
|---|---|
| `APEXA_GPU_MACHINE` | parsl machine name for `midas-pipeline --machine` (e.g. `polaris`) |
| `APEXA_GPU_ENDPOINT` | ssh `user@host` or a Globus-Compute endpoint id for a GPU node |
| `APEXA_GPU_AUTO_DISPATCH` | if set, `midas_integrate_series` returns a dispatch recommendation instead of running a large CPU job |
| `APEXA_COST_LOCAL_CPU` | cost ceiling for staying local on a CPU-only host (default 40) |

## What the tools now expose
- **FF / PF workflows** (`run_ff_hedm_full_workflow`, `run_pf_hedm_workflow`):
  `machine`, `n_nodes`, `shard_gpus` → forwarded to `midas-pipeline`
  (`--machine/--n-nodes/--shard-gpus`). This is the ANL-GPU path (parsl → Polaris
  or a Slurm cluster from `midas_parsl_configs`).
- **Series integration** (`midas_integrate_series`): `compute_target`
  (`auto|local|remote|cpu|cuda|mps`). Every run reports a `compute` block
  (target + reason + cost). `compute_target="remote"` (or `APEXA_GPU_AUTO_DISPATCH`)
  returns a dispatch recommendation for a large batch instead of grinding on CPU.

## The binding constraint: data locality
Put the GPU where it can already see the data. Two ANL options:
1. **A lab GPU node that mounts the beamline filesystem** (e.g. `/scratch/s1iduser`)
   — pragmatic, low-latency, no queue, **no data movement**. Register it as
   `APEXA_GPU_ENDPOINT` (ssh) or a Globus-Compute endpoint. **Best for interactive
   beamtime.**
2. **ALCF Polaris** via the shipped `midas_parsl_configs` `polarisConfig`
   (`--machine polaris`). Best for very large sweeps, but Polaris cannot mount
   `/scratch`, so **stage data via Globus** first.

## Standard practice (summary)
- Single frame / calibration / QC → **stay local** (Tier 0/1), never prompt.
- Batch series / FF-PF-NF on a **GPU host** → run local GPU (Tier 1),
  `device=cuda`, `--shard-gpus auto`.
- Batch series / FF-PF-NF on a **CPU-only host** → **Tier 2**: offload to the GPU
  endpoint that sees the data; only fall back to CPU (with a warning) if no
  endpoint is configured.
- Always emit the `compute` decision in the result so the run is auditable, and
  keep the outcome-manifest + verify-before-report discipline for remote runs.
