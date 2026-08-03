---
name: midas-ffpipeline
description: Run the MIDAS v11 differentiable PyTorch FF-HEDM pipeline (drop-in replacement for ff_MIDAS.py) with checkpoint/resume, multi-GPU sharding, and swappable solvers. Use when the user asks to run FF-HEDM via the new Python pipeline, resume an interrupted run, reprocess with a different solver, or shard across GPUs.
compatibility: Requires MIDAS v11 + the midas_ff_pipeline package (packages/midas_ff_pipeline). PyTorch with CUDA recommended; CPU/MPS supported.
metadata:
  author: pawan-tripathi
  version: "1.2"
  midas-version: "11.0"
  package: "midas_ff_pipeline"
  status: "deprecated (0.4.0; removed in 1.0.0) — use midas-ff-hedm"
---

> ## ⚠ DEPRECATED as of 0.4.0 — removed in 1.0.0: `midas-ff-pipeline` → `midas-pipeline`
> The standalone `midas-ff-pipeline` CLI now prints a `DeprecationWarning` on every
> invocation: *"deprecated as of 0.4.0 and will be removed in 1.0.0. Use
> `midas-pipeline run --scan-mode ff` (CLI) or `midas_pipeline.Pipeline(...)` (API)
> instead — the FF path is the same code under the hood."* All FF-HEDM orchestration
> has moved to **`midas-pipeline run --scan-mode ff`** (see
> `packages/MIDAS_FF_PIPELINE_DEPRECATION_PLAN.md`). Because it is literally the same
> code, its stage names, backends, and outputs are identical to [[midas-ff-hedm]] —
> refer there for the canonical stage list and performance numbers.
>
> **Default to [[midas-ff-hedm]] (`run_ff_hedm_full_workflow`)** — it wraps
> `midas-pipeline` and is the consolidated path. Use **this** skill's
> `run_ff_pipeline` (which still wraps `midas-ff-pipeline run`) only for the
> advanced controls not yet surfaced on `run_ff_hedm_full_workflow`: explicit
> solver/loss selection, multi-GPU sharding, NF-seeded cross-checks, and the
> `status`/`reprocess`/`inspect`/`simulate` companion subcommands. As those land
> in `midas-pipeline`, prefer [[midas-ff-hedm]].

## When to use this vs. `run_ff_hedm_full_workflow`

| Use **`run_ff_pipeline`** (this skill) when... | Use **`run_ff_hedm_full_workflow`** ([[midas-ff-hedm]]) when... |
|---|---|
| Sharding across multiple GPUs (`--shard-gpus`) | Standard FF reconstruction (the default) |
| Trying alternative solvers (LM, Adam, Nelder-Mead, batched LM) | Standard L-BFGS / c-omp refinement is fine |
| Trying alternative losses (angular / internal_angle) | Pixel residual is fine |
| Mixing FF-HEDM with NF cross-checks (`--nf-result-dir`) | FF-only, no NF companion data |
| Single-block re-refinement via `refine_grain_lattice` | Whole-pipeline run |

> **The MCP tool is `run_ff_pipeline`.** It wraps `midas-ff-pipeline run` (the
> package being consolidated into `midas-pipeline`). Both tools support
> `--resume auto/from`; resume alone is not a reason to pick one over the other.

---

## Quick start

```python
run_ff_pipeline(
    params       = "Parameters.txt",
    result       = "/data/runs/sample_001",
    layers       = "1-1",
    n_cpus       = 16,
    device       = "cuda",
    solver       = "lbfgs",
    loss         = "pixel",
    generate_h5  = True,
)
```

Outputs land under `result/LayerNr_<N>/` with consolidated `.h5` when `generate_h5=True`.

---

## Resume a previous run

```python
run_ff_pipeline(
    params       = "Parameters.txt",
    result       = "/data/runs/sample_001",   # same dir as before
    resume       = "auto",                    # or "from"
    resume_from  = "indexing",                # required when resume="from"
)
```

Stages (in order, used by `--from`, `--only`, `--skip`) — these are the **real**
`midas-pipeline` stage names (the deprecated CLI runs the same code, so it accepts
the same names, not the older `prepare/convert/...` labels):
`zip_convert → hkl → peakfit → transforms → cross_det_merge → global_powder → binning → indexing → refinement → process_grains → consolidation`
(full valid enum incl. multi-scan/V-map stages: `merge_overlaps, calc_radius,
merge_scans, seeding, find_grains, voxel_cleanup, sinogen, reconstruct, fuse, potts,
em_refine, grain_geometry, calc_radius_v, refine_vmap`). See [[midas-ff-hedm]].

Run a single stage:
```python
run_ff_pipeline(params=..., result=..., only=["refine"])
```

Skip a stage:
```python
run_ff_pipeline(params=..., result=..., skip=["integrate"])
```

---

## Multi-GPU sharding

```python
run_ff_pipeline(
    ...,
    device       = "cuda",
    shard_gpus   = "auto",          # or "0,1,2,3"
    group_size   = "auto",          # per-grain batch size
    pg_mode      = "spot_aware",    # peak-group mode
    batch        = True,
)
```

`shard_gpus="auto"` partitions grain seeds across all visible CUDA devices.

---

## Solver / loss matrix

| `solver` | `loss` | When |
|---|---|---|
| `lbfgs` | `pixel` | C-binary parity; default |
| `lm_batched` | `pixel` | Fastest GPU path for many small grains |
| `lm` | `pixel` | Single-grain LM (debug / few seeds) |
| `nelder_mead` | `angular` | Derivative-free; useful when grains are noisy |
| `adam` | `internal_angle` | Curriculum-style refinement experiments |

`mode` is usually left empty (auto from params); set `"iterative"` or `"all_at_once"` to override.

---

## NF cross-check

If a companion NF reconstruction has been completed:
```python
run_ff_pipeline(
    ...,
    nf_result_dir = "/data/runs/sample_001_nf",
    grains_file   = "/data/runs/sample_001_nf/Grains.csv",
)
```
This skips FF indexing and refines against NF-seeded grain orientations.

---

## CLI flags reference (forwarded by the wrapper)

| `run_ff_pipeline` kwarg | CLI flag | Notes |
|---|---|---|
| `params` | `--params` | **required** Parameters.txt |
| `result` | `--result` | **required** output dir |
| `zarr` | `--zarr` | Override zarr.zip path |
| `detectors` | `--detectors` | e.g. `"1"` or `"1,2,3,4"` |
| `layers` | `--layers` | e.g. `"1-1"` or `"1-3"` |
| `n_cpus` | `--n-cpus` | non-GPU stage threads |
| `device` | `--device` | `cuda` / `cpu` / `mps` |
| `dtype` | `--dtype` | `auto` / `float32` / `float64` |
| `resume` | `--resume` | `none` / `auto` / `from` |
| `resume_from` | `--from` | required when `resume="from"` |
| `only` | `--only` | repeatable allow-list |
| `skip` | `--skip` | repeatable skip-list |
| `solver` | `--solver` | see matrix above |
| `loss` | `--loss` | `pixel` / `angular` / `internal_angle` |
| `mode` | `--mode` | usually empty |
| `group_size` | `--group-size` | per-grain batch size |
| `shard_gpus` | `--shard-gpus` | `auto` or comma list |
| `pg_mode` | `--pg-mode` | peak-group mode |
| `raw_dir` | `--raw-dir` | when not in params |
| `grains_file` | `--grains-file` | skip indexing |
| `nf_result_dir` | `--nf-result-dir` | NF cross-check companion |
| `batch` | `--batch` | enable batched execution |
| `generate_h5` | `--generate-h5` | consolidated HDF5 |
| `extra_args` | (forwarded) | escape hatch for any other flag |

---

## Companion subcommands (call via `run_command`)

The `midas-ff-pipeline` CLI also exposes these — not yet wrapped as MCP tools:

```
midas-ff-pipeline status   <result_dir>            # which stages have completed
midas-ff-pipeline resume   <result_dir> --from <stage> ...
midas-ff-pipeline reprocess <result_dir>           # re-run with different solver/dtype
midas-ff-pipeline inspect  <layer_dir>             # checkpoint / file listing
midas-ff-pipeline simulate --out ... --params ... --n-grains N   # synthetic data
```

For routine status checks, prefer `run_command(command="midas-ff-pipeline status <dir> --json")`.

---

## Companion grain-fit refiner

Single-grain or block-wise refinement (mirrors the C `FitPosOrStrainsOMP` argv):
```python
refine_grain_lattice(
    param_file = "paramstest.txt",
    block_nr   = 0,
    num_blocks = 1,
    solver     = "lbfgs",
    loss       = "pixel",
    csv        = True,           # also write FitBest.csv
)
```
Use this *after* `run_ff_pipeline` when iterating on solver / loss / device for a single block — it's the same refiner under the hood without re-running indexing.

---

## NF preprocessing companion

For NF-HEDM data prep before reconstruction:
```python
preprocess_nf_data(subcommand="hex-grid",
                   args=["--paramFN", "params.txt", "--output", "grid.bin"])
```
Subcommands: `hex-grid`, `tomo-filter`, `diffr-spots`, `process-images`, `seed-orientations`.
Each has its own flag set — call with `args=["--help"]` to discover them.
