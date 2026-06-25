---
name: midas-ff-hedm
description: Run the MIDAS v11 FF-HEDM grain reconstruction pipeline via APEXA. Use when the user asks to run far-field HEDM, reconstruct grain orientations, index grains, or analyze diffraction data from a polycrystalline sample. Wraps midas-pipeline run --scan-mode ff.
compatibility: Requires MIDAS v11 + midas-suite (midas-pipeline ≥0.4.9, midas-nf-pipeline ≥0.1.1). Install via pip install midas-suite.
metadata:
  author: pawan-tripathi
  version: "1.1"
  midas-version: "11.0"
  package: "midas-pipeline"
  tool: "run_ff_hedm_full_workflow"
---

## When to use

The APEXA tool is `run_ff_hedm_full_workflow`. It wraps `midas-pipeline run --scan-mode ff`.
Use it for:
- Reconstructing grain orientations from FF-HEDM data
- Indexing a polycrystalline diffraction dataset
- Generating Grains.csv / IndexBest.bin from a MIDAS zarr archive
- Synthetic dataset testing via `midas-pipeline simulate`

**Do NOT use this skill for:** NF-HEDM reconstruction (`run_nf_hedm_reconstruction`), PF-HEDM (`run_pf_hedm_workflow`), or calibration (`midas_auto_calibrate`).

---

## Required inputs

Two inputs are always required:

| Parameter | Description |
|---|---|
| `param_file` | Path to `Parameters.txt` — geometry + scan config |
| `data_file` | Path to `.analysis.MIDAS.zip` zarr archive (single-detector FF scan) |

Optional but common:

| Parameter | Default | Notes |
|---|---|---|
| `result_folder` | `/tmp/ff_hedm_output` | Where LayerNr_N/ directories are created |
| `n_cpus` | 4 | CPU threads for non-GPU stages |
| `layers` | `"1-1"` | Layer range, e.g. `"1-3"` for 3 layers |
| `indexer_backend` | `"python"` | `"python"` (in-process, GPU-compatible) or `"c-omp"` (fast CPU) |
| `refine_backend` | `"python"` | `"python"` (PyTorch) or `"c-omp"` (fast, writes Grains.csv) |
| `detectors` | `None` | Path to `detectors.json` for multi-detector runs |

---

## Quick start

```
run FF-HEDM reconstruction,
  param_file=/path/to/Parameters.txt,
  data_file=/path/to/scan.analysis.MIDAS.zip,
  result_folder=/path/to/output,
  n_cpus=8
```

APEXA will fire the **pre-action reasoning gate** — write SITUATION / GAP / PLAN before calling the tool.

---

## What midas-pipeline does internally

FF-HEDM stages (in order):

```
zip_convert → hkl → peakfit → merge_overlaps → calc_radius → transforms →
cross_det_merge → global_powder → merge_scans → seeding → binning →
indexing → refinement → find_grains → process_grains → consolidation
```

Key stages:
- **peakfit** — detect diffraction spots from zarr frames
- **indexing** — match spots to HKL predictions for each candidate grain orientation
- **refinement** — fit grain position + orientation to matched spots
- **process_grains** — consolidate into Grains.csv (c-omp backend only; see bug note)
- **consolidation** — write consolidated HDF5

Resume from any stage with `--resume auto` or `--resume from --from <stage>`.

---

## Parameters.txt — minimum required keys

```
# Geometry (from calibration)
Lsd              <µm>           sample-to-detector distance
BC               <Y> <Z>        beam center (pixels)
Wavelength       <Å>
px               <µm>           pixel size
tx ty tz         <rad>          detector tilts
p0 p1 p2 p3 p4 p5               distortion coefficients

# Detector
NrPixels         <int>          detector pixels (square) or NrPixelsY / NrPixelsZ
BadPxIntensity   -2             APS convention
GapIntensity     -1             APS convention

# Crystal
SpaceGroup       <int>          e.g. 225 for FCC/Au
LatticeParameter <a b c α β γ>  e.g. 4.08 4.08 4.08 90 90 90

# Scan
OmegaRange       <start stop>   e.g. -180 180
OmegaStep        <deg>          e.g. 0.25
```

**Note:** When providing a zarr archive via `data_file`, you do NOT need `RawFolder`, `FileStem`, or `Ext` — midas-pipeline uses `--skip-validation` automatically. Do NOT call `validate_parameter_file` when a zarr is provided.

---

## Output files

All outputs land in `result_folder/LayerNr_1/`:

| File | Description | Always present? |
|---|---|---|
| `IndexBest.bin` | Per-seed indexing solutions (15 float64 cols, col 14 = nMatches) | ✅ Yes |
| `IndexBestFull.bin` | Matched spot pairs for each solution | ✅ Yes |
| `OrientPosFit.bin` | Refined grain orientations + positions | ✅ Yes |
| `FitBest.bin` | Refinement residuals per grain | ✅ Yes |
| `InputAll.csv` | All detected diffraction spots | ✅ Yes |
| `paramstest.txt` | Geometry used for this run | ✅ Yes |
| `hkls.csv` | HKL list used for indexing | ✅ Yes |
| `Grains.csv` | Consolidated grain list | ⚠️ c-omp refiner only |
| `consolidated_Output.h5` | Full consolidated HDF5 | Optional (`generate_h5=True`) |

---

## Reading IndexBest.bin (grain count without Grains.csv)

```python
import numpy as np
data = np.fromfile("IndexBest.bin", dtype=np.float64).reshape(-1, 15)
n_matches = data[:, 14]           # column 14 = nMatches
solved = data[n_matches > 0]      # seeds that found a grain orientation
n_grains = len(solved)
best_match = int(n_matches.max()) # max rings matched
print(f"{n_grains} grains solved, best nMatches={best_match}")
```

---

## Backend selection

| Combination | Grains.csv? | Speed | Notes |
|---|---|---|---|
| `indexer=python, refiner=python` | ❌ No | Moderate | Default; GPU-compatible; results in IndexBest.bin |
| `indexer=c-omp, refiner=c-omp` | ✅ Yes | Fast CPU | Requires compiled `midas_indexer` + `midas_fitgrain` binaries |
| `indexer=python, refiner=c-omp` | ❌ No | — | Mixed — c-omp refiner cannot read python indexer output format |

**Known bug (midas-process-grains 0.4.6):** python refiner writes `OrientPosFit.bin` to the layer root, but midas-process-grains expects it in `Results/` subdirectory → process_grains + consolidation are auto-skipped when python refiner is used. APEXA reads `IndexBest.bin` directly for grain count.

To get `Grains.csv`, set both `indexer_backend="c-omp"` and `refine_backend="c-omp"`. This requires compiled MIDAS binaries — check with `validate_midas_installation`.

---

## Synthetic dataset (simulate → reconstruct)

Generate a synthetic FF-HEDM dataset to test the pipeline without real data:

```bash
midas-pipeline simulate \
  --out /path/to/sim/ \
  --params Parameters.txt \
  --n-grains 50 \
  --seed 42
```

Outputs:
- `sim/<stem>.analysis.MIDAS.zip` — synthetic zarr archive
- `sim/GrainsSim.csv` — ground-truth grain orientations + positions (for comparison)
- `sim/detectors.json` — detector layout

Then reconstruct:
```
run FF-HEDM reconstruction,
  param_file=/path/to/sim/Parameters.txt,
  data_file=/path/to/sim/<stem>.analysis.MIDAS.zip,
  result_folder=/path/to/output,
  n_cpus=4
```

Expected performance (50 Au grains, python backend, 4 CPUs):
- ~8–12 minutes runtime
- ~230–240 seeds solved out of ~275 (~84–87%)
- Best nMatches ~110–120 rings
- ~45–50 unique grains recovered out of 50 ground truth

---

## Resume an interrupted run

```
run_command("midas-pipeline resume /path/to/output --from indexing --scan-mode ff --params Parameters.txt")
```

Or via the tool with args: the tool auto-detects completed stages and resumes if the result directory exists.

Check status of a previous run:
```
run_command("midas-pipeline status /path/to/output")
```

---

## Multi-detector runs

For multi-panel detectors (e.g. 4-panel Hydra at 1-ID):
```
run_ff_hedm_full_workflow(
    param_file    = "Parameters.txt",
    detectors     = "detectors.json",   # multi-detector layout
    result_folder = "/path/to/output",
    n_cpus        = 16
)
```

Generate a detectors.json from the MIDAS detector calibration output, or auto-create with `midas-pipeline simulate --n-detectors 4`.

---

## Checkpoint / resume stages

Valid stage names for `--from` / `--only` / `--skip`:
```
zip_convert, hkl, peakfit, merge_overlaps, calc_radius, transforms,
cross_det_merge, global_powder, merge_scans, seeding, binning,
indexing, refinement, find_grains, voxel_cleanup, sinogen, reconstruct,
fuse, potts, em_refine, process_grains, grain_geometry, consolidation,
calc_radius_v, refine_vmap
```

---

## Pre-action reasoning gate

Before calling `run_ff_hedm_full_workflow`, ALWAYS write:

```
SITUATION: <what data is available, how many frames, detector type, calibration status>
GAP: <what is missing or uncertain — parameter file? zarr file? calibration done?>
PLAN: <exact tool call you will make and why>
```

Do NOT skip this. The gate is enforced by APEXA's Analysis agent (`use_planning=True`).

---

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: RawFolder` during validation | validate_parameter_file called with zarr input | Skip validate_parameter_file when data_file is a zarr; midas-pipeline uses --skip-validation |
| `Cannot access local variable 're'` on import | Stale .pyc cache from older midas_comprehensive_server.py | `find . -name "*.pyc" -delete && pkill -f midas_comprehensive_server && restart` |
| `process_grains` fails silently | midas-process-grains 0.4.6 bug with python refiner | Expected — read IndexBest.bin instead; or use c-omp backend pair |
| Model hallucinates `~/opt/MIDAS` path | Model uses training-data path not actual MIDAS_PATH | Anti-hallucination guard in run_ff_hedm_full_workflow blocks this; actual path from MIDAS_PATH env var |
| `Grains.csv not found` after run | python refiner used | Normal — use IndexBest.bin for grain count |

---

## Workflow integration

Typical APEXA session for FF-HEDM:

1. **Calibrate** — `midas_auto_calibrate` → produces `refined_MIDAS_params_CeO2.txt`
2. **Integrate** — `midas_integrate_2d_to_1d` → produces `*_lineout.xy`, `*_caked.hdf.zarr.zip`
3. **Reconstruct** — `run_ff_hedm_full_workflow` → produces `IndexBest.bin`, `OrientPosFit.bin`
4. **Compare** — `read_grains_summary` → compare to `GrainsSim.csv` ground truth
5. **Visualize** — `run_midas_viewer(viewer="plotGrains3d", data_file="<result_dir>")` (3D grain map → HTML), or `plotFFSpots3dGrains` for spots-by-grain. See [[midas-visualize]].

---

## MIDAS v11 key additions (vs v10)

| Feature | Command |
|---|---|
| GPU indexing | `--device cuda` |
| Checkpoint resume | `--resume auto` |
| Python differentiable refiner | `--refine-backend python` |
| Consolidated HDF5 | `--generate-h5` |
| Synthetic forward model | `midas-pipeline simulate` |
| V-map (volumetric) | `--vmap-run` |
| NF→FF seeded indexing | `--nf-result-dir <nf_output>` |
| Multi-GPU sharding | `--shard-gpus auto` |
