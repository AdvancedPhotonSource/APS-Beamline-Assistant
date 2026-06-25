---
name: midas-hedm
description: Run full HEDM (High Energy Diffraction Microscopy) analysis pipelines using MIDAS v11. Use when the user asks for FF-HEDM, NF-HEDM, PF-HEDM, grain indexing, grain matching, strain mapping, microstructure reconstruction, or orientation mapping.
compatibility: Requires MIDAS v11 + midas-suite (midas-pipeline ≥0.4.9, midas-nf-pipeline ≥0.1.1). Install via pip install midas-suite. Calibrated detector parameters required.
metadata:
  author: pawan-tripathi
  version: "2.0"
  midas-version: "11.0"
  package: "midas-pipeline"
---

## HEDM Analysis Workflows (MIDAS v11)

In v11 every HEDM pipeline runs through a **pip-package CLI**, not the old
`*_MIDAS.py` driver scripts. FF and PF are the same engine
(`midas-pipeline run --scan-mode ff|pf`); NF has its own pure-Python orchestrator.

### Standard pipeline order

| Step | APEXA tool | v11 engine |
|---|---|---|
| 0. Validate | `validate_parameter_file`, `inspect_dataset_file` | `midas-params` ([[midas-validate]]) |
| 1. Calibration | `midas_auto_calibrate` | native `midas_calibrate` ([[midas-calibrate]]) |
| 2. Integration | `midas_integrate_2d_to_1d`, `midas_batch_integrate` | native `midas_integrate` ([[midas-integrate]]) |
| 3. FF-HEDM | `run_ff_hedm_full_workflow` | `midas-pipeline run --scan-mode ff` ([[midas-ff-hedm]]) |
| 4. NF-HEDM | `run_nf_hedm_reconstruction` | `midas-nf-pipeline run` |
| 5. PF-HEDM | `run_pf_hedm_workflow` | `midas-pipeline run --scan-mode pf` |
| 6. Grain matching | `match_grains` | `match_grains.py` (Hungarian) |
| 7. FF↔NF overlay | `overlay_ff_nf_results` | `PlotFFNF.py` |
| 8. Stress/strain | `compute_grain_stress`, `correct_d0_equilibrium` | native `midas_stress` |

> There is **no `run_ff_grain_tracking` tool**. For grain-to-grain comparison
> across states/layers use `match_grains` (Hungarian assignment) +
> `calculate_misorientation`.

For deep FF detail see [[midas-ff-hedm]] (the canonical FF skill) and
[[midas-ffpipeline]] (advanced solver/resume/sharding controls).

---

### FF-HEDM — `run_ff_hedm_full_workflow`

```
run_ff_hedm_full_workflow(
    result_folder   = "<output dir>",
    param_file      = "<refined_MIDAS_params*.txt>",
    data_file       = "<.analysis.MIDAS.zip zarr archive>",
    n_cpus          = 16,
    start_layer     = 1,
    end_layer       = 1,
    indexer_backend = "python",   # "python" (in-process, GPU-ok) or "c-omp" (fast CPU)
    refine_backend  = "python",   # "c-omp" → writes Grains.csv
    detectors       = None,       # detectors.json for multi-detector/Hydra
    resume_from     = None,       # stage name to restart from
)
```

- **`device`** is selected by the pipeline; pass GPU via the lower-level
  [[midas-ffpipeline]] tool (`run_ff_pipeline(device="cuda", shard_gpus="auto")`).
- **Grains.csv only with `refine_backend="c-omp"`** — the python refiner writes
  `IndexBest.bin` / `OrientPosFit.bin`; read those for grain counts (see [[midas-ff-hedm]]).

---

### NF-HEDM — `run_nf_hedm_reconstruction`

Pure-Python `midas-nf-pipeline` (no compiled binaries needed).

```
run_nf_hedm_reconstruction(
    result_folder       = "<output dir>",
    param_file          = "<NF Parameters.txt>",
    n_cpus              = 4,
    device              = "auto",     # auto | cpu | cuda
    ff_seed_orientations= False,      # seed Loop 0 from FF Grains.csv
    do_image_processing = True,       # ProcessImagesCombined
    start_layer         = 1,
    end_layer           = 1,
    min_confidence      = 0.6,        # Mic2GrainsList threshold
    resume_from         = None,       # stage label, e.g. "loop_1_seeded"
)
```

Stages: `image_processing → spot_search → seed_generation → loop_0_unseeded
[→ loop_N_seeded …] → parse_mic → mic2grains → consolidate`. Produces
`LayerNr_<N>/Grains.mic`, `GrainsLayer*.csv`, and a consolidated HDF5.

> Grain centroids/orientations come out of the `mic2grains` stage automatically.
> `extract_grain_centroids` (now `midas-nf-pipeline mic2grains`) is only needed to
> re-run that step on an existing `.mic`.

---

### PF-HEDM (scanning / point-focus) — `run_pf_hedm_workflow`

```
run_pf_hedm_workflow(
    result_folder   = "<output dir>",
    param_file      = "<params.txt>",
    data_file       = "<PF scan zarr>",
    n_scans         = <int>,          # number of scan positions
    scan_step_um    = <float>,
    beam_size_um    = <float>,
    n_cpus          = 16,
    indexer_backend = "c-omp",
    refine_backend  = "c-omp",
)
```

Adds PF-only stages on top of the FF chain: `merge_scans → seeding → find_grains
→ (voxel_cleanup) → sinogen → reconstruct → fuse → potts → em_refine`, plus the
V-map refinement pass (`calc_radius_v → refine_vmap`, enabled with `--vmap-run`
and the `--vmap-*` / `--soft-*` flags via the CLI). Writes per-layer `Grains.csv`
and a `v_map.h5` voxel orientation/strain map.

---

### v11 key changes vs v10

| v10 | v11 |
|---|---|
| `ff_MIDAS.py` driver | `midas-pipeline run --scan-mode ff` |
| `pf_MIDAS.py` driver | `midas-pipeline run --scan-mode pf` |
| `nf_MIDAS.py` driver | `midas-nf-pipeline run` |
| `CalibrantPanelShiftsOMP` | archived → native `midas_calibrate` / `CalibrantIntegratorOMP` |
| `Integrator` / `IntegratorZarrOMP` only | native `midas_integrate` in-process (OMP fallback) |
| separate C binaries everywhere | 18 pip packages (`midas-suite`); binaries are the fallback |
| `IndexerGPU` requires manual flag | `--device cuda` (GPU build optional; CPU `c-omp` default) |
| `--restartFrom` | `--resume {auto,none,from} --from <stage>` (FF/PF); `--restart-from <stage>` (NF) |
| separate outputs | consolidated HDF5 (`--generate-h5`) |
| `~/opt/MIDAS` | `~/Git/MIDAS` — set `MIDAS_PATH` in `.env` |

---

### Parameter file keys

```
Lsd              <µm>      sample-to-detector distance
BC               <Y> <Z>   beam center (pixels)
Wavelength       <Å>
px               <µm>      pixel size
SpaceGroup       <int>
LatticeParameter <a b c α β γ>
tx ty tz         <rad>     detector tilts
p0..p5                     distortion coefficients
NrPixels / NrPixelsY / NrPixelsZ
BadPxIntensity   -2        APS detector convention
GapIntensity     -1        APS detector convention
```

---

### Expected outputs per step

| Step | Key output files |
|---|---|
| Calibration | `refined_MIDAS_params_CeO2.txt` (+ `calibration.json`, `residual_corr.bin`) |
| Integration | `*_lineout.xy`, `<stem>.zarr.zip` |
| FF-HEDM | `IndexBest.bin`, `OrientPosFit.bin`, `InputAll.csv`; `Grains.csv` (c-omp refiner) |
| NF-HEDM | `Grains.mic`, `GrainsLayer*.csv`, consolidated HDF5 |
| PF-HEDM | per-layer `Grains.csv`, `v_map.h5` |

---

### Visualize results

Via [[midas-visualize]] / `run_midas_viewer`:
- FF grains → `plotGrains3d` (3D grain map → Plotly HTML)
- FF spots → `plotFFSpots3d`, `plotFFSpots3dGrains`
- PF sinograms/intensity → `pfIntensityViewer`
- NF microstructure → `nf_qt`
- FF↔NF overlay → `PlotFFNF`

---

### Manuals (in MIDAS installation)

- `MIDAS/manuals/FF_Calibration.md` — calibration reference
- `MIDAS/manuals/FF_Radial_Integration.md` — integration reference
- `MIDAS/manuals/FF_Analysis.md` — FF-HEDM analysis
- `MIDAS/manuals/NF_Analysis.md` — NF-HEDM reconstruction
- `MIDAS/manuals/PF_Analysis.md` — PF/scanning HEDM
