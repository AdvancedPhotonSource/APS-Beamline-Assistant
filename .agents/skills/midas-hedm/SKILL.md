---
name: midas-hedm
description: Run full HEDM (High Energy Diffraction Microscopy) analysis pipelines using MIDAS v11. Use when the user asks for FF-HEDM, NF-HEDM, PF-HEDM, grain indexing, grain matching, strain mapping, microstructure reconstruction, or orientation mapping.
compatibility: Requires MIDAS v11 + midas-suite ≥0.4.0 (midas-pipeline ≥0.6.1, midas-nf-pipeline ≥0.1.1). Install via pip install midas-suite. Calibrated detector parameters required.
metadata:
  author: pawan-tripathi
  version: "2.1"
  midas-version: "11.0"
  package: "midas-pipeline"
  verified-against: "midas-pipeline 0.6.1 / midas-nf-pipeline 0.1.1"
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
    process_grains  = True,       # run process_grains+consolidation (needs c-omp)
)
```

- **`device`** is selected by the pipeline; pass GPU via the lower-level
  [[midas-ffpipeline]] tool (`run_ff_pipeline(device="cuda", shard_gpus="auto")`).
- **Grains.csv only with `refine_backend="c-omp"`** — the python refiner writes
  `IndexBest.bin` / `OrientPosFit.bin`; read those for grain counts (see [[midas-ff-hedm]]).
  With `c-omp` and `process_grains=True` (default) the run also writes
  `processgrains_diagnostics.h5` and sets `report_ready`.
- **Before the run (FF §6b):** `calibrate_ring_thresholds(zarr_file=<...>.MIDAS.zip)`
  → data-derived `RingThresh` (never copy from a template). Read
  `diagnose_parameter_file` → `handbook_traps` for the silent-corruption checklist.
- **After the run (FF §9):** `generate_ff_reconstruction_report(run_dir=<...>/LayerNr_1)`
  → self-contained `report.html` (needs `Grains.csv`).

---

### NF-HEDM — `run_nf_hedm_reconstruction`

Wraps `midas-nf-pipeline run` (pure-Python; verified against midas-nf-pipeline
**0.1.1**). NF is its **own** package — there is **no `--scan-mode`** (that's
FF/PF). Unlike FF, NF has **no `simulate` subcommand**: it needs **real raw
images** in `DataDirectory` for the image-processing stage.

```
run_nf_hedm_reconstruction(
    result_folder       = "<output dir>",  # per-layer outputs → LayerNr_<N>/
    param_file          = "<NF param file>",
    n_cpus              = 4,
    device              = "auto",     # auto | cpu | cuda  (auto→cuda if present)
    ff_seed_orientations= False,      # seed Loop 0 from FF Grains.csv (else cache)
    do_image_processing = True,       # ProcessImagesCombined (False → SpotsInfo.bin must exist)
    start_layer         = 1,
    end_layer           = 1,
    min_confidence      = 0.6,        # last-loop Mic2GrainsList threshold
    resume_from         = "",         # stage label, e.g. "loop_1_seeded"
    dtype               = "auto",     # auto | fp32/float32 | fp64/float64
    refine              = "",         # "" (auto) | nm-triton | nm-batched | nm-serial | lbfgs+nm | lbfgs
    skip_validation     = False,      # skip midas-params preflight
    install_dir         = "",         # MIDAS install for seedOrientations cache (defaults to MIDAS_PATH)
)
```

**Single vs multi-resolution** is driven by the param file, not a flag:
single-resolution = `NumLoops=0` (no `GridRefactor` key); multi-resolution =
a `GridRefactor (starting_grid, scaling_factor, num_loops)` triplet.

**Real stage order** (one `run` covers both):
preprocess (HKL list → seed orientations [FF `Grains.csv` or packaged cache] →
hex grid → tomo filter → grid mask → diffraction spots) → **image_processing**
(ProcessImagesCombined, one pass per detector distance) → **fitting**
(`nm-triton` on CUDA if Triton present, else `nm-batched`) → **parse_mic** →
**consolidate** [→ `mic2grains` for multi-layer]. Multi-resolution repeats
`loop_<k>_seeded` → bad-voxel filter → `loop_<k>_unseeded` → binary merge.

**Outputs** (not `Grains.mic` — that name is never written):
- `<MicFileText>.mic` — the voxel orientation map (name from the param's `MicFileText`).
- `<stem>_consolidated.h5` — consolidated map + provenance ledger (also the
  `--resume` state file). Voxels at `/voxels/position` (single-res) or
  `/multi_resolution/<label>/voxels/position`; grains at `/grains/`. The tool
  reads voxel/grain **counts from this H5** and returns `total_voxels` /
  `total_grains`.
- `GrainsLayer<N>.csv` — per-layer grain list (from `mic2grains`).

**Real param keys** (see `NF_HEDM/Example/ps_au.txt`): `DataDirectory`,
`nDistances`, `Lsd` (one line per distance), `BC` (2 values per distance),
`tx/ty/tz`, `px`, `NrPixels`, `MaxRingRad`, `OmegaStart`, `OmegaStep`,
`StartNr`, `EndNr`, `NrFilesPerDistance`, `LatticeParameter` (6 values),
`Wavelength`, `SpaceGroup`, `Rsample`, `GridSize`, `MinConfidence`,
`NrOrientations`, `SaveNSolutions`, `MicFileBinary`, `MicFileText`.

> **Inline-comment pitfall (auto-handled).** The run-path reader keeps only the
> first token so `GridSize 2.5 # microns` parses, **but the final consolidation
> stage joins all trailing tokens and calls `float()` — so a commented param
> file crashes the run at the very last stage, after all the compute.** The
> APEXA tool auto-strips trailing `# …` comments into a `*.apexa_clean.<ext>`
> copy (full-line `#` comments are preserved) and reports it under
> `param_sanitized`. The bundled `ps_au.txt` example needs this.

**Geometry refinement before reconstruction** (NF Handbook §7) —
`refine_nf_parameters(param_file, multi_point=<bool>, objective="hard")` wraps
`midas-nf-pipeline refine-params`. Refines Lsd/BC/tilts/Wedge on a calibrant;
`multi_point=True` uses the param file's `GridPoints` (objective `hard` =
FracOverlap/C-parity, recommended; `soft` = L-BFGS). Paste the refined geometry
back into the param file, then run `run_nf_hedm_reconstruction`. **Do NOT set
`SkipFrame 1` on an NF param file** — the GE far-field rule does not apply to NF
(NF §3g); `diagnose_parameter_file(pipeline="nf")` flags it under `handbook_traps`.

**Offline subcommands** (pure-Python, fast — no images needed) for iterating on
an existing `.mic`, callable via `run_command`:
- `midas-nf-pipeline consolidate <mic> --paramFN <p>` → `<stem>_consolidated.h5`
- `midas-nf-pipeline mic2grains <p> <mic> <out.csv> [doNeighbor] [nCPUs] [minConf]`
- `midas-nf-pipeline parse-mic <p>` (rebuild the text `.mic` from binary)

> Grain centroids/orientations come out of the `mic2grains` stage automatically.
> `extract_grain_centroids` (= `midas-nf-pipeline mic2grains`) is only needed to
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
| Calibration | `refined_MIDAS_params_CeO2.txt` (+ `autocal.log`, `*corr.csv`) |
| Integration | `*_lineout.xy`, `<stem>.zarr.zip` |
| FF-HEDM | `IndexBest.bin`, `OrientPosFit.bin`, `InputAll.csv`; `Grains.csv` + `processgrains_diagnostics.h5` (c-omp refiner); `report.html` via `generate_ff_reconstruction_report` |
| NF-HEDM | `<MicFileText>.mic`, `<stem>_consolidated.h5`, `GrainsLayer*.csv` |
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
