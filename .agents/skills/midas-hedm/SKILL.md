---
name: midas-hedm
description: Run full HEDM (High Energy Diffraction Microscopy) analysis pipelines using MIDAS v10. Use when the user asks for FF-HEDM, NF-HEDM, PF-HEDM, grain indexing, grain tracking, strain mapping, microstructure reconstruction, or orientation mapping.
compatibility: Requires MIDAS v10 installation, midas_env conda environment, calibrated detector parameters
metadata:
  author: pawan-tripathi
  version: "1.0"
  midas-version: "10.0"
---

## HEDM Analysis Workflows (MIDAS v10)

### Standard pipeline order

```
1. Calibration      midas_auto_calibrate          AutoCalibrateZarr.py
2. Integration      midas_integrate_2d_to_1d      integrator.py (single)
                    midas_batch_integrate          integrator.py (batch)
3. FF-HEDM          run_ff_hedm_full_workflow      ff_MIDAS.py
4. NF-HEDM          run_nf_hedm_reconstruction     nf_MIDAS.py
5. PF-HEDM          run_pf_hedm_workflow           pf_MIDAS.py
6. Overlay          overlay_ff_nf_results
7. Grain tracking   run_ff_grain_tracking
```

---

### FF-HEDM — `run_ff_hedm_full_workflow`

```
run_ff_hedm_full_workflow(
    result_folder   = "<output dir>",
    param_file      = "<refined_MIDAS_params*.txt>",
    data_file       = "<scan HDF5 or first GE file>",
    n_cpus          = 32,
    start_layer     = 1,
    end_layer       = 1,
    use_gpu         = False,     # True → IndexerGPU + FitPosOrStrainsGPU
    resume_file     = None,      # path to checkpoint HDF5 to resume
    restart_from    = None       # layer name/number to restart from
)
```

**v10 GPU executables** (set `use_gpu=True`):
- `IndexerGPU` — grain indexing on CUDA GPU
- `FitPosOrStrainsGPU` — position/strain refinement on GPU
- `IndexerScanningGPU` — scanning HEDM variant

---

### NF-HEDM — `run_nf_hedm_reconstruction`

```
run_nf_hedm_reconstruction(
    result_folder = "<output dir>",
    param_file    = "<params.txt>",
    data_file     = "<NF scan data>",
    use_gpu       = False,   # True → FitOrientationGPU (-gpuFit flag)
    resume_file   = None,
    restart_from  = None
)
```

---

### PF-HEDM — `run_pf_hedm_workflow`

```
run_pf_hedm_workflow(
    result_folder = "<output dir>",
    param_file    = "<params.txt>",
    data_file     = "<PF scan data>",
    use_gpu       = False,
    do_tomo       = False    # True → run tomographic reconstruction step
)
```

---

### v10 key changes vs older MIDAS

| Old | v10 |
|---|---|
| `Integrator` binary | `IntegratorZarrOMP` (via `integrator.py`) |
| `CalibrantOMP` | `CalibrantPanelShiftsOMP` (via `AutoCalibrateZarr.py`) |
| `-StoppingStrain` | removed — use `--n-iterations` |
| `Indexer` only | `Indexer` (CPU) + `IndexerGPU` (v10) |
| No resume | `--resume`, `--restartFrom` flags |
| Separate outputs | Consolidated HDF5 output |
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
| Calibration | `refined_MIDAS_params_CeO2.txt` |
| Integration | `*_lineout.xy`, `*_caked.hdf.zarr.zip` |
| FF-HEDM | `*.HDF` (grain data), `grains.csv`, `Grains.csv` |
| NF-HEDM | `*.mic` (microstructure map), `nf_*.HDF` |
| PF-HEDM | `*.HDF`, pole figure data |

---

### Manuals (in MIDAS installation)

- `MIDAS/manuals/FF_Calibration.md` — calibration full reference
- `MIDAS/manuals/FF_Radial_Integration.md` — integration full reference
- `MIDAS/manuals/FF_Analysis.md` — FF-HEDM analysis
- `MIDAS/manuals/NF_Analysis.md` — NF-HEDM reconstruction
- `MIDAS/manuals/PF_Analysis.md` — PF/scanning HEDM
