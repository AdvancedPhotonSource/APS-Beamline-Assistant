---
name: midas-integrate
description: Integrate 2D diffraction images to 1D patterns using MIDAS integrator.py. Use when the user asks to integrate, cake, produce a lineout, convert 2D to 1D, run azimuthal integration, or process diffraction frames for phase identification or GSAS-II.
compatibility: Requires MIDAS v10 installation, midas_env conda environment, calibrated params file from midas-calibrate skill
metadata:
  author: pawan-tripathi
  version: "1.0"
  midas-version: "10.0"
  manual: MIDAS/manuals/FF_Radial_Integration.md
---

## Integration Workflow (MIDAS v10)

Two tools depending on scope. Both use `integrator.py` which calls `IntegratorZarrOMP`.

> **The old `Integrator` C binary is gone in v10. Never use it.**

---

### Single image → `midas_integrate_2d_to_1d`

Use for one image (check calibration quality, quick look at a single frame).

**Prerequisites:** `refined_MIDAS_params*.txt` from `midas_auto_calibrate`.

```
midas_integrate_2d_to_1d(
    image_file    = "<absolute path to image>",
    param_file    = "<refined_MIDAS_params*.txt>",
    dark_file     = "<dark image if available, else omit>",
    result_folder = "<output dir>",   # default: <image_dir>/integration
    n_cpus        = 4,
    convert_files = True
)
```

---

### Batch (scan / multiple frames) → `midas_batch_integrate`

Use for full scan processing.

```
midas_batch_integrate(
    data_file      = "<path to data file or first in series>",
    dark_file      = "<dark file>",
    parameter_file = "<refined_MIDAS_params*.txt>",
    start_frame    = <int>,
    end_frame      = <int>,
    result_folder  = "./integration_results",
    num_cpus       = 10,
    map_detector   = True,
    convert_files  = True
)
```

---

### v10 executable chain

```
integrator.py
  ├─ DetectorMapper      → Map.bin, nMap.bin, maskMap.bin  (first run only)
  └─ IntegratorZarrOMP   → lineout + caked zarr per file
```

`map_detector=True` / `-mapDetector 1` triggers `DetectorMapper` automatically. No need to run it manually.

### Output files

| File | Use |
|---|---|
| `*_lineout.xy` | 2θ vs intensity (text) — load in GSAS-II, plot directly |
| `*_lineout.bin` | Binary lineout (backward compatible) |
| `*_caked.hdf.zarr.zip` | Full caked data — GSAS-II compatible |
| `Map.bin`, `nMap.bin`, `maskMap.bin` | Geometry maps (generated once per detector/params) |

### Supported input formats

`.tif/.tiff`, `.ge/.ge1-.ge5`, `.h5/.hdf5/.nxs`, `.zip` (Zarr)

### Environment

- Uses `find_midas_python()` → `midas_env` conda Python
- `get_midas_env()` sets `PYTHONPATH=MIDAS/utils` so `midas_config` imports correctly
- Never use system `python3` directly
