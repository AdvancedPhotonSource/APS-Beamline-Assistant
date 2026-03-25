---
name: midas-integrate
description: Integrate 2D diffraction images to 1D patterns using MIDAS integrator.py. Use when the user asks to integrate, cake, produce a lineout, convert 2D to 1D, run azimuthal integration, or process diffraction frames for phase identification or GSAS-II.
compatibility: Requires MIDAS v10 (FF_HEDM/workflows/integrator.py), midas_env conda environment with zarr/numpy/scipy, calibrated params file from midas_auto_calibrate
metadata:
  author: pawan-tripathi
  version: "2.0"
  midas-version: "10.0"
  manual: MIDAS/manuals/FF_Radial_Integration.md
---

## Integration Workflow (MIDAS v10)

Two tools depending on scope. Both use `FF_HEDM/workflows/integrator.py` which calls `IntegratorZarrOMP`.

> **The old `Integrator` C binary is gone in v10 — never use it.**

---

### Single image → `midas_integrate_2d_to_1d`

Use for one image. `calibration_file` is **optional** — auto-detected from the image directory if omitted.

```
midas_integrate_2d_to_1d(
    image_file       = "<absolute path to image>",
    calibration_file = "<refined_MIDAS_params*.txt>",  # optional, auto-found
    dark_file        = "<dark image if available>",
    result_folder    = "<output dir>",                  # default: <image_dir>/integration
    n_cpus           = 4,
    convert_files    = True
)
```

**The `refined_MIDAS_params*.txt` from calibration already contains everything needed:**
`Lsd`, `BC`, `Wavelength`, `px`, `NrPixelsY/Z`, `RMin/Max/BinSize`, `EtaMin/Max/BinSize`, `FileStem`, `Folder`, `DataType`

---

### Batch (multiple frames/files) → `midas_batch_integrate`

```
midas_batch_integrate(
    data_file      = "<path to data HDF5 or first image>",
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

### How integrator.py uses the files (important)

- `-paramFN` and `-dataFN` are **both required** by integrator.py
- `-dataFN` must contain a **6-digit zero-padded number** (e.g. `CeO2_000001.tif` or `scan_004018.tif`)
- File number is extracted with regex `\d{6}` — used to determine which frame(s) to process
- If the calibration symlink `CeO2_000001.tif` exists in the image dir, use that as `-dataFN`
- `integrator.py` self-patches its own PYTHONPATH — no external PYTHONPATH needed
- Script location: `MIDAS/FF_HEDM/workflows/integrator.py` (NOT `utils/`)

### v10 integrator.py CLI flags

| Flag | Default | Notes |
|---|---|---|
| `-paramFN` | required | Parameter file path |
| `-dataFN` | required | First data file (6-digit number in name) |
| `-resultFolder` | `.` | Output directory |
| `-darkFN` | `''` | Dark field image |
| `-nCPUs` | `1` | Parallel files |
| `-nCPUsLocal` | `4` | OMP threads per file |
| `-mapDetector` | `1` | Run DetectorMapper if Map.bin missing |
| `-convertFiles` | `1` | Convert to Zarr before integrating |
| `-startFileNr` | `-1` | Start frame (-1 = read from dataFN) |
| `-endFileNr` | `-1` | End frame (-1 = single file) |
| `-writeMat` | `0` | Write .mat output |
| `-skipExisting` | flag | Skip already-processed files |
| `-liveViewer` | flag | Launch live viewer dashboard |
| `-peakFit` | flag | Enable 1D peak fitting |
| `-dataLoc` | `exchange/data` | HDF5 dataset path |
| `-darkLoc` | `exchange/dark` | Dark HDF5 dataset path |

Parameter overrides can be appended at end of command: `MinRad 10 MaxRad 1000 RBinSize 0.5`

---

### v10 executable chain

```
integrator.py
  ├─ DetectorMapper      → Map.bin, nMap.bin, maskMap.bin  (first run only, via -mapDetector 1)
  └─ IntegratorZarrOMP   → lineout + caked zarr per file
```

### Output files

| File | Use |
|---|---|
| `*_lineout.xy` | 2θ (degrees) vs intensity (text) — load in GSAS-II, plot directly |
| `*_lineout.bin` | Binary lineout (backward compatible) |
| `*_caked.hdf.zarr.zip` | Full caked data — GSAS-II compatible (zarr v2 required) |
| `Map.bin`, `nMap.bin`, `maskMap.bin` | Geometry maps (generated once per detector/params) |

### Parameter file keys needed for integration

From `refined_MIDAS_params*.txt` (auto-written by `midas_auto_calibrate`):

```
Lsd        <µm>       detector distance
BC         <Y> <Z>    beam center
Wavelength <Å>
px         <µm>       pixel size
NrPixelsY  <int>
NrPixelsZ  <int>
RMin       <px>       integration range start
RMax       <px>       integration range end
RBinSize   <px>       radial bin size
EtaMin     <deg>
EtaMax     <deg>
EtaBinSize <deg>
DataType   6          (6=TIFF, 8=HDF5)
FileStem   <stem>     from calibration symlink
Folder     <path>
```

> `midas_auto_calibrate` writes ALL of these automatically — no manual editing needed for standard integration.

### Supported input formats

`.tif/.tiff`, `.ge/.ge1-.ge5`, `.h5/.hdf5/.nxs`, `.zip` (Zarr)
