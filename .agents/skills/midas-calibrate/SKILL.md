---
name: midas-calibrate
description: Run MIDAS detector geometry calibration using a CeO2 or LaB6 calibrant image. Use when the user asks to calibrate, run calibration, find beam center, refine detector distance (Lsd), or mentions CeO2/LaB6/ceria calibrant files.
compatibility: Requires MIDAS v10 installation (MIDAS_PATH in .env), midas_env conda environment, APEXA MCP server running
metadata:
  author: pawan-tripathi
  version: "1.0"
  midas-version: "10.0"
  manual: MIDAS/manuals/FF_Calibration.md
---

## Calibration Workflow (MIDAS v10)

Use the `midas_auto_calibrate` MCP tool. It calls `AutoCalibrateZarr.py` which internally runs `CalibrantPanelShiftsOMP` in a single call with 40 iterations.

### Step 1 — Find files

Use `list_directory` to locate:
- Calibrant image: CeO2 or LaB6 diffraction image (`.tif`, `.ge`, `.h5`, `.zip`)
- Parameter file: `ps_*.txt`, `Parameters.txt`, or any `*params*.txt`
- Dark file (optional): file starting with `dark_`

### Step 2 — Call the tool

```
midas_auto_calibrate(
    image_file       = "<absolute path>",
    param_file       = "<absolute path to parameter file>",
    dark_file        = "<absolute path to dark, or omit>",
    n_iterations     = 40,
    mult_factor      = 2.5,
    eta_bin_size     = 5.0,
    first_ring_nr    = 1,
    bad_px_intensity = -2,
    gap_intensity    = -1
)
```

### Supported formats

| Extension | Format |
|---|---|
| `.tif`, `.tiff` | TIFF |
| `.ge`, `.ge1`–`.ge5` | GE binary |
| `.h5`, `.hdf5`, `.hdf`, `.nxs` | HDF5 |
| `.zip` | Zarr |

### v10 rules — never get these wrong

- **No `-StoppingStrain`** — removed. Use `n_iterations` instead.
- **`lsd_guess` is in µm** — 650 mm = 650000 µm (auto-parsed from filename if `_650mm_` present)
- **Output file has material suffix**: `refined_MIDAS_params_CeO2.txt` not `refined_MIDAS_params.txt`
- **PYTHONPATH must include `MIDAS/utils/`** — handled automatically by `get_midas_env()`
- **Use `midas_env` conda Python** — not system `python3`

### Calibrant auto-detection from filename

If the image filename contains these patterns, no param file needed for material/energy:
- `ceo2`, `ceria` → CeO2, space group 225
- `lab6` → LaB6, space group 221
- `71p676keV` → wavelength 0.17298 Å
- `650mm` → Lsd guess 650000 µm

### Output files (written to image directory)

- `refined_MIDAS_params_<material>.txt` — **primary output**, use for integration and FF-HEDM
- `autocal.log` — iteration history
- `calibrant_screen_out.csv` — raw C binary output
- `<stem>.lineout.xy` — 2θ vs intensity for visual check

### Convergence guide

| MeanStrain | Quality |
|---|---|
| < 50 µε | Excellent |
| 50–200 µε | Good |
| 200–500 µε | Acceptable — try more iterations |
| > 500 µε | Poor — check image, param file, rings |
