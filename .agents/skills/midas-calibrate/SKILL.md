---
name: midas-calibrate
description: Run MIDAS detector geometry calibration using a CeO2 or LaB6 calibrant image. Use when the user asks to calibrate, run calibration, find beam center, refine detector distance (Lsd), or mentions CeO2/LaB6/ceria calibrant files.
compatibility: Requires MIDAS v11 + midas-suite (midas_calibrate ≥0.2.7). Native-Python calibration runs in-process; no compiled C binaries required for the primary path.
metadata:
  author: pawan-tripathi
  version: "2.0"
  midas-version: "11.0"
  package: "midas_calibrate"
  manual: MIDAS/manuals/FF_Calibration.md
---

## Calibration Workflow (MIDAS v11)

In v11, calibration is the **native-Python `midas_calibrate` package**, run
in-process by APEXA. The legacy C binary `CalibrantPanelShiftsOMP` is **archived**;
`CalibrantIntegratorOMP` is the active C/OpenMP superset used only as a fallback.

### Calibration engines — three, not two (don't confuse them)

There are **two Python generations sitting above one C/OpenMP binary** — *not*
"Python vs C++". Both `midas_calibrate` packages are pure-Python PyTorch
(`setuptools` backend, `torch>=2.1`, zero C/C++/CMake sources).

| Engine | Language | Who calls it | Outputs |
|---|---|---|---|
| **`midas_calibrate` (v1)** | Python / PyTorch | **APEXA's `midas_auto_calibrate`** (native, in-process via `apexa_midas_native.native_calibrate`) | `refined_MIDAS_params_<material>.txt` only |
| **`midas_calibrate_v2`** | Python / PyTorch | the package's own notebooks (one-shot `calibrate()`, Bayesian/Laplace UQ, multi-panel, residual maps) | `calibration.json` + `residual_corr.bin` (+ refined params) |
| **`CalibrantIntegratorOMP`** | **C / OpenMP** | `AutoCalibrateZarr.py` and the `midas-autocalibrate` CLI (subprocess **fallback**) | `refined_MIDAS_params*.txt`, `*corr.csv`, `autocal.log` |

- **APEXA uses v1** (`midas_calibrate`) for the native path, falling back to the
  **C/OMP** binary via `AutoCalibrateZarr.py`. It does **not** invoke
  `midas_calibrate_v2` today — so `calibration.json` / `residual_corr.bin` are
  *not* produced by `midas_auto_calibrate`.
- `v1` vs `v2` is an **API generation** difference (v2 adds UQ, multi-panel,
  residual-correction maps), both differentiable PyTorch. The only C/C++ code in
  the calibration path is the OMP binary.

> **Robust / seeded retry (e.g. a low-SNR frame that landed in a false basin):**
> call `midas_auto_calibrate(..., seed_from_params="<good neighbour's
> refined_MIDAS_params*.txt>")`. It reads a complete, correct seed (both BC coords
> + Lsd in µm) and runs the proven v1 engine. Do **NOT** hand-write runner scripts
> against the `midas_calibrate_v2` package — it has its own open issues (e.g.
> "keeping 1 of N rings" → IndexError) and is not the supported path.

Two MCP tools:

| | **`midas_auto_calibrate`** (preferred) | **`run_ff_calibration`** (lower-level) |
|---|---|---|
| Input | Point at an image — auto-detects format/energy/material | A prepared `Parameters.txt` |
| Engine | native `midas_calibrate` in-process → `AutoCalibrateZarr.py` fallback | `midas-autocalibrate` CLI → `CalibrantIntegratorOMP` fallback |
| Use when | Almost always | You already have a full param file and want the CLI path |

> Run [[midas-validate]] (`inspect_dataset_file`, `enumerate_bragg_rings`) first if
> you're unsure of the geometry or which rings to expect.

### Step 1 — Find files

Use `list_directory` to locate:
- Calibrant image: CeO2 or LaB6 diffraction image (`.tif`, `.ge`, `.h5`, `.zip`)
- Parameter file (optional): `ps_*.txt`, `Parameters.txt`, or any `*params*.txt`
- Dark file (optional): file starting with `dark_`

### Step 2 — Call the tool

```
midas_auto_calibrate(
    image_file       = "<absolute path>",
    param_file       = "<absolute path to parameter file>",   # optional — auto-detected
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

### v11 rules — never get these wrong

- **Native engine is the primary path** — `midas_auto_calibrate` runs the
  `midas_calibrate` package in-process. It falls back to `AutoCalibrateZarr.py`
  (which now drives `CalibrantIntegratorOMP`) only if the native engine is
  unavailable. Set `APEXA_USE_NATIVE_MIDAS=0` to force the subprocess path.
- **`CalibrantPanelShiftsOMP` is archived** — never invoke it directly.
  `CalibrantIntegratorOMP` is the active superset (integrated tilt/BC/Lsd +
  panel shifts + outlier-ring rejection in one call).
- **No `-StoppingStrain`** — removed. Use `n_iterations` instead.
- **`lsd_guess` is in µm** — 650 mm = 650000 µm (auto-parsed from filename if `_650mm_` present)
- **Output file has material suffix**: `refined_MIDAS_params_CeO2.txt`, not `refined_MIDAS_params.txt`
- **PYTHONPATH / env** — handled automatically by `get_midas_env()`.

### `ImTransOpt` (image transformation) — auto-detected, but verify

`ImTransOpt` controls flip/transpose of the raw image to align with the MIDAS
lab frame. Per `MIDAS/manuals/README.md` it is **detector-mount specific** — there
is *no* reliable extension-based rule (a Pilatus TIFF can need `2`, a GE file
can need `0`, depending on how the detector is physically oriented).

The `midas_auto_calibrate` tool resolves it in this order:

1. **`image_transform` kwarg** — explicit user value wins.
2. **`ImTransOpt` line in the supplied `parameters_file`** — re-used as-is.
3. **Sibling `parameters.txt` / `Parameters.txt` / `params.txt` next to the image** — auto-picked.
4. **Fallback: `0` (no transform) + warning** — agent should flag this to the user.

Codes (from `MIDAS/manuals/README.md`):

| `ImTransOpt` | Effect |
|---|---|
| `0` | No transform |
| `1` | Flip left/right |
| `2` | Flip top/bottom |
| `3` | Transpose |

> **If calibration converges to nonsense or rings are mirrored, ImTransOpt is
> the first thing to suspect.** Verify against a physical fiducial (beam-stop wire,
> fiducial dot) whose position on the detector you can predict.

Pass multiple transforms space-separated (applied in order):
```
midas_auto_calibrate(..., image_transform="1 3")   # flip LR then transpose
```

### Calibrant auto-detection from filename

If the image filename contains these patterns, no param file needed for material/energy:
- `ceo2`, `ceria` → CeO2, space group 225
- `lab6` → LaB6, space group 221
- `71p676keV` → wavelength 0.17298 Å
- `650mm` → Lsd guess 650000 µm

### Output files (written to image directory)

- `refined_MIDAS_params_<material>.txt` — **primary output**, use for integration and FF-HEDM
- `autocal.log` — iteration history (subprocess/`AutoCalibrateZarr.py` path)
- `<stem>.lineout.xy` — 2θ vs intensity for visual check
- `*corr.csv` — per-ring residuals (`AutoCalibrateZarr.py`/`CalibrantIntegratorOMP`
  path; view with `plot_calibrant_results`)

> **`midas_auto_calibrate` writes the refined `.txt` only** — both its native engine
> (v1 `midas_calibrate`, `result.params.write()`) and the `AutoCalibrateZarr.py`
> fallback produce `refined_MIDAS_params_<material>.txt` as the canonical output.
> The `calibration.json` + `residual_corr.bin` artifacts belong to the *separate*
> `midas_calibrate_v2` one-shot `calibrate(output_dir=…)` API used in the package
> notebooks — APEXA does **not** invoke that path today, so do not expect those
> files from `midas_auto_calibrate`.

### Convergence guide

| MeanStrain | Quality |
|---|---|
| < 50 µε | Excellent |
| 50–200 µε | Good |
| 200–500 µε | Acceptable — try more iterations |
| > 500 µε | Poor — check image, param file, rings |

> **Saturation check:** unattenuated CeO2 saturates the detector at UINT32_MAX,
> producing flat-topped rings that won't refine. If MeanStrain stays high, check
> the `*_lineout.xy` signal-to-noise before suspecting the calibration code.

### Visualize the result

After calibration, use [[midas-visualize]]:
- `plot_calibrant_results` — ring-fit residuals (`*corr.csv`)
- `plot_lineout_comparison` — measured lineout vs ideal calibrant ring positions
