---
name: midas-mask
description: Build a detector MaskFile (uint8 TIFF, 1=masked) for MIDAS FF/NF/integration. Use when the user asks to mask dead pixels, module gaps/strides, hot pixels/zingers, NaNs, or diffuse blobs; to make/convert a mask; or when a run gives "0 peaks" that a bad mask could explain.
compatibility: Requires the APEXA .venv (numpy, tifffile, PIL, h5py, scipy; differentiable mode also needs torch + midas_integrate_v2). Detector geometry from knowledge_base/data/detector_presets.json.
metadata:
  author: pawan-tripathi
  version: "1.0"
  midas-version: "11.0"
  tool: build_detector_mask
---

## Detector masking (MIDAS MaskFile contract)

Beamline frames carry defects that corrupt HEDM: dead pixels, module strides/gaps
(Pilatus/Eiger tiles), hot pixels/zingers, NaN/Inf, and diffuse blobs. `build_detector_mask`
produces a downstream-correct **MaskFile** and, report-only by default, hands back a
paste-ready `MaskFile` line.

### The downstream contract — get this exactly right
- **Format:** uint8 TIFF, **DataType code 7** (tiff-uint8). Values are strictly {0, 1}.
- **Convention: `1 = masked` (excluded), `0 = valid`.** This is the MIDAS convention
  (see integrator param `MaskFile <path>  uint8 TIFF, 0=valid, 1=masked`). An **inverted**
  mask (1=valid) masks the whole detector → **"0 peaks"**. If a downstream run finds zero
  peaks right after a mask was added, suspect an inverted mask first.
- **Orientation: RAW detector orientation.** The C readers apply `ImTransOpt` themselves —
  the mask is in the same raw frame as the data, do NOT pre-apply transforms.
- **Shape is `(NrPixelsZ, NrPixelsY)`** = (rows/Z/slow axis, cols/Y/fast axis). NrPixelsY is
  detector *width*, NrPixelsZ is *height*.

### Modes (csv `modes=`, OR-combined)
| Mode | Catches | Needs |
|---|---|---|
| `statistical` (default) | dead (≤ floor), hot/zingers (>k·MAD over 3×3 median), NaN/Inf | image_path |
| `value` | sentinel pixels (e.g. Pilatus `-1,-2`) | image_path + sentinels (or detector default_sentinels) |
| `module_gap` | tile gaps/strides from the registry geometry | detector with module_gaps (Pilatus 2M only, sourced) |
| `differentiable` | subtle/systematic bad pixels that break ring (η) uniformity | image_path + **calibrated** param_file |
| `convert` | wrap an existing `.mask`/TIFF as a MaskFile (nonzero → masked) | image_path |
| `combine` | OR several existing mask TIFFs together | combine_files |

### Detector registry (`knowledge_base/data/detector_presets.json`)
Pass `detector=` (name/alias) to resolve `NrPixelsY/Z`, pixel size, `module_gaps`, default
sentinels, and DataType — every value sourced from a real MIDAS repo line. Or pass
`nr_pixels_y`/`nr_pixels_z` directly. Known: eiger2_500k/4m/9m/16m, pilatus3_1m/2m/6m,
varex_2923, ge, ge5, hydra. **Only Pilatus 2M has a concrete module-gap layout in the MIDAS
repo** — others are `null` (module_gap no-ops with a warning; flat panels like Varex have none).

### Differentiable mode (v1 headline) — the traps
- **Requires a calibrated param file** (Lsd/BC/tilts/px/binning) for the pixel→(R,η) forward
  model. No param file ⇒ the lint hard-blocks with an actionable error. Never guess geometry.
- **`LearnableMask(NrPixelsZ, NrPixelsY)` is Z-FIRST.** A transposed call raises on a
  non-square detector. The extracted hard mask is already True=masked (feeds the write gate
  directly).
- **Cost:** the forward is recomputed every Adam step → O(NrPixelsY×NrPixelsZ) per step,
  expensive full-res on CPU. Levers: `diff_crop_rings=True` (focuses the η loss on ring
  bands, default), fewer `diff_iterations` (200–500 typical), float32. `diff_smoothness>0`
  for clustered blobs; `diff_sparsity` (default 1e-4) keeps the mask from over-growing.

### Report-only vs. write_param (opt-in patch)
- **Default: report-only.** Returns `output_path`, mask stats, and the `MaskFile <path>` line.
  It does NOT touch any param file.
- **`write_param=True`** backs up `param_file` → `param_file.bak.<n>` (never overwriting an
  existing `.bak`) **before** editing, then inserts/replaces the `MaskFile` line in place, and
  reports the backup path + the old/new line diff.

### Convention gate (for future contributors)
Every mode converges to a single boolean `mask_1masked` (True=masked) before ONE
`_write_mask_tiff` write. If you ever wire in a producer with the opposite convention
(`build_gasket_mask` returns True=**allowed**; calibrate_v2 `detect_panel_mask` returns
1=**valid**), **invert it (`~`)** at the boundary — otherwise you write an inverted MaskFile.

### Typical calls
```python
# 1) Quick dead+hot+nan mask on a dark, resolve dims from the detector name:
build_detector_mask(image_path=".../dark.tif", detector="pilatus 2m", modes="statistical")

# 2) Pilatus module gaps + sentinels + dead pixels, all combined, then patch the params:
build_detector_mask(image_path=".../dark.tif", detector="pilatus 2m",
                    modes="module_gap,value,statistical",
                    param_file=".../params.txt", write_param=True)

# 3) Differentiable auto-learn (needs calibrated geometry):
build_detector_mask(image_path=".../frame.tif", detector="pilatus 2m",
                    modes="differentiable", param_file=".../refined_MIDAS_params.txt",
                    diff_iterations=300, diff_crop_rings=True)
```
