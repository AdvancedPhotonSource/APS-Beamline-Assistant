---
name: midas-gsasii
description: GSAS-II peak fitting and Rietveld refinement on MIDAS caked output. Use when the user asks to refine, run GSAS-II, fit peaks with Rietveld, get Rwp, extract lattice parameters from integrated data, or run combined integration + refinement pipeline.
compatibility: Requires MIDAS v11 (utils/gsas_ii_refine.py, FF_HEDM/workflows/integrate_and_refine.py), GSAS-II (conda install gsas2pkg -c briantoby), zarr==2.18.3
metadata:
  author: pawan-tripathi
  version: "1.0"
  midas-version: "11.0"
  manual: MIDAS/manuals/GSAS-II_Integration.md
---

## GSAS-II Refinement Workflows (MIDAS v11)

Two MCP tools depending on the use case:

| | **Tool A — Standalone Refinement** | **Tool B — Combined Pipeline** |
|---|---|---|
| MCP tool | `run_gsas_refinement` | `run_live_analysis` |
| Input | `.zarr.zip` (already integrated) | Raw data + params (or existing .zarr.zip) |
| Script | `utils/gsas_ii_refine.py` | `FF_HEDM/workflows/integrate_and_refine.py` |
| Best for | Refining after integration is done | End-to-end: integrate → refine in one step |

> **CIF files are required for refinement.** Use `fetch_cif_from_mp` (KnowledgeAgent) to download from Materials Project if the user doesn't have one.

---

## Tool A: `run_gsas_refinement`

Refines caked histograms from a `.zarr.zip` using multi-stage Rietveld refinement.

```
run_gsas_refinement(
    data_file        = "<path to .zarr.zip>",
    cif_files        = ["CeO2.cif"],         # one or more phases
    output_dir       = "refinement/",         # default
    bkg_terms        = 6,                     # Chebyshev background terms
    two_theta_limits = [2.0, 15.0],           # optional 2θ range
    no_atoms         = False,                 # skip atomic refinement
    no_export        = False,                 # skip CIF/CSV export
    n_cpus           = 4,                     # parallel histograms
    instprm_file     = "instrument.instprm",  # optional
)
```

### Refinement stages

| Stage | Refines | Parameters |
|---|---|---|
| 1 | Background + Scale | Chebyshev coefficients, phase scale |
| 2 | Unit Cell | a, b, c, α, β, γ |
| 3a | Gaussian profile | U, V, W |
| 3b | Lorentzian profile | X, Y, SH/L |
| 4 | Atomic positions + thermal | Coordinates, Uiso (optional, skip with `no_atoms`) |

---

## Tool B: `run_live_analysis`

Combined pipeline: Integration (Stage 1) → GSAS-II Refinement (Stage 2).

### Batch backend (CPU)

```
run_live_analysis(
    backend     = "batch",
    param_file  = "refined_MIDAS_params_CeO2.txt",
    data_file   = "data/sample_000001.h5",
    cif_files   = ["CeO2.cif"],
    output_dir  = "refinement/",
    n_cpus      = 8,
)
```

### Stream backend (GPU, folder)

```
run_live_analysis(
    backend    = "stream",
    param_file = "refined_MIDAS_params_CeO2.txt",
    folder     = "/data/experiment/scan_01/",
    cif_files  = ["CeO2.cif"],
    output_dir = "refinement/",
    n_cpus     = 8,
)
```

### Stream backend (GPU, live PVA)

```
run_live_analysis(
    backend    = "stream",
    param_file = "setup.txt",
    pva        = True,
    pva_ip     = "10.54.105.139",
    cif_files  = ["CeO2.cif"],
    output_dir = "refinement/",
    n_cpus     = 8,
)
```

### Skip integration (refine existing .zarr.zip)

```
run_live_analysis(
    backend          = "batch",
    param_file       = "params.txt",
    cif_files        = ["CeO2.cif"],
    skip_integration = True,
    zarr_file        = "output/CeO2_caked.hdf.zarr.zip",
)
```

### Integration only (no GSAS-II)

```
run_live_analysis(
    backend         = "stream",
    param_file      = "params.txt",
    folder          = "/data/scan_01/",
    cif_files       = [],
    skip_refinement = True,
)
```

---

## CIF File Fetcher: `fetch_cif_from_mp`

Downloads CIF files from the Materials Project database. Routed to KnowledgeAgent.

```
fetch_cif_from_mp(
    formula    = "CeO2",
    output_dir = ".",              # saves CeO2_mp-XXXXX.cif
    mp_api_key = None,             # from env MP_API_KEY or ~/.config/.pmgrc.yaml
)
```

Returns up to 3 structures sorted by thermodynamic stability (energy above hull).

---

## Output files

| File | Tool | Description |
|---|---|---|
| `hist_NNNN.gpx` | Both | Per-histogram GSAS-II project file |
| `hist_NNNN_<phase>.cif` | Both | Refined crystal structure (unless `no_export`) |
| `hist_NNNN_data.csv` | Both | 2θ, observed, calculated, difference |
| `refinement_summary.json` | Both | Aggregated results: Rwp, lattice params, counts |

### refinement_summary.json structure

```json
{
  "data_file": "CeO2_caked.hdf.zarr.zip",
  "total_histograms": 360,
  "succeeded": 358,
  "failed": 2,
  "skipped": 0,
  "mean_Rwp": 4.23,
  "histograms": [
    {
      "index": 0,
      "Rwp": 3.87,
      "cell_params": {"a": 5.4116, "b": 5.4116, "c": 5.4116, "alpha": 90, "beta": 90, "gamma": 90},
      "status": "converged"
    }
  ]
}
```

---

## Prerequisites

1. **GSAS-II**: `conda install gsas2pkg -c briantoby` (in midas_env)
2. **zarr**: `pip install zarr==2.18.3` (version-sensitive for MIDAS zarr.zip format)
3. **mp-api** (for CIF fetcher): `uv sync --extra extra` or `pip install mp-api`
4. **MP API key** (for CIF fetcher): Get from https://next-gen.materialsproject.org/api → set `MP_API_KEY` env var

---

## Typical workflow

```
1. Calibrate:    midas_auto_calibrate (CalibrationAgent)
2. Integrate:    midas_integrate_2d_to_1d (AnalysisAgent) → .zarr.zip
3. Fetch CIF:    fetch_cif_from_mp("CeO2") (KnowledgeAgent) → CeO2.cif
4. Refine:       run_gsas_refinement(zarr.zip, CeO2.cif) (AnalysisAgent)
5. Visualize:    plot_integrator_peaks.py or plot_caked_peaks.py (VisualizationAgent)
```

Or combined (steps 2+4 in one call):
```
1. Calibrate:    midas_auto_calibrate
2. Fetch CIF:    fetch_cif_from_mp("CeO2")
3. Integrate+Refine: run_live_analysis(backend="batch", ...)
4. Visualize:    plot_caked_peaks.py
```
