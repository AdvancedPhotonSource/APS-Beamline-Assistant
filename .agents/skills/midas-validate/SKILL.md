---
name: midas-validate
description: Validate MIDAS parameter files and datasets before running calibration, integration, or HEDM. Use when the user asks to validate/check a parameter file, diagnose why a param file is wrong, inspect a dataset (zarr/HDF5/GE), enumerate expected Bragg rings, or verify the MIDAS installation. Run this FIRST in any MIDAS workflow.
compatibility: Requires MIDAS v11 + midas-suite ≥0.4.0 (midas-params ≥0.3.5). All checks run through pip-package CLIs — no compiled binaries needed.
metadata:
  author: pawan-tripathi
  version: "1.0"
  midas-version: "11.0"
  package: "midas_params"
---

## Why validate first

A bad parameter file fails *late* — after a 20-minute pipeline, or worse, it
"succeeds" with silently wrong geometry. The MIDAS v11 `midas-params` package
catches these up front. Validation is the cheap, deterministic gate that belongs
at the **start** of every calibration / integration / HEDM session.

> All five tools below are pure-Python (`midas-params` CLI + `validate_midas_installation`).
> None require compiled C binaries or a GPU. Safe to run anywhere.

---

## The five validation tools

| Tool | Answers | Backed by |
|---|---|---|
| `validate_midas_installation` | Is MIDAS wired up? Which binaries/packages are present? | native package probe + `bin/` scan |
| `validate_parameter_file` | Is this param file complete + self-consistent for FF/NF/PF/RI? | `midas-params validate` |
| `diagnose_parameter_file` | *Why* is it wrong, and how do I fix it? (LLM-ready) | `midas-params diagnose` |
| `inspect_dataset_file` | What's actually in this zarr/HDF5/GE file? | `midas-params inspect` |
| `enumerate_bragg_rings` | Which rings should I see at this geometry? | `midas-params rings` |

---

## 1. `validate_midas_installation` — environment preflight

Run once at the start of a session (or when something fails mysteriously).

```
validate_midas_installation()
```

Reports:
- **`native_packages`** — the 18 v11 pip packages (midas_calibrate, midas_integrate,
  midas_pipeline, midas_nf_pipeline, midas_params, midas_stress, …) and their versions.
  **This is the primary signal in v11** — most workflows run through these in-process.
- **`executables`** — active C/OpenMP binaries (IndexerOMP, FitPosOrStrainsOMP,
  CalibrantIntegratorOMP, IntegratorZarrOMP, …). Used only as the fallback path.
- **`optional_executables`** — GPU builds (need CUDA; absent on macOS/CPU) and
  binaries archived in v11 (CalibrantPanelShiftsOMP, GrainTracking, CalcStrains).
  **Missing entries here are NOT failures** — they're expected on most installs.

> If `native_packages` are all present, the system is healthy even if many
> `optional_executables` are missing.

---

## 2. `validate_parameter_file` — completeness + consistency

```
validate_parameter_file(
    param_file = "<absolute path to Parameters.txt / paramstest.txt>",
    pipeline   = "ff"      # ff | nf | pf | ri  — which pipeline to validate against
)
```

- Checks required keys are present, types are right, and values are physically
  sane for the named pipeline.
- **`pipeline` matters** — FF, NF, PF, and radial-integration (RI) need different
  key sets. Pick the one matching the next step. Default `"ff"`.
- Returns a pass/fail report with the specific missing/invalid keys.

When a **zarr archive** is provided directly to the FF/PF workflow, RawFolder /
FileStem / StartNr / EndNr are not needed — the pipeline runs with
`--skip-validation`. **Do NOT call `validate_parameter_file` on a param file
destined for a zarr-input run** (it will false-fail on `KeyError: RawFolder`).

---

## 3. `diagnose_parameter_file` — fix-it guidance

When `validate_parameter_file` fails and you need the *why*:

```
diagnose_parameter_file(
    param_file    = "<path>",
    pipeline      = "ff",
    output_format = "json"      # json (LLM-ready) | text (human)
)
```

Produces a structured diagnosis payload: each problem, its severity, and a
concrete remedy. Use `output_format="json"` when an agent will act on it;
`"text"` when showing the user directly.

---

## 4. `inspect_dataset_file` — what's really in the data

Before trusting a param file's claims, look at the actual dataset:

```
inspect_dataset_file(dataset_file = "<path to .zarr.zip / .h5 / .ge / .tif>")
```

Auto-extracts the real geometry/scan metadata from the file: detector dimensions,
frame count, omega range, dataset paths, and any embedded analysis parameters.
Use this to:
- Confirm frame count / omega range before integration or HEDM (avoids the
  "model assumed 360 frames, file has 920" failure).
- Recover geometry from a zarr when the sidecar param file is missing.
- Pick the right `DataType` / `NrPixelsY/Z`.

---

## 5. `enumerate_bragg_rings` — expected ring positions

```
enumerate_bragg_rings(param_file = "<path with Wavelength, Lsd, lattice, SG>")
```

Lists the visible Bragg rings (hkl, d-spacing, 2θ, ring radius in px) for the
crystal + geometry in the param file. Use it to:
- Sanity-check calibration: do predicted rings line up with observed rings?
- Choose `RMin` / `RMax` / `first_ring_nr` for integration.
- Decide which rings to index in FF-HEDM (`OverAllRingToIndex`, `RingThresh`).

> The lower-level `midas-params rings` CLI can also take `--wavelength / --lsd /
> --lattice / --space-group` directly (no param file) — useful for "what would I
> see if…" exploration via `run_command`.

---

## Where validation sits in the workflow

```
inspect_dataset_file        ← what is this data?
        │
validate_parameter_file     ← is my param file OK for the next step?
        │  (fails?) → diagnose_parameter_file → fix → re-validate
        │
enumerate_bragg_rings       ← do the expected rings match what I'll calibrate to?
        ↓
midas_auto_calibrate  →  midas_integrate_2d_to_1d  →  run_ff_hedm_full_workflow
   (midas-calibrate)        (midas-integrate)            (midas-hedm)
```

See also: [[midas-calibrate]], [[midas-integrate]], [[midas-hedm]], [[midas-ff-hedm]].
