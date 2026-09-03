---
name: gsas2-agentic
description: Autonomous Rietveld refinement of powder patterns with no recipe supplied — the agent picks its own parameter order, can retrieve its own starting structure from COD, and returns a trust verdict. Use when the user asks to refine a powder pattern, track phases through an in-situ series, quantify a phase mixture, identify a structure from a pattern, or judge whether an existing refinement can be acted on. For MIDAS caked .zarr.zip output use midas-gsasii instead.
compatibility: Requires the GSAS-II conda env (conda install gsas2full -c briantoby) and the Agentic-GSAS-II repository. Set APEXA_GSASII_PYTHON and APEXA_AGENTIC_GSAS2 if they are not at their defaults.
metadata:
  author: pawan-tripathi
  version: "1.0"
  server: gsas2_server.py
  upstream: https://github.com/pawantr/Agentic-GSAS-II
---

## When to use this rather than midas-gsasii

| | **gsas2-agentic** (this) | **midas-gsasii** |
|---|---|---|
| Input | any powder pattern: `.xye`, `.fxye`, `.dat`, GSAS native | MIDAS caked `.zarr.zip` |
| Starting model | you supply CIFs, **or** it retrieves them | you supply CIFs |
| Strategy | derivative-driven, chooses its own order | fixed recipe |
| Returns | + phase fractions, trust verdict, decision trace | R<sub>wp</sub>, cell |
| Scope | one pattern, or a sequential series | one pattern |

Both wrap GSAS-II. Neither supersedes the other — pick by where the data
came from.

## Tools

| Tool | Use |
|---|---|
| `refine_pattern` | one pattern, CIFs supplied |
| `propose_structures` | pattern + a name → ranked COD candidates with M₂₀ |
| `refine_series_submit` | an in-situ stack; returns a job id |
| `refinement_status` | poll that job |
| `assess_refinement` | trust verdict on a refinement already on disk |

## Typical flows

**Refine a pattern you have a structure for**

```
refine_pattern(data_file="scan_042.xye", cif_files=["LiCoO2.cif"],
               instprm_file="11bm.instprm", output_dir="ref/042")
```

**Refine a pattern you do not have a structure for**

```
propose_structures(data_file="unknown.xye", text="goethite", top_n=3)
→ pick the highest M20 candidate, then pass its cif path to refine_pattern
```

**Track phases through an in-situ series**

```
refine_series_submit(data_files=[...], cif_files=[...]) → job_id
refinement_status(job_id=...)   # poll; do not block on it
```

## Reading the trust verdict

Every refinement returns `trust: {trustworthy, flags, note}`. A fit
always produces numbers; the verdict says whether they should be acted
on. Flags name their reason rather than scoring:

- `stopped on a time budget` — a throughput bound, not convergence
- `hit the step cap` — the fit was still improving when cut off
- `Rietveld exceeds the Le Bail floor by N points` — the structural
  model is the limiting factor, not the profile or background
- `direct cell refinement was rolled back` — the cell step made the fit
  worse and was undone
- `no parameter was accepted` — the model never moved

**In a closed loop, treat a flagged result as a reason to skip the
point, not to widen the search around it.** A wrong cell that stops
predicting any observed peak costs almost nothing in R<sub>wp</sub>, so
the residual alone will not catch it.

## M₂₀ as a confidence signal

`propose_structures` returns de Wolff M₂₀ for each candidate, computed
from the pattern alone with no reference structure involved. On a
240-pattern mineral benchmark the rate of a wrong phase fell
monotonically with it:

| M₂₀ | wrong phase |
|---|---|
| ≥ 20 | 3.2% |
| 10–20 | 7.5% |
| 5–10 | 11.2% |
| < 5 | 17.6% |

Low M₂₀ means widen the search, not refine harder.

## Limitations worth knowing before you rely on them

- **Lattice parameter through a series.** `refine_series_submit` does not
  currently produce a thermal-expansion trend: on the 17-temperature
  benchmark the cell is reported identically at every temperature,
  because that entry point predates the framework's direct
  cell-refinement path. Phase fractions and residuals are unaffected.
- **Absolute weight fractions** carry roughly 17 wt% mean error on known
  mixtures, traced to uncorrected microabsorption. Relative change
  between conditions is far better behaved and is usually what a
  yield-optimisation loop needs.
- **Cubic phases** reach their lattice through a separate path from all
  other symmetries; if a cubic cell comes back unrefined, check the
  `cell_route` record rather than assuming the fit failed.

## Operational notes

- Refinements take minutes. `refine_pattern` blocks up to 30 min; a
  series always submits and polls.
- `refine_pattern` is idempotent on its scientific inputs — re-issuing
  the same call skips the work. Pass `force=true` to override.
- Every run writes a GPX, a CIF, diagnostic plots and a JSON decision
  trace to `output_dir`. Reruns reproduce the trace exactly, so a result
  stays checkable later.
- The framework runs in a subprocess on the GSAS-II interpreter, so a
  GSAS-II crash does not take the agent down.
