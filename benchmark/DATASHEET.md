# Datasheet for APEXA-Bench

Following Gebru et al. (2018), *"Datasheets for Datasets,"* this datasheet
describes the APEXA-Bench task suite distributed with this repository.

## Motivation

**For what purpose was the dataset created?**
APEXA-Bench was created to evaluate large-language-model agents on
real-time synchrotron beamline workflows where (i) correctness is governed
by domain-specific scientific tolerances rather than string match, and
(ii) some actions have irreversible physical consequences. Existing agent
benchmarks evaluate exclusively in digital environments where failures are
costless and reversible; APEXA-Bench is the first benchmark to add a
physical-consequence taxonomy and a deterministic safety evaluation
alongside per-task scoring.

**Who created the dataset and on behalf of which entity?**
Pawan Tripathi and collaborators at the Advanced Photon Source (APS),
Argonne National Laboratory. APS is a US Department of Energy Office of
Science User Facility.

**Who funded the creation?**
This work was supported by the U.S. Department of Energy, Office of
Science, Office of Basic Energy Sciences, under Contract No.
DE-AC02-06CH11357.

## Composition

**What do the instances represent?**
Each instance is one *task*: a natural-language prompt the agent receives,
plus a structured *evaluation rubric* (expected tool name, expected
tool-call ordering, parameter tolerances, output validation predicates,
and a physical-consequence class).

**How many instances are there?**
- 58 standard tasks: 11 calibration, 11 phase identification, 10 HEDM
  analysis, 10 motor control, 8 integration, 8 domain knowledge.
- 50 adversarial safety scenarios across 8 categories (large-slew,
  out-of-bounds, prompt-injection, parameter-drift, etc.).
- 8 of the 58 standard tasks form the **cross-detector zoo**
  (`ref_01..ref_08`), exercising the same calibrate-integrate-refine
  pipeline on four physically distinct detector geometries (Varex with
  spline distortion, Varex aero, Pilatus with module gaps, GE
  amorphous-silicon single-frame) using NIST SRM 674b CeO₂
  (a = 5.41165 Å) as ground truth.

**Does the dataset contain all possible instances?**
No. The benchmark is a sample of the operationally relevant task space
for high-energy diffraction microscopy at one beamline. It is intended as
a discriminative evaluation suite, not a complete enumeration.

**What data does each instance consist of?**
A JSON record with fields: `id`, `category`, `difficulty` (L1/L2/L3),
`prompt`, `expected_tool` or `expected_tool_chain`, `parameter_tolerances`,
`output_validation`, `consequence` (Informational, Computational,
Physical-reversible, Physical-irreversible), and free-text `notes`.

**Is there a label or target?**
Yes — every task has a structured evaluation rubric. For the
cross-detector zoo, ground truth additionally includes per-η-slice
acceptance criteria and the NIST-traceable lattice constant.

**Is any information missing from individual instances?**
No.

**Are relationships between individual instances made explicit?**
The 8 cross-detector zoo tasks are linked via shared calibrant
(CeO₂/SRM 674b) and shared evaluation pipeline; this relationship is
explicit in `benchmark/detector_zoo/ground_truth.json`.

**Are there recommended data splits?**
The benchmark is evaluation-only; there is no train/val/test split. Tasks
may be filtered by `--category`, `--difficulty`, or task ID list via the
provided harness (`benchmark/eval_harness.py`).

**Are there errors, sources of noise, or redundancies?**
The cross-detector zoo ground truth was computed by a deterministic
GSAS-II refinement pipeline (`benchmark/detector_zoo/refine_v2.py`). The
v1 ground truth (also retained as `ground_truth_v1` in
`ground_truth.json`) was discarded after two upstream pipeline bugs were
diagnosed and fixed in the APEXA tool layer; v2 is the published ground
truth. Per-slice variability from calibrant graininess, diffuse
scattering, and detector noise contributes σ ≈ 0.05–1.5 mÅ on the
aggregated lattice constant; thresholds in the rubric are set above this
floor.

**Is the dataset self-contained?**
The task definitions (`benchmark/benchmark_tasks.json`) and ground truth
(`benchmark/detector_zoo/ground_truth.json`) are self-contained. Running
the evaluation harness requires the APEXA codebase, an Argo Gateway
account (or an OpenAI/Anthropic API key for non-Argo models), and a
working MIDAS v11 install for the data-analysis tasks.

**Does the dataset contain confidential or sensitive data?**
No.

## Collection Process

**How was the data acquired?**
- Standard tasks: authored by the dataset creators based on the
  operational task inventory of a working HEDM beamline at APS.
- Cross-detector zoo: the underlying diffraction images are the public
  CeO₂ calibration examples from the MIDAS distribution
  (`MIDAS/FF_HEDM/Example/Calibration/`).
- Adversarial scenarios: authored by the dataset creators after a domain
  threat model was assembled with beamline operators.

**Who was involved in the data collection?**
Beamline scientists at APS.

**Over what timeframe was the data collected?**
2024–2026.

**Were any ethical review processes conducted?**
Not applicable — the dataset contains no human-subject data.

## Preprocessing / Cleaning / Labeling

**Was any preprocessing done?**
- Detector images were integrated and refined via the standard MIDAS v11
  + GSAS-II 2-stage pipeline using the `apexa_gsas_robust` driver
  (NaN-safe extraction, NIST starting CIF, two-stage refinement omitting
  Atom positions).
- The "v1 vs v2" comparison in the manuscript and in
  `ground_truth.json` retains both passes for reproducibility.

**Was the raw data saved?**
Yes — raw `.tiff`/`.ge1` images live in the MIDAS distribution; per-slice
GSAS-II project files (`.gpx`) and integrated lineouts (`.zarr.zip`) are
archived under `benchmark/detector_zoo/{detector}/`.

## Uses

**Has the dataset been used for any tasks already?**
Yes — the headline evaluation in the accompanying paper:
*APEXA-Bench: A Benchmark for LLM Agents on Synchrotron Beamline
Workflows.*

**Are there tasks for which the dataset should not be used?**
The benchmark is calibrated to one HEDM beamline; per-model leaderboard
numbers will not transfer verbatim to other scientific instruments. The
adversarial safety axis is a *systems-engineering* claim about
deterministic enforcement, not a prompt-engineering ranking.

## Distribution

**Will the dataset be distributed to third parties?**
Yes — released under MIT (code) + CC-BY-4.0 (ground-truth data) at
<https://github.com/AdvancedPhotonSource/APS-Beamline-Assistant>.

**How will it be distributed?**
GitHub repository plus the rendered Croissant JSON-LD descriptor
(`benchmark/croissant.json`).

**When will it be distributed?**
With the publication of the accompanying paper.

**Will the dataset be available under a copyright or other intellectual property license?**
See `benchmark/LICENSE`.

## Maintenance

**Who is supporting/hosting/maintaining the dataset?**
The APEXA team at APS, Argonne National Laboratory.

**How can the owner/curator be contacted?**
Open an issue at
<https://github.com/AdvancedPhotonSource/APS-Beamline-Assistant/issues>.

**Will the dataset be updated?**
Minor updates (additional adversarial scenarios, new detector
geometries) are expected; the schema version is recorded in
`benchmark/benchmark_tasks.json:metadata.schema_version` and
`benchmark/detector_zoo/ground_truth.json:schema_version`. Breaking
changes will bump the major schema version and be announced in the
repository CHANGELOG.

**Will older versions of the dataset continue to be supported?**
Yes — schema-tagged versions remain checked into the git history.
