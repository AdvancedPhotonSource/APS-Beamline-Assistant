# Changelog

All notable changes to APEXA (Advanced Photon EXperiment Assistant) are
documented in this file. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with semantic
versioning. Until v1.0.0, breaking changes may land in minor versions.

## [Unreleased]

### Added — MIDAS Mar-2026 package release integration
- Bumped the MIDAS pin to `midas-suite[ff,pdf,defect,dfxm,xaf,ultrafast,grain-odf,pf-odf,pink]>=0.4.0`
  (unified `midas-pipeline` 0.6.1, `midas-calibrate-v2` 0.5.2, and the full v2/aux stack).
- **8 new MCP tools** wiring MIDAS's brand-new v0.1.0 capability packages
  (MIDAS server 42→50 tools):
  - `compute_pair_distribution` — midas-pdf, **real**: integrated I(Q)→G(r) Faber-Ziman PDF.
  - `analyze_grain_defects` — midas-defect, **real**: dislocation rods / asterism /
    polytype / defect inventory from FF-HEDM diffuse scattering.
  - `fit_grain_odf` — midas-grain-odf, **real**: per-grain orientation distribution fit.
  - `simulate_2d_diffraction` — midas-2d, synthetic coherent/ultrafast forward model.
  - `design_xaf_experiment` — midas-xaf, synthetic anvil-cell (cross-axis) design.
  - `simulate_dfxm_image` — midas-dfxm, synthetic DFXM forward image.
  - `invert_pf_grain_odf` — midas-pf-odf, capability present (real I/O deferred upstream).
  - `invert_pink_beam` — midas-pink, capability present (real I/O deferred upstream).
- New `_capability_runner.py` sidecar runs all 8 under the `.venv` interpreter with a
  clean env (the pip torch stack breaks under C++ DYLD/LD injection); synthetic/deferred
  tools carry `"mode"` + `"real_data_supported"` flags and never fabricate real results.
- `recommend_workflow` now maps exposed data → the correct FF/NF/PF workflow, cites the
  relevant Agent Skill, distinguishes powder vs single-crystal calibrants (Au → NF
  beam-position, not FF powder geometry), and surfaces the in-plane `tx` post-reconstruction
  calibration step. Modality is now decided **by reading the parameter file's keys**
  (`nDistances`/`MicFile*`/`MinConfidence` → NF; `RingThresh`/`OverAllRingToIndex` → FF;
  `nScans > 1` → PF) — the authoritative, name-independent signal, ranked above the
  path/directory-name heuristics. A `.txt` is recognized as a param file by its contents
  too, so an oddly-named file (e.g. `ps_au.txt`) is no longer missed.
- `midas_auto_calibrate` (v2 engine) now returns an honest error for unsupported
  calibrants instead of silently substituting CeO2.

### Experimental
- Native MIDAS dispatch (`apexa_midas_native.py`) behind
  `APEXA_USE_NATIVE_MIDAS=1`; the subprocess path remains the default.

## [0.1.0] — 2026-05-18

First public release. APEXA has been deployed at the Advanced Photon Source
and exercised on real beamline tasks at Sectors 1-ID and 20-ID. This release
captures the state of the framework as evaluated in the accompanying
manuscripts in preparation.

### Agent layer

- Hierarchical multi-agent architecture: an `OrchestratorAgent` routes
  natural-language queries to one of five specialist agents based on a
  keyword-scoring function over 143 domain terms.
- Specialist agents with task-tuned temperature: `MotorAgent` (T=0.2),
  `CalibrationAgent` (T=0.3), `VisualizationAgent` (T=0.3),
  `AnalysisAgent` (T=0.5), `KnowledgeAgent` (T=0.6).
- Text-based tool-calling protocol (`TOOL_CALL:` / `ARGUMENTS:` regex
  parser) that works through institutional gateways which strip
  vendor-native tool-call structures. Co-exists with native API
  `tool_calls` when available.
- DSPy in-context router (`apexa_dspy_router.py`) as an alternative to
  keyword routing; rescues smaller models that collapse under keyword
  routing on multi-step tasks.
- AutoGen multi-agent baseline (`apexa_autogen_baseline.py`) for
  cross-orchestrator comparison.
- Conversation-history selection: anchor (first message) + last N=8 turns,
  bounding token cost without losing the original user query.

### MCP servers and tools (57 total)

- **Core server** (`beamline_core_server.py`, 9 tools): filesystem ops,
  shell commands, X-ray energy↔wavelength calculations, beamline-parameter
  validation, calibrant catalog.
- **MIDAS server** (`midas_comprehensive_server.py`, 35 tools): full FF/NF/
  PF-HEDM workflows wrapping MIDAS v11, AutoCalibrateZarr-based calibration,
  2D→1D azimuthal integration, Materials Project CIF fetcher (150 000+
  structures), MIDAS viewer integration, knowledge-base queries, grain
  stress analysis, slip-system analysis.
- **EPICS motor server** (`epics_motor_server.py`, 13 tools): real-time
  motor control via channel access — read position, absolute/relative
  moves, jog, tweak, home, stop, limits.
- Tool registry built once at connection time (O(1) name→server lookup),
  replacing the O(n × servers) live-RPC waterfall of naive multi-server
  setups.

### Safety (tool-layer, not prompt-layer)

- Five-stage safety pipeline enforced inside the motor server before any
  `caput` is issued:
  1. hardware limit-switch verification (HLS/LLS),
  2. soft-limit validation (HLM/LLM),
  3. large-move confirmation guard (>50% of travel range requires
     `confirm_large_move=True`),
  4. velocity bound check against VMAX,
  5. jog-duration cap (30 s) to prevent runaway motion.
- `stop_motor` is the sole exception — bypasses all checks so motion can
  always be halted.
- Auto-discovery of motor count from the IOC; overridable via
  `EPICS_MOTOR_COUNT` env var.

### Knowledge base / RAG

- Retrieval-augmented generation over curated HEDM literature,
  crystallography textbooks, and beamline documentation.
- Nomic embeddings with cosine similarity; embedder configurable via
  `APEXA_EMBED_MODEL`.
- Citation-aware chunking with `.bib` sidecars; `query_hedm_knowledge`
  returns excerpts with citations, `get_bibtex` returns the entry.
- Zotero sync tool for keeping the index fresh against an external
  reference library.
- Pre-warmed at server startup, not on first query.

### Agent Skills

- `.agents/skills/` directory: version-locked procedural-knowledge documents
  shipping with the framework. Covers `midas-calibrate`,
  `midas-integrate`, `midas-hedm`, `midas-ffpipeline`, `midas-visualize`,
  `midas-gsasii`, and `motor-control`. Skills track the installed software
  (MIDAS v11, EPICS), not the LLM's training corpus.

### Interfaces

- CLI (`argo_mcp_client.py`) with tab completion, themed markdown rendering,
  citation highlighting, and `prompt_toolkit` input.
- Web server (`web_server.py`) with React frontend and WebSocket streaming
  to a real-time visualization panel.
- Both interfaces share a single `run_query()` entry point and a single
  orchestrator instance.

### Model and gateway support

- Argo Gateway integration with retry logic for 502/503/429 responses and
  user-friendly error surfaces.
- Verified across four frontier models: GPT-5-mini, GPT-5.4,
  Claude Opus 4.7, Gemini 2.5 Pro.
- Model-specific parameter handling (o-series and GPT-5 use
  `max_completion_tokens` and reject `temperature`; Haiku/Sonnet variants
  reject `top_p`).

### Visualization

- Unified MIDAS visualization API exposed as a single MCP tool
  (`run_midas_viewer`).
- Lineout plots, integrator peak overlays, calibrant residuals, caked
  patterns, live viewer.

### MIDAS v11 adaptation

- Integrator flag passthrough so callers can drive newer `integrator.py`
  options without server changes.
- `ImTransOpt` resolver for MIDAS image-transform conventions.
- Support for v11 GPU executables (`IndexerGPU`, `FitPosOrStrainsGPU`) and
  the new HDF5 output layout.
- Engine selection (GSAS-II / MAUD) through a single tool surface.

### Data and infrastructure

- Centralized X-ray unit conversions (`apexa_units.py`): xrayutilities
  backend with a CODATA-2018 fallback for subprocess environments. Replaces
  scattered `12.398 / E` literals.
- Interaction logger (`interaction_logger.py`): structured JSONL capturing
  agent, tool chain, per-call timings, and success flags. Foundation for
  continual-learning evaluation.
- Two distinct MIDAS environments (`get_midas_env` for C++ binaries vs
  `get_midas_python_env` for Python helpers) to avoid h5py/libhdf5 symbol
  mismatch.

### Hardening

- Shell-injection fixes in `run_command` (`shell=True` → `shlex` + list
  args).
- Symlink resolution across file-path handling tools.
- Anti-hallucination guards: never reference non-existent flags
  (e.g. pyFAI/`.poni` paths), block obviously wrong `run_command` patterns
  for GSAS-II refinement, soften model output when it strays from
  ground-truth file layouts.
- Path-confusion guard when the user points at a non-CWD directory.

### Documentation

- `README.md`, `USER_MANUAL.md`, `QUICK_REFERENCE.md`, `WEB_UI_GUIDE.md`,
  `CLAUDE.md` (codebase context for AI assistants),
  `tutorials-demo/APEXA_Command_Reference.md`.

### Dependencies and environment

- Python 3.13+, managed via `uv`.
- Optional extras (`uv sync --extra extra`): pyfai, vtk, seaborn,
  xrayutilities.
- `.env` configuration: `ANL_USERNAME`, `ARGO_MODEL`, `MIDAS_PATH`,
  `EPICS_MOTOR_COUNT`, `APEXA_EMBED_MODEL`, `APEXA_SHOW_TIMING`,
  `APEXA_USE_NATIVE_MIDAS`.
