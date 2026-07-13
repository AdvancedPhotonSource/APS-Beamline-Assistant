# APEXA - Advanced Photon EXperiment Assistant

AI-powered agentic framework for HEDM data analysis at Argonne National Laboratory's Advanced Photon Source.

---

## Quick Start

### Command-Line Interface (CLI)
```bash
./setup_user.sh                  # One-time setup
./start_beamline_assistant.sh    # Start APEXA CLI
```

### Gradio UI (Recommended for interactive use)
```bash
./start_gradio_ui.sh             # Opens at http://localhost:7860
```

### Web UI
```bash
python web_server.py             # Opens at http://localhost:8000
```

---

## Architecture

```
User (natural language)
         |
    Argo Gateway (GPT-4o / Claude / Gemini)
         |
    OrchestratorAgent (apexa_agents.py)
         |
    +----------------+----------------+------------------+-------------------+
    | Calibration    | Analysis       | Knowledge        | Visualization     |
    | Agent          | Agent          | Agent            | Agent             |
    +-------+--------+-------+--------+--------+---------+--------+----------+
            |                |                 |                  |
            +----------------+-----------------+      MIDAS viewer scripts
                             |
                  +----------+----------+
                  |   core (10 tools)   |
                  |   midas (39 tools)  |
                  |   motor (13 tools)  |
                  +---------------------+
```

### Specialist Agents
| Agent | Routes when |
|---|---|
| CalibrationAgent | calibrate, CeO2, beam center, Lsd, detector distance |
| AnalysisAgent | integrate, HEDM, grain, GSAS-II, refine, workflow (default) |
| KnowledgeAgent | explain, what is, literature, best practice |
| VisualizationAgent | plot, visualize, lineout, caked, heatmap, show |

### MCP Servers (`servers.config`)
| Server | File | Tools |
|---|---|---|
| core | `beamline_core_server.py` | 10 tools: file ops, shell commands, X-ray calculations |
| midas | `midas_comprehensive_server.py` | 39 tools: FF/NF/PF-HEDM, calibration, single + **series/batch** integration, GSAS-II refinement, CIF fetcher, visualization, validation, stress |
| motor | `epics_motor_server.py` | 13 tools: EPICS motor control (read/move/jog/limits) |

### Agent Skills (`.agents/skills/`)
Canonical MIDAS workflow reference — correct v11 flags, scripts, output files:
- `midas-validate` — parameter-file / dataset validation (run first)
- `midas-calibrate` — native `midas_calibrate` workflow (AutoCalibrateZarr fallback)
- `midas-integrate` — single (`midas_integrate_2d_to_1d`), **series** (`midas_integrate_series`, many files, one call, per-frame darks), and GPU-streaming integration
- `midas-hedm` / `midas-ff-hedm` — FF/NF/PF-HEDM full pipeline
- `midas-gsasii` — GSAS-II refinement, live analysis pipeline, CIF fetcher
- `midas-visualize` — MIDAS viewer scripts for lineouts, caked, grains, 3D spots/PF

### Compute dispatch (`docs/COMPUTE_DISPATCH.md`)
APEXA tiers work to **local CPU / local GPU / remote GPU endpoint** by task size and
available hardware. Large batch/HEDM on a CPU-only beamline host can offload to an
ANL GPU (ALCF Polaris via `--machine polaris`, or a lab GPU node) — set
`APEXA_GPU_MACHINE` or `APEXA_GPU_ENDPOINT`. FF/PF workflows accept
`machine`/`n_nodes`/`shard_gpus`; `midas_integrate_series` accepts `compute_target`.

---

## Example Usage

```
APEXA> calibrate the CeO2 image in test1
  -> midas_auto_calibrate
  Refined BC: (809.55, 700.52), Lsd: 641.95 mm

APEXA> integrate CeO2 in test1 using the refined params
  -> midas_integrate_2d_to_1d
  Output: CeO2_000001.tif.analysis.MIDAS_lineout.xy

APEXA> show me the lineout
  -> run_command (plot_lineout_results.py)
  [viewer window opens]

APEXA> convert 61.332 keV to wavelength
  -> xray_calculate
  Wavelength: 0.20215 Angstroms

APEXA> run FF-HEDM workflow on /data/experiment
  -> run_ff_hedm_full_workflow
  Found 2,347 grains
```

---

## In-Session Commands & Runtime Switches

Type these at the `APEXA>` prompt (CLI):

| Command | What it does |
|---|---|
| `model <name>` | Switch the LLM mid-session (e.g. `model gpt55`). `models` lists all. |
| `models` | Show available models with context/output sizes and notes |
| `session new [name]` | Archive the current conversation and start a fresh one (optionally named) |
| `session save [name]` | Save the conversation (named snapshot, or unnamed) |
| `session load <name>` \| `session switch <name>` | Restore a saved session and continue it (new turns append) |
| `session resume` | Reload the most-recent session (auto-saved after every turn) |
| `session list` \| `session summary` | List sessions / show current session info |
| `timing` | Toggle per-response API timing display |
| `ls [path]` | Quick directory listing |
| `help` | Show all commands |
| `quit` | Exit (session is auto-saved; resume with `session resume`) |

**Which model?** `gpt55` (default) or `gpt54` for tool-heavy execution (calibration/
integration — reliable tool-call format). `claudeopus48` for planning/reasoning/writing
(may drift on tool calls, so prefer GPT for running MIDAS tools). See `models` for the
full list (GPT-5 family, Claude Opus 4.6–4.8 / Sonnet 4.5–4.6 / Haiku 4.5, Gemini 3.x).

### Environment switches (set before launch, or in `.env`)

| Variable | Default | Effect |
|---|---|---|
| `APEXA_AGENT_MODE` | `single` | `single` = one persistent reasoning loop, full context across turns (Claude-Code style). `legacy` = keyword-routed specialists + guards. |
| `ARGO_MODEL` | `gpt55` | Default model at startup (overridable in-session with `model`). |
| `APEXA_FORCE_LEGACY_MIDAS` | *(unset)* | `1` = use the fast legacy C++ calibration/integration engine directly (skip the pip attempt). |
| `APEXA_CALIB_TIMEOUT` | `1800` | Calibration subprocess timeout in seconds. |
| `APEXA_SHOW_TIMING` | *(unset)* | `1` = show API response times (same as the `timing` command). |
| `MIDAS_PATH` | *(auto)* | Path to the MIDAS install (auto-detected if unset). |
| `MIDAS_PYTHON` | *(auto)* | Override the conda Python used for legacy MIDAS scripts. |

**Calibration engine:** legacy (v1, search-based) is the default and the robust path.
The differentiable v2 engine is opt-in (`midas_auto_calibrate(..., calibration_engine="v2")`)
— it needs a close initial beam-center guess and can fail on off-center detectors.

---

## Configuration

**User Settings** (`.env`):
```bash
ANL_USERNAME=your_username
ARGO_MODEL=gpt55             # default; or gpt54, claudeopus48, gemini35flash
APEXA_AGENT_MODE=single      # single (default) | legacy
MIDAS_PATH=~/Git/MIDAS       # Optional - auto-detected
# APEXA_FORCE_LEGACY_MIDAS=1 # Optional - fast legacy calibration/integration
```

**Server Configuration** (`servers.config`):
```bash
core:beamline_core_server.py
midas:midas_comprehensive_server.py
```

---

## Requirements

- **Python:** 3.13+ (with [`uv`](https://github.com/astral-sh/uv) package manager)
- **Network:** ANL access for Argo Gateway
- **MIDAS:** v11 with `midas_env` conda environment
- **Memory:** 16+ GB RAM (64+ GB recommended for FF-HEDM)

`uv` handles the virtual environment automatically — users never need to activate it.
`uv sync` installs all ~168 packages in ~1 second. Optional extras: `uv sync --extra extra`

---

## MIDAS Auto-Detection

APEXA searches for MIDAS in this order:
1. `$MIDAS_PATH` environment variable
2. `~/Git/MIDAS`
3. `~/opt/MIDAS`
4. `/home/beams/S*USER/opt/MIDAS` (beamline systems)
5. `~/MIDAS`
6. `/opt/MIDAS`
7. `~/.MIDAS`

---

## Documentation

- **[USER_MANUAL.md](USER_MANUAL.md)** — Complete guide with examples
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Command cheat sheet
- **[WEB_UI_GUIDE.md](WEB_UI_GUIDE.md)** — Browser-based interface
- **[GRADIO_UI_GUIDE.md](GRADIO_UI_GUIDE.md)** — Gradio chat interface
- **[docs/development/architecture.md](docs/development/architecture.md)** — Developer architecture
- **[.agents/skills/](.agents/skills/)** — MIDAS workflow reference (Agent Skills)

---

## Credits

**Development:**
- Pawan Tripathi - Lead Developer
- Advanced Photon Source, Argonne National Laboratory

**Core Dependencies:**
- [MIDAS](https://github.com/marinerhemant/MIDAS) v11 - Hemant Sharma
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [uv](https://github.com/astral-sh/uv) - Package manager
- Argo Gateway - Argonne National Laboratory

---

## License

Copyright (c) 2024-2026 UChicago Argonne, LLC
See [LICENSE](LICENSE) for details.
