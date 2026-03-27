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
                  |   core (9 tools)    |
                  |   midas (21 tools)  |
                  +---------------------+
```

### Specialist Agents
| Agent | Routes when |
|---|---|
| CalibrationAgent | calibrate, CeO2, beam center, Lsd, detector distance |
| AnalysisAgent | integrate, HEDM, grain, workflow, batch (default) |
| KnowledgeAgent | explain, what is, literature, best practice |
| VisualizationAgent | plot, visualize, lineout, caked, heatmap, show |

### MCP Servers (`servers.config`)
| Server | File | Tools |
|---|---|---|
| core | `beamline_core_server.py` | 9 tools: file ops, shell commands, X-ray calculations |
| midas | `midas_comprehensive_server.py` | 21 tools: FF/NF/PF-HEDM, calibration, integration |

### Agent Skills (`.agents/skills/`)
Canonical MIDAS workflow reference — correct v11 flags, scripts, output files:
- `midas-calibrate` — AutoCalibrateZarr.py workflow
- `midas-integrate` — integrator.py (CPU) and integrator_batch_process.py (GPU)
- `midas-hedm` — FF/NF/PF-HEDM full pipeline
- `midas-visualize` — MIDAS viewer scripts for lineouts, caked, grains

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

## Configuration

**User Settings** (`.env`):
```bash
ANL_USERNAME=your_username
ARGO_MODEL=gpt4o              # or claudesonnet4, gemini25pro
MIDAS_PATH=~/Git/MIDAS        # Optional - auto-detected
```

**Server Configuration** (`servers.config`):
```bash
core:beamline_core_server.py
midas:midas_comprehensive_server.py
```

---

## Requirements

- **Python:** 3.13+ (with `uv` package manager)
- **Network:** ANL access for Argo Gateway
- **MIDAS:** v11 with `midas_env` conda environment
- **Memory:** 16+ GB RAM (64+ GB recommended for FF-HEDM)

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
