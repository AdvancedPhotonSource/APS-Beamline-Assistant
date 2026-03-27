# Quick Reference - APEXA Commands

## Calibration (CalibrationAgent)
```
APEXA> calibrate the CeO2 image in test1
APEXA> calibrate using LaB6_650mm.tif with 40 iterations
APEXA> convert 61.332 keV to wavelength
APEXA> what's the d-spacing for CeO2 (111)?
APEXA> list common calibrants
```

## Integration (AnalysisAgent)
```
APEXA> integrate CeO2_000001.tif using refined params
APEXA> integrate with dark file subtraction
APEXA> batch integrate all .tif files in /data with 8 CPUs
```

## FF-HEDM (AnalysisAgent)
```
APEXA> run FF-HEDM workflow on /data/experiment
APEXA> run FF-HEDM with 32 CPUs
```

## NF-HEDM (AnalysisAgent)
```
APEXA> run NF-HEDM reconstruction with FF seed orientations
APEXA> reconstruct microstructure with 10 CPUs
```

## Visualization (VisualizationAgent)
```
APEXA> show me the lineout for CeO2 in test1
APEXA> plot the caked output from test1/integration
APEXA> show calibration results for test1
APEXA> view the 2D diffraction image
APEXA> compare lineouts with ideal ring positions
```

## Files
```
APEXA> list files in /data
APEXA> read the Parameters.txt file
APEXA> what's in the integration folder?
```

## Knowledge (KnowledgeAgent)
```
APEXA> what is FF-HEDM?
APEXA> explain Bragg's law
APEXA> what are typical parameters for steel?
APEXA> get material properties for Ti-6Al-4V
```

## X-ray Calculations
```
APEXA> convert 61.332 keV to wavelength
APEXA> calculate d-spacing for 2-theta 10.5 degrees at 0.2066 angstroms
APEXA> calculate strain for measured d=3.155 and reference d=3.124
```

---

## CLI Commands (not sent to AI)
| Command | Description |
|---------|-------------|
| `models` | Show available AI models |
| `model <name>` | Switch model (gpt4o, claudesonnet4, gemini25pro) |
| `tools` | List all analysis tools |
| `servers` | Show connected servers |
| `ls <path>` | List directory |
| `clear` | Clear conversation history |
| `help` | Show help |
| `quit` | Exit |

---

## System Overview

```
User (natural language)
         |
    Argo Gateway (GPT-4o / Claude / Gemini)
         |
    OrchestratorAgent
         |
    +----------------+----------+-----------+----------------+
    | Calibration    | Analysis | Knowledge | Visualization  |
    +----------------+----------+-----------+----------------+
                         |
              +----------+----------+
              |   core (9 tools)    |
              |   midas (21 tools)  |
              +---------------------+
```

**Servers:** `core` (beamline_core_server.py) + `midas` (midas_comprehensive_server.py)

See **USER_MANUAL.md** for complete documentation.
