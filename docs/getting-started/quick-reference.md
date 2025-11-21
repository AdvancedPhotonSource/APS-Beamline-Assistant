# Quick Reference - APEXA Commands

## Most Common Commands

### Calibration
```
APEXA> calibrate using the CeO2 image
APEXA> auto-calibrate with stopping strain 1e-3
```

### Integration
```
APEXA> integrate the .tif file to 1D
APEXA> integrate with dark file dark.ge5
APEXA> batch integrate all .tif files
```

### FF-HEDM
```
APEXA> run FF-HEDM workflow here
APEXA> run FF-HEDM with 32 CPUs
```

### Files
```
APEXA> list files in /data
APEXA> read Parameters.txt
APEXA> find all .ge5 files
```

### Analysis
```
APEXA> identify phases from peaks at 12.5, 18.2, 25.8
APEXA> plot 2D image sample.tif
APEXA> plot radial profile
```

---

## Changes Made

### ✅ Fixed Bugs
1. **Tool name duplication** (midas_midas_auto_calibrate → midas_auto_calibrate)
2. **Removed verbose debug output** ("Extracted tool:", "Arguments:")  
3. **Cleaner console experience** - just shows "→ Tool Name" now

### ✅ Created Documentation
1. **USER_MANUAL.md** - Comprehensive guide with:
   - All capabilities and use cases
   - Example prompts for demonstrations
   - Troubleshooting guide
   - System requirements

2. **QUICK_REFERENCE.md** (this file) - Fast command lookup

---

## For Demo Presentation

### Opening
```
./start_beamline_assistant.sh

# Shows:
# ✓ filesystem (filesystem_server.py)
# ✓ executor (command_executor_server.py)
# ✓ midas (midas_comprehensive_server.py)
```

### Demo Flow

**1. Setup Check**
```
APEXA> what can you do?
APEXA> list files here
```

**2. Calibration Demo**
```
APEXA> calibrate using the CeO2 image
→ Auto Calibrate
[runs AutoCalibrateZarr.py]
✓ Outputs: refined_MIDAS_params.txt
```

**3. Integration Demo**
```
APEXA> integrate sample.tif using refined parameters
→ Integrate 2D To 1D
✓ Outputs: sample.dat
```

**4. Analysis Demo**
```
APEXA> plot the integrated pattern
→ Creates sample_1d.png

APEXA> identify phases from the peaks
→ Phase Identification
Returns: "Detected phases: Ti (α), TiO₂ (rutile)"
```

**5. Batch Processing Demo**
```
APEXA> batch integrate all .tif files in /data
→ Processes 50 files with progress tracking
```

---

## Clean Output Example

**Before (cluttered):**
```
  Extracted tool: midas_midas_auto_calibrate
  Arguments: {'image_file': '/path/...', 'parameters_file': '/path/...'}
→ Midas Midas Auto Calibrate
[11/17/25 14:37:30] WARNING Tool 'midas_midas_auto_calibrate' not listed
```

**After (clean):**
```
→ Auto Calibrate
[processing...]
✓ Calibration complete
```

---

## System Overview

```
User Query
    ↓
Argo-AI (GPT-4o/Claude/Gemini)
    ↓
MCP Client (argo_mcp_client.py)
    ↓
┌─────────┬──────────┬──────────┐
│filesystem│  midas   │ executor │
└─────────┴──────────┴──────────┘
    ↓           ↓          ↓
  Files    MIDAS Tools  Commands
```

**Automatic Features:**
- Context awareness (remembers previous files/directories)
- Proactive suggestions ("Next steps: ...")
- Error prevention (validates before execution)
- Smart caching (faster repeated operations)

---

See **USER_MANUAL.md** for complete documentation.
