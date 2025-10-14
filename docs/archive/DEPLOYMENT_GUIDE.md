# MIDAS Comprehensive MCP Server - Deployment Guide

## Overview

This deployment provides full FF-HEDM, NF-HEDM, PF-HEDM, and analysis capabilities through MCP tools accessible via natural language through the Argo Gateway AI.

## New Files Created

### 1. MCP Server
**File:** `midas_comprehensive_server.py`

Complete MIDAS MCP server with **20+ tools** covering:
- FF-HEDM production workflows
- NF-HEDM reconstruction
- PF-HEDM scanning
- Calibration and grain tracking
- Data management and utilities

### 2. Parameter Templates
**Directory:** `parameter_templates/`

Pre-configured templates for:
- `ff_hedm_template.txt` - Far-Field HEDM
- `nf_hedm_template.txt` - Near-Field HEDM
- `pf_hedm_template.txt` - Point-Focus HEDM
- `calibration_template.txt` - Detector calibration

### 3. Workflow Manager
**File:** `midas_workflows.py`

Orchestrates complex multi-step workflows with progress tracking:
- Asynchronous execution
- Step-by-step progress
- Error handling and recovery
- Workflow logging

### 4. Updated Command Executor
**File:** `command_executor_server.py`

Whitelisted MIDAS commands for direct bash access to MIDAS executables.

---

## Installation

### Step 1: Update System Prompt

Add to `argo_mcp_client.py` system prompt (after line 340):

```python
# In argo_mcp_client.py, in the system_prompt section

🔬 FF-HEDM PRODUCTION TOOLS:
- midas_run_ff_hedm_full_workflow
  Complete FF-HEDM workflow: data conversion, peak search, indexing, refinement
  Args: {"result_folder": "/path", "param_file": "Parameters.txt", "data_file": "data.zip", "n_cpus": 32}

- midas_run_pf_hedm_workflow
  Point-Focus scanning HEDM for 3D orientation maps
  Args: {"param_file": "Parameters.txt", "positions_file": "positions.csv", "n_cpus": 32}

- midas_run_ff_calibration
  Detector calibration with standard reference materials
  Args: {"param_file": "Parameters.txt", "calibrant": "CeO2", "fit_tilt": true}

- midas_run_ff_grain_tracking
  Track grains through deformation/temperature series
  Args: {"grains_files": ["grains1.csv", "grains2.csv"], "tracking_tolerance": 50.0}

🔬 NF-HEDM RECONSTRUCTION TOOLS:
- midas_run_nf_hedm_reconstruction
  Complete NF-HEDM voxel-by-voxel orientation mapping
  Args: {"param_file": "nf_params.txt", "ff_seed_orientations": true, "ff_grains_file": "Grains.csv", "n_cpus": 10}

- midas_convert_nf_to_dream3d
  Convert NF-HEDM .mic to DREAM.3D HDF5 for visualization
  Args: {"nf_mic_file": "Grains.mic", "output_hdf5": "dream3d.h5", "voxel_size": 1.0}

- midas_overlay_ff_nf_results
  Compare FF and NF grain maps
  Args: {"ff_grains_file": "Grains.csv", "nf_mic_file": "Grains.mic"}

🔬 ADVANCED ANALYSIS TOOLS:
- midas_calculate_misorientation
  Calculate misorientation between grain pairs
  Args: {"grains_file": "Grains.csv", "grain_id_1": 1, "grain_id_2": 2, "space_group": 225}

- midas_run_forward_simulation
  Simulate diffraction from known microstructure
  Args: {"input_grains_file": "input.csv", "param_file": "Parameters.txt", "output_prefix": "sim"}

- midas_extract_grain_centroids
  Extract grain centroids from NF reconstruction
  Args: {"nf_mic_file": "Grains.mic", "min_grain_size": 100}

🔬 DATA MANAGEMENT:
- midas_batch_convert_ge_to_tiff
  Batch convert GE detector files to TIFF
  Args: {"ge_folder": "/data/ge", "output_folder": "/data/tiff", "parallel": true}

- midas_create_parameter_file
  Generate MIDAS parameter file programmatically
  Args: {"lattice_constants": [4.08, 4.08, 4.08, 90, 90, 90], "space_group": 225, "detector_distance": 1000000, "beam_center": [1022, 1022], "wavelength": 0.22291, "omega_step": -0.25}

- midas_validate_installation
  Validate MIDAS installation and dependencies
  Args: {"midas_path": "~/opt/MIDAS"}

- midas_get_workflow_status
  Check running workflow status
  Args: {"result_folder": "/path/to/results", "workflow_type": "ff"}

🔬 BASIC ANALYSIS (LEGACY):
- midas_detect_diffraction_rings
  Detect rings in 2D patterns
  Args: {"image_path": "/path/image.tif"}

- midas_identify_crystalline_phases
  Identify phases from peak positions
  Args: {"peak_positions": [12.5, 18.2, 25.8], "tolerance": 0.1}
```

### Step 2: Start with New Server

Replace `fastmcp_midas_server.py` with `midas_comprehensive_server.py`:

```bash
# Old command:
# uv run argo_mcp_client.py midas:fastmcp_midas_server.py executor:command_executor_server.py filesystem:filesystem_server.py

# New command:
uv run argo_mcp_client.py \
  midas:midas_comprehensive_server.py \
  executor:command_executor_server.py \
  filesystem:filesystem_server.py
```

---

## Usage Examples

### Example 1: Complete FF-HEDM Analysis

```
Beamline> Run FF-HEDM analysis on /beamline/data/steel_RT with 64 CPUs using parameters in /beamline/params/steel_params.txt

AI will execute:
  Tool: midas_run_ff_hedm_full_workflow
  Args:
    - result_folder: /beamline/results/steel_RT
    - param_file: /beamline/params/steel_params.txt
    - data_file: /beamline/data/steel_RT/raw_data.zip
    - n_cpus: 64
    - start_layer: 1
    - end_layer: 1
    - do_peak_search: true

Result: Grains.csv with grain orientations, positions, strains
```

### Example 2: Combined FF-NF Workflow

```
Beamline> First run FF-HEDM on the steel sample, then use those results to seed NF-HEDM reconstruction for detailed grain mapping

AI will execute:
  1. midas_run_ff_hedm_full_workflow (FF analysis)
  2. midas_run_nf_hedm_reconstruction (NF with FF seeds)
  3. midas_overlay_ff_nf_results (validation)

Result: High-resolution 3D grain map with FF validation
```

### Example 3: Detector Calibration

```
Beamline> Calibrate the detector using CeO2 standard

AI will execute:
  Tool: midas_run_ff_calibration
  Args:
    - param_file: Parameters.txt
    - calibrant: CeO2
    - fit_tilt: true

Result: Calibrated detector parameters (Lsd, BC, tilt angles)
```

### Example 4: Grain Tracking Through Deformation

```
Beamline> Track grains through my in-situ tensile test from 0% to 10% strain

AI will execute:
  Tool: midas_run_ff_grain_tracking
  Args:
    - grains_files: [grains_0pct.csv, grains_2pct.csv, grains_5pct.csv, grains_10pct.csv]
    - tracking_tolerance: 50.0

Result: Tracked grain evolution with strain history
```

### Example 5: Create Parameter File

```
Beamline> Create FF-HEDM parameter file for FCC aluminum with lattice parameter 4.05 Å, detector at 1 meter, beam center at pixel 1024,1024, wavelength 0.173 Å

AI will execute:
  Tool: midas_create_parameter_file
  Args:
    - lattice_constants: [4.05, 4.05, 4.05, 90, 90, 90]
    - space_group: 225
    - detector_distance: 1000000
    - beam_center: [1024, 1024]
    - wavelength: 0.173
    - omega_step: -0.25

Result: Parameters.txt ready for analysis
```

### Example 6: Misorientation Analysis

```
Beamline> Calculate the misorientation between grain 42 and grain 87 in my steel sample

AI will execute:
  Tool: midas_calculate_misorientation
  Args:
    - grains_file: Grains.csv
    - grain_id_1: 42
    - grain_id_2: 87
    - space_group: 225

Result: Misorientation angle, axis, and boundary type (e.g., Σ3 twin)
```

### Example 7: Data Format Conversion

```
Beamline> Convert all GE detector files in /beamline/raw_data to TIFF format

AI will execute:
  Tool: midas_batch_convert_ge_to_tiff
  Args:
    - ge_folder: /beamline/raw_data
    - output_folder: /beamline/tiff_data
    - parallel: true
    - n_processes: 8

Result: TIFF images ready for MIDAS processing
```

---

## Environment Setup

### Required Environment Variables

Add to `.env` file:

```bash
# MIDAS installation path
MIDAS_PATH=~/opt/MIDAS

# Argo Gateway credentials
ANL_USERNAME=your_anl_username
ARGO_MODEL=gpt4o
```

### Python Dependencies

Ensure these are installed (already in your environment):

```bash
# Core
mcp httpx python-dotenv numpy scipy matplotlib pandas

# MIDAS-specific
fabio pyFAI scikit-image h5py zarr numcodecs

# Parallel processing
parsl numba multiprocess

# Visualization
plotly seaborn
```

---

## Tool Capabilities Matrix

| Workflow | Tools | Input | Output | Time |
|----------|-------|-------|--------|------|
| **FF-HEDM** | run_ff_hedm_full_workflow | Raw images, params | Grains.csv | 30-120 min |
| **PF-HEDM** | run_pf_hedm_workflow | Scanning data, positions | 3D orientation map | 60-180 min |
| **NF-HEDM** | run_nf_hedm_reconstruction | Images, FF seeds | Grains.mic | 120-360 min |
| **Calibration** | run_ff_calibration | Calibrant images | Detector params | 5-15 min |
| **Tracking** | run_ff_grain_tracking | Multiple Grains.csv | Tracked grains | 1-5 min |
| **Misorientation** | calculate_misorientation | Grains.csv, IDs | Angle, axis | <1 min |
| **Simulation** | run_forward_simulation | Grain structure | Synthetic data | 10-30 min |
| **Conversion** | batch_convert_ge_to_tiff | GE files | TIFF images | 5-30 min |

---

## Troubleshooting

### Issue: MIDAS executables not found

**Solution:**
```bash
Beamline> Validate MIDAS installation

# Will check for:
# - All executables in ~/opt/MIDAS/bin/
# - Python workflow scripts
# - Required dependencies
```

### Issue: Workflow fails at specific step

**Solution:**
```bash
Beamline> Check workflow status in /path/to/results

# Returns:
# - Current workflow status
# - Last completed step
# - Error messages
# - Log file locations
```

### Issue: Parameter file errors

**Solution:**
Use parameter templates as starting point:
```bash
cp parameter_templates/ff_hedm_template.txt ./Parameters.txt
# Edit Parameters.txt with your experimental values
```

---

## Performance Optimization

### CPU Allocation

| Dataset Size | Recommended CPUs | Expected Time |
|--------------|------------------|---------------|
| Small (<1GB) | 8-16 | 30-60 min |
| Medium (1-10GB) | 32-64 | 60-120 min |
| Large (>10GB) | 64-128 | 120-240 min |

### Memory Requirements

- **FF-HEDM:** 2-4 GB per CPU
- **NF-HEDM:** 4-8 GB per CPU
- **Peak Search:** 8-16 GB total
- **Indexing:** 16-32 GB total

### Parallelization

- **FF-HEDM:** Parallelizes over layers and peaks
- **NF-HEDM:** Parallelizes over voxels
- **PF-HEDM:** Parallelizes over scan positions

---

## Integration with Existing Beamline Workflows

### Data Pipeline

```
1. Data Collection
   ↓
2. GE → TIFF Conversion (midas_batch_convert_ge_to_tiff)
   ↓
3. Quick Quality Check (midas_detect_diffraction_rings)
   ↓
4. Detector Calibration (if needed) (midas_run_ff_calibration)
   ↓
5. FF-HEDM Analysis (midas_run_ff_hedm_full_workflow)
   ↓
6. NF-HEDM Reconstruction (midas_run_nf_hedm_reconstruction)
   ↓
7. Visualization (midas_convert_nf_to_dream3d)
   ↓
8. Advanced Analysis (tracking, misorientation, etc.)
```

### Automated Workflows

The workflow manager (`midas_workflows.py`) can be imported for automated processing:

```python
from midas_workflows import MIDASWorkflowManager
import asyncio

async def process_beamline_data():
    manager = MIDASWorkflowManager()

    config = {
        "result_folder": "./results",
        "param_file": "./Parameters.txt",
        "data_file": "./raw_data.zip",
        "n_cpus": 64
    }

    result = await manager.run_ff_workflow(config)
    return result

asyncio.run(process_beamline_data())
```

---

## Next Steps

1. **Test Installation**
   ```bash
   Beamline> Validate MIDAS installation
   ```

2. **Run Example Workflow**
   Use FF-HEDM example from MIDAS:
   ```bash
   Beamline> Run FF-HEDM simulation in ~/opt/MIDAS/FF_HEDM/Example with 20 CPUs
   ```

3. **Process Real Data**
   Start with calibration, then run full workflow

4. **Integrate with Automation**
   Use workflow manager for batch processing

---

## Support

For issues or questions:
- Check MIDAS documentation: `~/opt/MIDAS/README.md`
- Review workflow logs: `result_folder/workflow_log.json`
- Validate installation: Use `midas_validate_installation` tool
- Contact: Hemant Sharma (hsharma@anl.gov)

---

## Summary

You now have **comprehensive MIDAS capabilities** accessible through natural language:

✅ **20+ MCP tools** covering all HEDM workflows
✅ **Parameter templates** for quick setup
✅ **Workflow orchestration** with progress tracking
✅ **Command whitelisting** for direct MIDAS executable access
✅ **AI-powered** analysis through Argo Gateway

The system transforms complex MIDAS workflows into simple natural language queries at the beamline!
