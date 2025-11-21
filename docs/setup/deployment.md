# APEXA Beamline Deployment Guide

Complete guide for deploying the updated APEXA with robust MIDAS auto-calibration on S1IDUSER beamline.

---

## What's New in pawan-modular-v2

Critical fixes and enhancements for MIDAS workflows:

1. ✅ **Fixed auto-calibration blocking** - No longer requires pyFAI in MCP server environment
2. ✅ **All 15 official parameters** - stopping_strain, mult_factor, eta_bin_size, etc.
3. ✅ **Convergence metrics** - Reports iterations, final strain, excluded rings
4. ✅ **Diagnostic output** - Clear startup messages showing MIDAS status
5. ✅ **Enhanced error messages** - Helpful troubleshooting information

**Key Commits:**
- [`30c530d`] **Latest:** Comprehensive environment handling for all MIDAS tools
- [`e404934`] Auto-detect MIDAS Python (conda midas_env)
- [`28d2db3`] Fix MIDAS path priority (check ~/opt/MIDAS first)
- [`e47b1ce`] Robust auto-calibration based on official manual
- [`1b921fb`] Complete MIDAS workflows reference

---

## Quick Start

```bash
# 1. Update code
cd ~/CAI/git/APS-Beamline-Assistant
git fetch && git checkout pawan-modular-v2 && git pull

# 2. Restart APEXA (this will restart all servers)
./start_beamline_assistant.sh
```

**Expected output:**
```
Found MIDAS installation at: /home/beams/S1IDUSER/opt/MIDAS
✓ AutoCalibrateZarr.py found at /home/beams/S1IDUSER/opt/MIDAS/utils/AutoCalibrateZarr.py
Using MIDAS conda environment: /home/beams/S1IDUSER/miniconda3/envs/midas_env/bin/python
✓ MIDAS scientific dependencies available
```

**Key Features:**
- **Automatic environment detection** - No manual conda activation needed!
- **UV + Conda separation** - MCP server runs in UV, MIDAS tools use conda midas_env
- **Complete path handling** - Python scripts use conda, C++ binaries get proper LD_LIBRARY_PATH

---

## Environment Architecture

APEXA uses a **dual-environment strategy** that automatically handles UV and conda:

```
┌─────────────────────────────────────────────────────────────┐
│ UV Environment (beamline-assistant)                         │
│ - Runs MCP servers (filesystem, executor, midas)            │
│ - Runs APEXA client                                          │
│ - No MIDAS Python dependencies needed                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Automatically detects & uses
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Conda midas_env                                              │
│ - Python with zarr, diplib, numpy, scipy, etc.              │
│ - Used by AutoCalibrateZarr.py and all MIDAS Python scripts │
│ - Shared libraries for C++ binaries (Integrator, etc.)      │
└─────────────────────────────────────────────────────────────┘
```

**How it works:**
1. `find_midas_python()` - Auto-detects conda midas_env Python
2. `get_midas_env()` - Sets LD_LIBRARY_PATH for C++ binaries
3. All MIDAS tools use correct environment automatically

**No manual activation needed!**

---

## Detailed Deployment

### Step 1: Verify MIDAS Installation

MIDAS must have `AutoCalibrateZarr.py` for auto-calibration:

```bash
# Check common locations
ls -la /home/beams/S1IDUSER/.MIDAS/utils/AutoCalibrateZarr.py
ls -la /home/beams/S1IDUSER/opt/MIDAS/utils/AutoCalibrateZarr.py

# Required structure
MIDAS/
├── utils/AutoCalibrateZarr.py          ← CRITICAL
├── FF_HEDM/bin/CalibrantOMP            ← Used by AutoCalibrateZarr.py
└── FF_HEDM/bin/Integrator              ← For 2D→1D integration
```

**If missing**, update or install MIDAS:

```bash
# Option 1: Update existing
cd /home/beams/S1IDUSER/.MIDAS && git pull

# Option 2: Fresh install
cd /home/beams/S1IDUSER/opt
git clone https://github.com/marinerhemant/MIDAS.git
export MIDAS_PATH=/home/beams/S1IDUSER/opt/MIDAS
```

### Step 2: Install MIDAS Python Dependencies

AutoCalibrateZarr.py requires: `zarr`, `diplib`, `numpy`, `scipy`, `matplotlib`, `pandas`, `plotly`, `h5py`, `numba`

```bash
# Test dependencies
python -c "import zarr, diplib, numpy, scipy; print('OK')"

# If missing, install
cd /home/beams/S1IDUSER/opt/MIDAS
conda env create -f environment.yml
conda activate midas

# Or with pip
pip install zarr diplib numpy scipy matplotlib pandas plotly h5py numba scikit-image fabio
```

### Step 3: Update APEXA Code

```bash
cd /home/beams/S1IDUSER/beamline-assistant-dev
git fetch origin
git checkout pawan-modular-v2
git pull origin pawan-modular-v2

# Verify commits
git log --oneline -5
# Should show: 1b921fb, e47b1ce, ac60af6, f3caf16
```

### Step 4: Restart MCP Server

```bash
# Kill old server
ps aux | grep midas_comprehensive_server
pkill -f midas_comprehensive_server

# Start new server
cd /home/beams/S1IDUSER/beamline-assistant-dev
uv run python midas_comprehensive_server.py &

# Check logs for diagnostic output
tail -f mcp_server.log
```

**Expected output:**
```
Found MIDAS installation at: /home/beams/S1IDUSER/opt/MIDAS
✓ AutoCalibrateZarr.py found at /home/beams/S1IDUSER/opt/MIDAS/utils/AutoCalibrateZarr.py
✓ MIDAS scientific dependencies available
```

**Warning sign:**
```
⚠ AutoCalibrateZarr.py NOT found at ...
```

### Step 5: Restart APEXA Client

```bash
pkill -f argo_mcp_client
cd /home/beams/S1IDUSER/beamline-assistant-dev
uv run python argo_mcp_client.py
```

---

## Verification Tests

### Test 1: Auto-Calibrate CeO2

In APEXA:
```
autocalibrate the CeO2_650mm_61p332keV_2DFocused_0p1s_att200_004018.tif file in /home/beams/S1IDUSER/CAI/git/demo_data with energy=61.332keV, pixel_size=172 microns
```

**Expected:**
```
✓ Auto-calibration completed successfully!

Refined Parameters:
  Beam Center: (1040.23, 1030.45) pixels
  Distance (Lsd): 650.12 mm
  Tilts: tx=0.000123, ty=-0.000045, tz=0.000089 rad

Convergence:
  Iterations: 5
  Final Mean Strain: 0.000038
  Target Strain: 0.000040
  Status: CONVERGED ✓

Output Files:
  • refined_MIDAS_params.txt
  • autocal.log
```

### Test 2: Integration

```
integrate the CeO2 image to 1D using refined_MIDAS_params.txt
```

Should produce `_1d.dat` file.

---

## Troubleshooting

### Issue: APEXA Using pyFAI Instead of MIDAS

**Symptom:** APEXA tries random shell commands, mentions pyFAI

**Cause:** MCP server not restarted with new code

**Fix:**
```bash
# Force restart
pkill -9 -f midas_comprehensive_server
cd /home/beams/S1IDUSER/beamline-assistant-dev
git branch  # Verify on pawan-modular-v2
uv run python midas_comprehensive_server.py &
```

### Issue: "AutoCalibrateZarr.py not found"

**Fix:**
```bash
# Find MIDAS
find /home/beams/S1IDUSER -name "AutoCalibrateZarr.py" 2>/dev/null

# Set MIDAS_PATH if not in standard location
export MIDAS_PATH=/path/to/MIDAS
# Add to ~/.bashrc to persist
```

### Issue: "ModuleNotFoundError: zarr"

**Fix:**
```bash
# Check which Python is used
head -1 /home/beams/S1IDUSER/opt/MIDAS/utils/AutoCalibrateZarr.py

# Install for that Python
python3 -m pip install zarr diplib numpy scipy matplotlib pandas plotly h5py numba
```

### Issue: Calibration NOT CONVERGED

**Solutions:**

1. Relax tolerance:
   ```
   autocalibrate with stopping_strain=0.001
   ```

2. Better initial guess:
   ```
   autocalibrate with lsd_guess=650000, bc_x_guess=1024, bc_y_guess=1024
   ```

3. Image transform:
   ```
   autocalibrate with image_transform="2"
   # "0"=none, "1"=flip LR, "2"=flip UD, "3"=transpose
   ```

4. Save diagnostics:
   ```
   autocalibrate with save_plots_hdf="diag.h5"
   ```

---

## APEXA Command Reference

**Basic:**
```
autocalibrate the CeO2.tif file with energy=61.332keV, pixel_size=172
```

**With options:**
```
autocalibrate with lsd_guess=650000, stopping_strain=0.0001, mult_factor=2.5, image_transform="2"
```

**Full advanced:**
```
autocalibrate with lsd_guess=650000, bc_x_guess=1024, bc_y_guess=1024, stopping_strain=0.00004, mult_factor=2.5, first_ring_nr=1, eta_bin_size=5.0, save_plots_hdf="diagnostics.h5", image_transform="2"
```

**Integration:**
```
integrate the sample.tif image using refined_MIDAS_params.txt
```

---

## Parameter File Template

Create `Parameters.txt`:

```txt
# Material (CeO2 example)
SpaceGroup 225
LatticeParameter 5.411 5.411 5.411 90 90 90
Wavelength 0.2021

# Detector
px 172.0
NrPixels 2048 2048

# Initial guesses (will be refined)
Lsd 650000
BC 1024 1024

# Processing
SkipFrame 0
OmegaStep 0.25
```

---

## Output Files

**Input:**
- `CeO2_*.tif` - Calibrant image
- `Parameters.txt` - Initial parameters

**Output:**
- `refined_MIDAS_params.txt` ⭐ **Main result** (use for integration)
- `autocal.log` - Iteration history with convergence metrics
- `calibrant_screen_out.csv` - Raw CalibrantOMP output
- `*.zarr.zip` - Converted data

---

## Version Info

- **Branch:** `pawan-modular-v2`
- **Latest Commit:** [`1b921fb`]
- **Date:** 2025-10-23
- **Based on:** https://github.com/marinerhemant/MIDAS/blob/master/manuals/FF_autocalibrate.md
