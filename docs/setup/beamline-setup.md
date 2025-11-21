# APEXA Beamline Setup Instructions

## Current Issue on S1IDUSER Beamline

APEXA's auto-calibration feature is not working because the MIDAS installation at `/home/beams/S1IDUSER/.MIDAS` does not contain the `AutoCalibrateZarr.py` script.

## Diagnostic Information

When the MCP server starts, you should see one of these messages in the logs:

**✓ Working installation:**
```
Found MIDAS installation at: /home/beams/S1IDUSER/.MIDAS
✓ AutoCalibrateZarr.py found at /home/beams/S1IDUSER/.MIDAS/utils/AutoCalibrateZarr.py
```

**✗ Missing AutoCalibrateZarr.py:**
```
Found MIDAS installation at: /home/beams/S1IDUSER/.MIDAS
⚠ AutoCalibrateZarr.py NOT found at /home/beams/S1IDUSER/.MIDAS/utils/AutoCalibrateZarr.py
  Auto-calibration will not work until MIDAS is properly installed
```

## Required MIDAS Structure

Your MIDAS installation should have this structure:

```
/home/beams/S1IDUSER/.MIDAS/
├── utils/
│   ├── AutoCalibrateZarr.py          ← REQUIRED for auto-calibration
│   ├── deprecated_AutoCalibrate.py
│   └── ... other Python scripts
├── FF_HEDM/
│   ├── bin/
│   │   ├── Integrator               ← REQUIRED for 2D→1D integration
│   │   └── CalibrantOMP             ← Used by AutoCalibrateZarr.py
│   └── v7/
├── NF_HEDM/
└── build/
```

## Solutions

### Option 1: Update Existing MIDAS Installation (Recommended)

If `/home/beams/S1IDUSER/.MIDAS` is an older MIDAS version, update it:

```bash
cd /home/beams/S1IDUSER/.MIDAS
git pull origin main  # or master
git status  # Check if AutoCalibrateZarr.py is now present
```

Verify the script exists:
```bash
ls -la /home/beams/S1IDUSER/.MIDAS/utils/AutoCalibrateZarr.py
```

### Option 2: Install Fresh MIDAS to Different Location

If the existing `.MIDAS` is not a git repository or is heavily modified:

```bash
# Install to ~/opt/MIDAS
cd /home/beams/S1IDUSER
mkdir -p opt
cd opt
git clone https://github.com/marinerhemant/MIDAS.git
cd MIDAS
# Follow build instructions in README.md
```

Then set environment variable in your APEXA startup script:
```bash
export MIDAS_PATH=/home/beams/S1IDUSER/opt/MIDAS
```

### Option 3: Use MIDAS_PATH Environment Variable

If MIDAS is installed elsewhere on the beamline system:

```bash
# Find existing MIDAS installations
find /home/beams -name "AutoCalibrateZarr.py" 2>/dev/null

# Set MIDAS_PATH to the directory containing utils/
export MIDAS_PATH=/path/to/MIDAS
```

Add this to your shell profile or APEXA startup script.

## Verification Steps

After installing/updating MIDAS:

1. **Verify AutoCalibrateZarr.py exists:**
   ```bash
   ls -la /home/beams/S1IDUSER/.MIDAS/utils/AutoCalibrateZarr.py
   ```

2. **Check MIDAS Python dependencies:**
   ```bash
   python -c "import zarr, diplib, numpy, scipy; print('All dependencies OK')"
   ```

   If this fails, install MIDAS conda environment:
   ```bash
   cd /home/beams/S1IDUSER/.MIDAS
   conda env create -f environment.yml
   conda activate midas
   ```

3. **Restart APEXA MCP server** to pick up the new MIDAS installation

4. **Test auto-calibration:**
   ```bash
   # In APEXA
   autocalibrate the CeO2.tif file with energy=61.332keV, pixel_size=172
   ```

5. **Check server logs** for diagnostic output:
   - Should show: `✓ AutoCalibrateZarr.py found at ...`
   - Should NOT show: `⚠ AutoCalibrateZarr.py NOT found`

## What APEXA Will Do Once MIDAS is Properly Installed

When you ask APEXA to auto-calibrate a CeO2 image, it will:

1. Create a MIDAS parameters file with initial guesses
2. Run `/home/beams/S1IDUSER/.MIDAS/utils/AutoCalibrateZarr.py`
3. AutoCalibrateZarr.py calls `CalibrantOMP` executable for refinement
4. Parse `refined_MIDAS_params.txt` to extract:
   - Beam center (BC_X, BC_Y)
   - Sample-to-detector distance (Lsd)
   - Detector tilt angles (tx, ty, tz)
   - Distortion parameters (p0, p1, p2, p3)
5. Return calibrated parameters for use in data reduction

## Troubleshooting

### Error: "AutoCalibrateZarr.py not found"

The error message will show:
- Which MIDAS_ROOT was detected
- Whether utils/ directory exists
- What Python files are in utils/
- Alternative scripts if found

Follow the suggestions in the error message.

### Error: "ModuleNotFoundError: No module named 'zarr'"

AutoCalibrateZarr.py requires MIDAS Python dependencies:
- zarr
- diplib
- numpy
- scipy
- matplotlib
- pandas
- plotly
- h5py
- numba

Install using MIDAS conda environment:
```bash
cd /home/beams/S1IDUSER/.MIDAS
conda env create -f environment.yml
conda activate midas
```

### APEXA Still Using pyFAI Instead of MIDAS

This means `midas_auto_calibrate` returned an error. Check:
1. MCP server logs for diagnostic output
2. Verify AutoCalibrateZarr.py exists
3. Verify MIDAS Python dependencies are installed
4. Restart MCP server after fixing MIDAS installation

## Contact

For issues with:
- **MIDAS installation**: https://github.com/marinerhemant/MIDAS/issues
- **APEXA/beamline-assistant**: Your local beamline computing team
