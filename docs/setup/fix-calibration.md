# Fix for Auto-Calibration on S1IDUSER Beamline

## Problem
APEXA auto-calibration was failing with:
```
ModuleNotFoundError: No module named 'zarr'
```

This happened because APEXA was using the UV venv Python instead of the MIDAS conda environment.

## Solution Deployed

The code now automatically detects the beamline's MIDAS conda environment:
- **Location**: `~/opt/miniconda3/envs/midas_202411/bin/python`
- **No manual activation needed** - APEXA finds it automatically!

## Steps to Update on Beamline

1. **Pull the latest code**:
```bash
cd /home/beams/S1IDUSER/CAI/git/APS-Beamline-Assistant
git pull origin pawan-modular-v2
```

2. **Restart APEXA** (stop and start the beamline assistant)

3. **Test calibration**:
```bash
APEXA> generate calibration parameters using the Ceria tif file in /home/beams/S1IDUSER/CAI/git/demo_data
```

## What Changed

The `find_midas_python()` function now:
1. Checks for `MIDAS_PYTHON` environment variable (manual override)
2. Searches `~/opt/miniconda3` (beamline location)
3. Looks for `midas_202411` environment specifically
4. Falls back to `midas_env`, `midas`, or `MIDAS`
5. Validates all dependencies are present

## Expected Output

When starting APEXA, you should now see:
```
Found conda installation at: /home/beams/S1IDUSER/opt/miniconda3
✓ Found MIDAS conda environment 'midas_202411': /home/beams/S1IDUSER/opt/miniconda3/envs/midas_202411/bin/python
```

When running auto-calibration:
```
✓ Found MIDAS conda environment 'midas_202411': /home/beams/S1IDUSER/opt/miniconda3/envs/midas_202411/bin/python
Running AutoCalibrateZarr.py...
[Calibration proceeds successfully]
```

## Manual Override (If Needed)

If the automatic detection doesn't work for some reason, you can manually specify:

```bash
export MIDAS_PYTHON=/home/beams/S1IDUSER/opt/miniconda3/envs/midas_202411/bin/python
```

Then start APEXA. It will use this Python for all MIDAS operations.

## Verification

To verify the MIDAS environment is properly detected, run:
```bash
python3 find_midas_env.py
```

This will show all Python environments on the system and which ones have MIDAS dependencies.

## Key Point

**You do NOT need to run `conda activate midas_202411` before starting APEXA!**

The code automatically uses the full path to the midas_202411 Python interpreter, which includes all the MIDAS dependencies. The activation is only needed if you want to manually run MIDAS commands outside of APEXA.

## Technical Details

When APEXA runs AutoCalibrateZarr.py, it now executes:
```python
/home/beams/S1IDUSER/opt/miniconda3/envs/midas_202411/bin/python \\
    ~/opt/MIDAS/utils/AutoCalibrateZarr.py \\
    -dataFN <image> \\
    -paramFN <params> \\
    -ConvertFile 3 \\
    -StoppingStrain 0.0001 \\
    ...
```

This Python interpreter automatically has access to:
- zarr
- diplib
- numba
- h5py
- scikit-image
- plotly
- All other MIDAS dependencies

## If You Still See Errors

1. Verify the conda environment exists:
```bash
ls ~/opt/miniconda3/envs/midas_202411/bin/python
```

2. Verify it has dependencies:
```bash
~/opt/miniconda3/envs/midas_202411/bin/python -c "import zarr, diplib, numba; print('OK')"
```

3. Check APEXA logs for the detection output

4. If all else fails, use the manual override with `export MIDAS_PYTHON=...`

## Contact

If issues persist after pulling the update, contact Pawan or share the APEXA startup logs showing the Python detection output.
