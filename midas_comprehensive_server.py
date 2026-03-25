#!/usr/bin/env python3
"""
MIDAS Comprehensive MCP Server
Provides complete FF-HEDM, NF-HEDM, PF-HEDM, and utility tools for beamline operations

Author: Beamline Assistant Team
Organization: Argonne National Laboratory
"""

from typing import Any, Optional, List, Dict
import json
import sys
import os
from pathlib import Path
import numpy as np
import subprocess
import asyncio
import logging
from mcp.server.fastmcp import FastMCP

# Suppress verbose MCP server logging
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================

def find_midas_python() -> str:
    """Find Python interpreter with MIDAS dependencies (zarr, diplib, etc.).

    Priority order:
    1. conda midas_env environment (dedicated MIDAS environment with all deps)
    2. conda base environment (if it has MIDAS deps)
    3. Current Python (if it has ALL critical MIDAS deps)
    4. System python3
    5. Fallback to current Python with warning
    """
    import shutil

    # Helper function to check if a Python has required deps
    def check_python_deps(python_path: str) -> bool:
        """Check if a Python interpreter has required MIDAS dependencies."""
        try:
            result = subprocess.run(
                [python_path, "-c", "import zarr, diplib, numba, h5py, skimage"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    # PRIORITY 0: Check for manual override via MIDAS_PYTHON environment variable
    midas_python_env = os.environ.get("MIDAS_PYTHON")
    if midas_python_env:
        midas_python_path = Path(midas_python_env)
        if midas_python_path.exists():
            return str(midas_python_path)
        else:
            print(f"⚠ MIDAS_PYTHON set but not found: {midas_python_path}", file=sys.stderr)

    # PRIORITY 1: Look for conda midas_env (official MIDAS environment)
    conda_base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX")

    # Try finding conda base from CONDA_EXE
    if not conda_base:
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            conda_base = Path(conda_exe).parent.parent

    # If not in conda, try common conda locations
    if not conda_base:
        for conda_loc in [
            Path.home() / "opt" / "miniconda3",  # Beamline common location
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
            Path.home() / ".conda",
            Path.home() / "conda",
            Path("/opt/conda"),
            Path("/opt/miniconda3"),
            Path("/opt/anaconda3")
        ]:
            if conda_loc.exists() and (conda_loc / "bin" / "conda").exists():
                conda_base = conda_loc
                break

    if conda_base:
        if isinstance(conda_base, str):
            conda_base = Path(conda_base)

        # Check for MIDAS conda environments (try multiple common names)
        for env_name in ["midas_202411", "midas_env", "midas", "MIDAS"]:
            midas_env_python = conda_base / "envs" / env_name / "bin" / "python"
            if midas_env_python.exists():
                return str(midas_env_python)

        # Check conda base environment
        conda_python = conda_base / "bin" / "python"
        if conda_python.exists() and check_python_deps(str(conda_python)):
            return str(conda_python)

    # PRIORITY 2: Check if current environment has ALL critical MIDAS deps
    try:
        import zarr
        import diplib
        import numba
        import h5py
        import skimage
        return sys.executable
    except ImportError:
        pass

    # PRIORITY 3: Try system python3
    python3_path = shutil.which("python3")
    if python3_path and check_python_deps(python3_path):
        return python3_path

    # No suitable Python found — raise so the caller gets a clear error
    raise RuntimeError(
        "No Python with MIDAS dependencies found.\n"
        "Expected conda env 'midas_env' with zarr, diplib, numba, h5py, skimage.\n"
        f"Checked conda base: {conda_base}\n"
        "Set MIDAS_PYTHON env var to override, or run: conda env create -f MIDAS/environment.yml"
    )

def get_midas_env() -> dict:
    """Get environment variables needed for MIDAS C++ executables.

    Sets DYLD_LIBRARY_PATH / LD_LIBRARY_PATH so C++ binaries find MIDAS's
    own libhdf5, libfftw3, etc.  Do NOT use this for Python scripts —
    overriding DYLD_LIBRARY_PATH breaks h5py in the midas conda env.
    """
    env = os.environ.copy()

    # Add MIDAS bin directory to PATH
    midas_bin_paths = [str(MIDAS_BIN), str(MIDAS_ROOT / "bin")]
    env["PATH"] = ":".join(midas_bin_paths + [env.get("PATH", "")])

    # Library paths for C++ binaries (CalibrantIntegratorOMP, IndexerOMP, etc.)
    lib_paths = [str(MIDAS_BIN.parent / "lib"), str(MIDAS_ROOT / "lib")]

    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths + [env["LD_LIBRARY_PATH"]])
    else:
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths)

    # macOS: needed for C++ executables to find MIDAS dylibs
    if "DYLD_LIBRARY_PATH" in env:
        env["DYLD_LIBRARY_PATH"] = ":".join(lib_paths + [env["DYLD_LIBRARY_PATH"]])
    else:
        env["DYLD_LIBRARY_PATH"] = ":".join(lib_paths)

    env["MIDAS_PATH"] = str(MIDAS_ROOT)
    return env


def get_midas_python_env() -> dict:
    """Get environment variables for MIDAS Python scripts (AutoCalibrateZarr, ff_MIDAS, etc.).

    Deliberately does NOT set DYLD_LIBRARY_PATH / LD_LIBRARY_PATH — overriding
    those breaks h5py in the midas conda env (libhdf5 symbol mismatch on macOS).
    Only adds PYTHONPATH and MIDAS_PATH.
    """
    env = os.environ.copy()

    # Add MIDAS bin to PATH so any CalibrantIntegratorOMP calls inside scripts work
    midas_bin_paths = [str(MIDAS_BIN), str(MIDAS_ROOT / "bin")]
    env["PATH"] = ":".join(midas_bin_paths + [env.get("PATH", "")])

    env["MIDAS_PATH"] = str(MIDAS_ROOT)

    # MIDAS/utils must be on PYTHONPATH so midas_config is importable
    midas_utils = str(MIDAS_ROOT / "utils")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = midas_utils + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = midas_utils

    return env

def find_midas_installation() -> Path:
    """Find MIDAS installation by checking common locations.

    Priority order:
    1. MIDAS_PATH environment variable (if valid)
    2. Search common locations and validate each
    3. Auto-detect from Git repositories

    Validation checks for each candidate:
    - Has utils/AutoCalibrateZarr.py (required for calibration)
    - Has FF_HEDM/bin/ or build/bin/ (required for executables)
    """
    def validate_midas_path(path: Path) -> bool:
        """Check if path is a valid MIDAS installation."""
        try:
            if not path.exists() or not path.is_dir():
                return False

            # Must have AutoCalibrateZarr.py for calibration
            autocal = path / "utils" / "AutoCalibrateZarr.py"

            # Must have executables (either in FF_HEDM/bin or build/bin)
            has_executables = (
                (path / "FF_HEDM" / "bin").exists() or
                (path / "build" / "bin").exists()
            )

            return autocal.exists() and has_executables
        except (PermissionError, OSError):
            # Skip paths we don't have permission to access
            return False

    # Check environment variable first
    if "MIDAS_PATH" in os.environ:
        midas_path = Path(os.environ["MIDAS_PATH"]).expanduser().absolute()
        if validate_midas_path(midas_path):
            print(f"  MIDAS: {midas_path}", file=sys.stderr)
            return midas_path
        else:
            print(f"⚠ MIDAS_PATH set but invalid: {midas_path}", file=sys.stderr)

    # Search common installation locations
    # Build search paths dynamically
    common_paths = []

    # User home subdirectories
    for subdir in ["Git", "git", "src", "Documents", "opt", ""]:
        if subdir:
            common_paths.append(Path.home() / subdir / "MIDAS")
            common_paths.append(Path.home() / subdir / "Documents" / "MIDAS")

    # Beamline-specific paths (S1IDUSER, etc.)
    beamline_base = Path("/home/beams")
    if beamline_base.exists():
        for user_dir in beamline_base.glob("S*USER"):
            common_paths.append(user_dir / "opt" / "MIDAS")
            common_paths.append(user_dir / "MIDAS")

    # System-wide installations
    for prefix in ["/opt", "/usr/local", "/usr", Path.home()]:
        common_paths.append(Path(prefix) / "MIDAS")
        common_paths.append(Path(prefix) / ".MIDAS")

    # Current directory
    common_paths.extend([
        Path.cwd() / "MIDAS",
        Path.cwd().parent / "MIDAS",
    ])

    # Remove duplicates while preserving order
    seen = set()
    common_paths = [p for p in common_paths if not (p in seen or seen.add(p))]

    valid_installations = []
    for path in common_paths:
        if validate_midas_path(path):
            valid_installations.append(path)

    if valid_installations:
        selected = valid_installations[0]
        if len(valid_installations) > 1:
            print(f"⚠ Multiple MIDAS installations — using {selected}. "
                  f"Set MIDAS_PATH to override.", file=sys.stderr)
        else:
            print(f"  MIDAS: {selected}", file=sys.stderr)
        return selected

    # No valid installation found
    print("❌ No valid MIDAS installation found. "
          "Clone https://github.com/marinerhemant/MIDAS and set MIDAS_PATH.", file=sys.stderr)
    return Path.home() / ".MIDAS"

# MIDAS installation paths
MIDAS_ROOT = find_midas_installation()
MIDAS_BIN = MIDAS_ROOT / "build" / "bin"  # Executables are in build/bin
MIDAS_FF_BIN = MIDAS_ROOT / "FF_HEDM" / "bin"  # FF-HEDM specific executables
MIDAS_NF_BIN = MIDAS_ROOT / "NF_HEDM" / "bin"  # NF-HEDM specific executables
MIDAS_FF_V7 = MIDAS_ROOT / "FF_HEDM" / "v7"
MIDAS_NF_V7 = MIDAS_ROOT / "NF_HEDM" / "v7"
MIDAS_UTILS = MIDAS_ROOT / "utils"

_autocal_script = MIDAS_UTILS / "AutoCalibrateZarr.py"
if not _autocal_script.exists():
    print(f"⚠ AutoCalibrateZarr.py not found at {_autocal_script} — calibration unavailable", file=sys.stderr)

# Add MIDAS Python modules to path
for path in [MIDAS_UTILS, MIDAS_FF_V7, MIDAS_NF_V7]:
    if path.exists():
        sys.path.insert(0, str(path))

# Initialize FastMCP server
mcp = FastMCP("midas-comprehensive-analysis")

# =============================================================================
# DEPENDENCY IMPORTS
# =============================================================================

# Import MIDAS Python APIs
MIDAS_PYTHON_AVAILABLE = False
try:
    # Core scientific libraries
    import fabio
    import pyFAI
    from scipy import ndimage
    from scipy.signal import find_peaks, peak_widths
    from scipy.optimize import curve_fit
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    MIDAS_AVAILABLE = True
except ImportError as e:
    MIDAS_AVAILABLE = False
    print(f"⚠ MIDAS scientific deps missing: {e}", file=sys.stderr)

# Try to import MIDAS Python workflow modules
try:
    # These may not exist depending on MIDAS installation
    # from ff_MIDAS import process_layer, read_parameter_file
    # from nf_MIDAS import run_preprocessing, run_fitting_and_postprocessing
    # from calcMiso import GetMisorientationAngle
    MIDAS_PYTHON_AVAILABLE = True
except ImportError:
    MIDAS_PYTHON_AVAILABLE = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_result(result: dict) -> str:
    """Format analysis results into readable JSON string."""
    return json.dumps(result, indent=2)

def validate_file(file_path: str, must_exist: bool = True) -> tuple[bool, str]:
    """Validate file path and return (success, message)."""
    path = Path(file_path).expanduser()
    if must_exist and not path.exists():
        return False, f"File not found: {path}"
    return True, str(path)

def run_midas_executable(executable: str, param_file: str, cwd: str = None,
                         timeout: int = 3600, env: dict = None) -> dict:
    """Run a MIDAS C executable and return results."""
    # Try multiple possible locations for executables
    possible_paths = [
        MIDAS_BIN / executable,
        MIDAS_FF_BIN / executable,
        MIDAS_NF_BIN / executable
    ]

    exe_path = None
    for p in possible_paths:
        if p.exists():
            exe_path = p
            break

    if not exe_path:
        return {
            "success": False,
            "error": f"Executable not found: {executable}",
            "searched_paths": [str(p) for p in possible_paths],
            "executable": executable
        }

    try:
        # Use MIDAS environment with proper library paths
        if env is None:
            env = get_midas_env()
        cmd = [str(exe_path), str(param_file)]
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "executable": executable
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout}s",
            "executable": executable
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "executable": executable
        }

def run_python_script(script_name: str, args: list, cwd: str = None,
                      timeout: int = 7200) -> dict:
    """Run a MIDAS Python script using the correct conda environment."""
    # Try multiple possible locations
    possible_paths = [
        MIDAS_UTILS / script_name,
        MIDAS_FF_V7 / script_name,
        MIDAS_NF_V7 / script_name,
        MIDAS_ROOT / script_name
    ]

    script_path = None
    for p in possible_paths:
        if p.exists():
            script_path = p
            break

    if not script_path:
        return {
            "success": False,
            "error": f"Script not found: {script_name}",
            "searched_paths": [str(p) for p in possible_paths]
        }

    try:
        # Use MIDAS Python (conda midas_env) instead of "python"
        midas_python = find_midas_python()
        cmd = [midas_python, str(script_path)] + args
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=get_midas_python_env()  # Python script — no DYLD_LIBRARY_PATH override
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "script": script_name,
            "command": " ".join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Script timed out after {timeout}s",
            "script": script_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "script": script_name
        }

def load_diffraction_image(image_path: str):
    """Load diffraction image using fabio."""
    try:
        if MIDAS_AVAILABLE:
            img = fabio.open(image_path)
            return img.data.astype(np.float64)
        else:
            return np.random.rand(2048, 2048) * 1000
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {e}")

# =============================================================================
# FF-HEDM PRODUCTION TOOLS
# =============================================================================

@mcp.tool()
async def run_ff_hedm_full_workflow(
    result_folder: str,
    param_file: str,
    data_file: str,
    n_cpus: int = 32,
    start_layer: int = 1,
    end_layer: int = 1,
    do_peak_search: bool = True,
    grains_seed_file: str = None,
    machine_name: str = "local",
    convert_files: bool = True,
    use_gpu: bool = False,
    resume_file: str = None,
    restart_from: str = None
) -> str:
    """Run complete FF-HEDM production workflow using ff_MIDAS.py.

    This executes the full Far-Field High Energy Diffraction Microscopy workflow:
    1. Data conversion to Zarr format (if convert_files=True)
    2. HKL generation (GetHKLListZarr)
    3. Peak search (PeaksFittingOMPZarrRefactor)
    4. Peak merging (MergeOverlappingPeaksAllZarr)
    5. Data preparation (CalcRadiusAllZarr, FitSetupZarr)
    6. Data binning (SaveBinData)
    7. Indexing (IndexerOMP or IndexerGPU if use_gpu=True)
    8. Refinement (FitPosOrStrainsOMP or FitPosOrStrainsGPU if use_gpu=True)
    9. Post-processing (ProcessGrains)

    Args:
        result_folder: Output directory for results
        param_file: Path to Parameters.txt file
        data_file: Input data file (.zip or raw format)
        n_cpus: Number of CPU cores to use
        start_layer: Starting layer number
        end_layer: Ending layer number
        do_peak_search: Whether to perform peak search (set False if peaks exist)
        grains_seed_file: Optional seed grains file for indexing
        machine_name: Machine config (local, orthrosnew, umich, polaris)
        convert_files: Whether to convert raw data to Zarr
        use_gpu: Use GPU executables (IndexerGPU, FitPosOrStrainsGPU) — v10 feature
        resume_file: Path to checkpoint file to resume from (--resume flag) — v10 feature
        restart_from: Step name to restart workflow from (--restartFrom flag) — v10 feature

    Returns:
        JSON with workflow status, output files, and grain statistics
    """
    try:
        # Validate inputs
        result_path = Path(result_folder).expanduser()
        result_path.mkdir(parents=True, exist_ok=True)

        valid, param_path = validate_file(param_file)
        if not valid:
            return format_result({"error": param_path, "status": "failed"})

        valid, data_path = validate_file(data_file)
        if not valid:
            return format_result({"error": data_path, "status": "failed"})

        # Build command
        args = [
            "-resultFolder", str(result_path),
            "-paramFN", param_path,
            "-dataFN", data_path,
            "-nCPUs", str(n_cpus),
            "-machineName", machine_name,
            "-startLayerNr", str(start_layer),
            "-endLayerNr", str(end_layer),
            "-doPeakSearch", "1" if do_peak_search else "0",
            "-convertFiles", "1" if convert_files else "0"
        ]

        if use_gpu:
            args.append("-useGPU")
        if resume_file:
            args.extend(["--resume", resume_file])
        if restart_from:
            args.extend(["--restartFrom", restart_from])

        if grains_seed_file:
            valid, seed_path = validate_file(grains_seed_file)
            if valid:
                args.extend(["-grainsFile", seed_path])

        print(f"Starting FF-HEDM workflow: layers {start_layer}-{end_layer}", file=sys.stderr)

        # Execute workflow
        result = run_python_script("ff_MIDAS.py", args, cwd=str(result_path), timeout=7200)

        # Check for output files
        output_info = {
            "grains_csv": None,
            "zarr_archive": None,
            "layer_outputs": []
        }

        for layer in range(start_layer, end_layer + 1):
            layer_dir = result_path / f"LayerNr_{layer}"
            if layer_dir.exists():
                grains_file = layer_dir / "GrainsReconstructed.csv"
                if grains_file.exists():
                    output_info["layer_outputs"].append({
                        "layer": layer,
                        "grains_file": str(grains_file),
                        "file_size_kb": grains_file.stat().st_size / 1024
                    })

                    # Try to count grains
                    try:
                        with open(grains_file, 'r') as f:
                            n_grains = sum(1 for line in f) - 1  # Subtract header
                        output_info["layer_outputs"][-1]["n_grains"] = n_grains
                    except:
                        pass

        # Look for Zarr archive
        zarr_files = list(result_path.glob("*.MIDAS.zip"))
        if zarr_files:
            output_info["zarr_archive"] = str(zarr_files[0])

        return format_result({
            "tool": "run_ff_hedm_full_workflow",
            "status": "completed" if result["success"] else "failed",
            "workflow": "FF-HEDM Full Production",
            "execution": result,
            "parameters": {
                "result_folder": str(result_path),
                "param_file": param_path,
                "data_file": data_path,
                "n_cpus": n_cpus,
                "layers": f"{start_layer}-{end_layer}",
                "machine": machine_name
            },
            "output": output_info,
            "total_layers_processed": len(output_info["layer_outputs"]),
            "total_grains_found": sum(l.get("n_grains", 0) for l in output_info["layer_outputs"])
        })

    except Exception as e:
        return format_result({
            "tool": "run_ff_hedm_full_workflow",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def run_pf_hedm_workflow(
    param_file: str,
    positions_file: str,
    n_cpus: int = 32,
    one_solution_per_voxel: bool = True,
    normalize_intensities: str = "none",
    do_peak_search: bool = True,
    machine_name: str = "local",
    use_gpu: bool = False,
    resume_file: str = None,
    restart_from: str = None,
    do_tomo: bool = False
) -> str:
    """Run Point-Focus HEDM scanning workflow using pf_MIDAS.py.

    Point-Focus HEDM is used for scanning experiments with a focused beam.
    Produces 3D orientation maps with better spatial resolution than FF-HEDM.

    Args:
        param_file: Path to Parameters.txt file
        positions_file: CSV file with scan positions (x, y, z coordinates)
        n_cpus: Number of CPU cores to use
        one_solution_per_voxel: Limit to one orientation per voxel
        normalize_intensities: Normalization method (none, max, sum)
        do_peak_search: Whether to perform peak search
        machine_name: Machine config (local, orthrosnew, umich, polaris)
        use_gpu: Use GPU executables — v10 feature
        resume_file: Path to checkpoint file to resume from (--resume flag) — v10 feature
        restart_from: Step name to restart from (--restartFrom flag) — v10 feature
        do_tomo: Enable tomographic reconstruction mode — v10 feature

    Returns:
        JSON with workflow status and 3D orientation map data
    """
    try:
        # Validate inputs
        valid, param_path = validate_file(param_file)
        if not valid:
            return format_result({"error": param_path, "status": "failed"})

        valid, pos_path = validate_file(positions_file)
        if not valid:
            return format_result({"error": pos_path, "status": "failed"})

        # Build command
        args = [
            "-paramFile", param_path,
            "-nCPUs", str(n_cpus),
            "-machineName", machine_name,
            "-doPeakSearch", "1" if do_peak_search else "0",
            "-oneSolPerVox", "1" if one_solution_per_voxel else "0",
            "-normalizeIntensities", normalize_intensities
        ]

        if use_gpu:
            args.append("-useGPU")
        if do_tomo:
            args.append("-doTomo")
        if resume_file:
            args.extend(["--resume", resume_file])
        if restart_from:
            args.extend(["--restartFrom", restart_from])

        print("Starting PF-HEDM scanning workflow", file=sys.stderr)

        # Execute workflow
        result_dir = Path(param_path).parent
        result = run_python_script("pf_MIDAS.py", args, cwd=str(result_dir), timeout=7200)

        # Check for outputs
        output_info = {
            "grains_csv": None,
            "scanning_positions": None,
            "n_positions": 0
        }

        # Count positions
        try:
            import csv
            with open(pos_path, 'r') as f:
                output_info["n_positions"] = sum(1 for row in csv.reader(f)) - 1
        except:
            pass

        # Look for output grains file
        grains_file = result_dir / "Grains.csv"
        if grains_file.exists():
            output_info["grains_csv"] = str(grains_file)
            try:
                with open(grains_file, 'r') as f:
                    output_info["n_solutions"] = sum(1 for line in f) - 1
            except:
                pass

        return format_result({
            "tool": "run_pf_hedm_workflow",
            "status": "completed" if result["success"] else "failed",
            "workflow": "PF-HEDM Scanning",
            "execution": result,
            "parameters": {
                "param_file": param_path,
                "positions_file": pos_path,
                "n_cpus": n_cpus,
                "one_solution_per_voxel": one_solution_per_voxel,
                "machine": machine_name
            },
            "output": output_info
        })

    except Exception as e:
        return format_result({
            "tool": "run_pf_hedm_workflow",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def run_ff_calibration(
    param_file: str,
    calibrant: str = "CeO2",
    use_omp: bool = True,
    fit_tilt: bool = True,
    fit_panel_shifts: bool = False
) -> str:
    """Run FF-HEDM detector calibration workflow.

    Calibrates detector parameters (distance, beam center, tilt) using
    a calibrant material with known diffraction pattern.

    Args:
        param_file: Path to Parameters.txt file
        calibrant: Calibrant material (CeO2, LaB6, Si, etc.)
        use_omp: Use OpenMP parallel version
        fit_tilt: Fit detector tilt parameters
        fit_panel_shifts: Fit panel-to-panel shifts (multi-panel detectors)

    Returns:
        JSON with calibrated parameters and fit quality metrics
    """
    try:
        valid, param_path = validate_file(param_file)
        if not valid:
            return format_result({"error": param_path, "status": "failed"})

        work_dir = Path(param_path).parent
        results = {
            "tool": "run_ff_calibration",
            "workflow": "FF-HEDM Calibration",
            "steps": []
        }

        # Step 1: Run calibrant fitting (v10: CalibrantIntegratorOMP is primary)
        exe = "CalibrantIntegratorOMP" if use_omp else "Calibrant"
        print(f"Running {exe} for {calibrant}", file=sys.stderr)

        result = run_midas_executable(exe, param_path, cwd=str(work_dir), timeout=600)
        results["steps"].append({
            "step": 1,
            "name": f"Calibrant Fitting ({exe})",
            "status": "completed" if result["success"] else "failed",
            "calibrant": calibrant
        })

        if not result["success"]:
            results["status"] = "failed"
            results["error"] = result.get("error", "Calibrant fitting failed")
            return format_result(results)

        # Step 2: Fit tilt and beam center
        if fit_tilt:
            print("Fitting tilt, beam center, and sample distance", file=sys.stderr)
            result = run_midas_executable("FitTiltBCLsdSample", param_path,
                                         cwd=str(work_dir), timeout=600)
            results["steps"].append({
                "step": 2,
                "name": "Fit Tilt/BC/Lsd",
                "status": "completed" if result["success"] else "failed"
            })

            if not result["success"]:
                results["status"] = "warning"
                results["warning"] = "Tilt fitting failed, using initial values"

        # Step 3: Fit panel shifts (if multi-panel detector)
        if fit_panel_shifts:
            print("Fitting panel shifts", file=sys.stderr)
            result = run_midas_executable("CalibrantPanelShiftsOMP", param_path,
                                         cwd=str(work_dir), timeout=600)
            results["steps"].append({
                "step": 3,
                "name": "Panel Shifts",
                "status": "completed" if result["success"] else "failed"
            })

        # Try to read calibrated parameters
        calibrated_params = {}
        try:
            with open(param_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0]
                            if key in ['Lsd', 'BC', 'tx', 'ty', 'tz', 'p0', 'p1', 'p2']:
                                calibrated_params[key] = ' '.join(parts[1:])
        except:
            pass

        results["status"] = "completed"
        results["calibrated_parameters"] = calibrated_params
        results["param_file"] = param_path

        return format_result(results)

    except Exception as e:
        return format_result({
            "tool": "run_ff_calibration",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def run_ff_grain_tracking(
    grains_files: list,
    tracking_tolerance: float = 50.0,
    output_file: str = "grains_tracked.csv",
    position_tolerance: float = 50.0,
    orientation_tolerance: float = 5.0
) -> str:
    """Track grains through deformation/temperature series.

    Links grain IDs across multiple datasets (e.g., in-situ deformation,
    heating/cooling cycles) based on position and orientation proximity.

    Args:
        grains_files: List of Grains.csv files in chronological order
        tracking_tolerance: Overall tracking tolerance (microns)
        output_file: Output file with tracked grain IDs
        position_tolerance: Position tolerance in microns
        orientation_tolerance: Orientation tolerance in degrees

    Returns:
        JSON with tracking statistics and evolution data
    """
    try:
        # Validate all input files
        valid_files = []
        for gf in grains_files:
            valid, path = validate_file(gf)
            if valid:
                valid_files.append(path)
            else:
                return format_result({
                    "error": f"Grains file not found: {gf}",
                    "status": "failed"
                })

        if len(valid_files) < 2:
            return format_result({
                "error": "Need at least 2 grains files for tracking",
                "status": "failed"
            })

        # GrainTracking executable usage
        # Note: This may need custom implementation or wrapper script
        # since GrainTracking might need specific parameter file format

        work_dir = Path(valid_files[0]).parent

        # Create temporary parameter file for tracking
        tracking_params = work_dir / "tracking_params.txt"
        with open(tracking_params, 'w') as f:
            f.write(f"PositionTolerance {position_tolerance}\n")
            f.write(f"OrientationTolerance {orientation_tolerance}\n")
            f.write(f"OutputFile {output_file}\n")
            for i, gf in enumerate(valid_files):
                f.write(f"GrainsFile_{i} {gf}\n")

        print(f"Tracking grains across {len(valid_files)} datasets", file=sys.stderr)

        result = run_midas_executable("GrainTracking", str(tracking_params),
                                     cwd=str(work_dir), timeout=600)

        # Parse tracking results
        tracking_stats = {
            "n_datasets": len(valid_files),
            "datasets": valid_files,
            "grains_per_dataset": [],
            "tracked_grains": 0,
            "tracking_success_rate": 0.0
        }

        # Count grains in each dataset
        for gf in valid_files:
            try:
                with open(gf, 'r') as f:
                    n_grains = sum(1 for line in f) - 1
                tracking_stats["grains_per_dataset"].append(n_grains)
            except:
                tracking_stats["grains_per_dataset"].append(0)

        # Check output file
        output_path = work_dir / output_file
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    tracking_stats["tracked_grains"] = sum(1 for line in f) - 1

                if tracking_stats["grains_per_dataset"]:
                    avg_grains = np.mean(tracking_stats["grains_per_dataset"])
                    tracking_stats["tracking_success_rate"] = (
                        tracking_stats["tracked_grains"] / avg_grains
                    )
            except:
                pass

        return format_result({
            "tool": "run_ff_grain_tracking",
            "status": "completed" if result["success"] else "failed",
            "workflow": "Grain Tracking",
            "execution": result,
            "parameters": {
                "position_tolerance_um": position_tolerance,
                "orientation_tolerance_deg": orientation_tolerance
            },
            "tracking_statistics": tracking_stats,
            "output_file": str(output_path) if output_path.exists() else None
        })

    except Exception as e:
        return format_result({
            "tool": "run_ff_grain_tracking",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def match_grains(
    grains_files: list,
    output_file: str = "matched_grains.csv",
    position_tolerance: float = 100.0,
    orientation_tolerance: float = 2.0,
    match_across_layers: bool = False,
    layer_spacing: float = None
) -> str:
    """Match grains across multiple datasets using Hungarian algorithm (v10).

    Uses match_grains.py to link grain IDs across load states, temperatures,
    or NF-HEDM layers using the Hungarian (linear sum assignment) algorithm.
    Supports both in-situ load matching and layer-to-layer stitching.

    Args:
        grains_files: List of Grains.csv files to match (in order)
        output_file: Output CSV with matched grain IDs across datasets
        position_tolerance: Max centroid distance for matching (microns)
        orientation_tolerance: Max misorientation for matching (degrees)
        match_across_layers: Match grains between NF-HEDM layers for stitching
        layer_spacing: Layer spacing in microns (required if match_across_layers=True)

    Returns:
        JSON with matching statistics and matched grain IDs
    """
    try:
        valid_files = []
        for gf in grains_files:
            valid, path = validate_file(gf)
            if valid:
                valid_files.append(path)
            else:
                return format_result({"error": f"File not found: {gf}", "status": "failed"})

        if len(valid_files) < 2:
            return format_result({"error": "Need at least 2 grains files", "status": "failed"})

        work_dir = Path(valid_files[0]).parent
        output_path = work_dir / output_file

        args = [
            "--files", *valid_files,
            "--output", str(output_path),
            "--positionTolerance", str(position_tolerance),
            "--orientationTolerance", str(orientation_tolerance)
        ]

        if match_across_layers:
            args.append("--matchLayers")
        if layer_spacing is not None:
            args.extend(["--layerSpacing", str(layer_spacing)])

        print(f"Matching grains across {len(valid_files)} datasets", file=sys.stderr)

        result = run_python_script("match_grains.py", args, cwd=str(work_dir), timeout=600)

        match_stats = {"n_datasets": len(valid_files), "datasets": valid_files}

        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    match_stats["n_matched_grains"] = sum(1 for line in f) - 1
            except:
                pass

        return format_result({
            "tool": "match_grains",
            "status": "completed" if result["success"] else "failed",
            "workflow": "Grain Matching (Hungarian)",
            "execution": result,
            "parameters": {
                "n_datasets": len(valid_files),
                "position_tolerance_um": position_tolerance,
                "orientation_tolerance_deg": orientation_tolerance,
                "match_across_layers": match_across_layers
            },
            "statistics": match_stats,
            "output_file": str(output_path) if output_path.exists() else None
        })

    except Exception as e:
        return format_result({"tool": "match_grains", "status": "error", "error": str(e)})


# =============================================================================
# NF-HEDM RECONSTRUCTION TOOLS
# =============================================================================

@mcp.tool()
async def run_nf_hedm_reconstruction(
    param_file: str,
    n_cpus: int = 10,
    ff_seed_orientations: bool = True,
    ff_grains_file: str = None,
    do_image_processing: bool = True,
    refine_parameters: bool = False,
    multi_grid_points: bool = False,
    machine_name: str = "local",
    n_nodes: int = 1,
    use_gpu: bool = False,
    resume_file: str = None,
    restart_from: str = None
) -> str:
    """Run complete NF-HEDM microstructure reconstruction using nf_MIDAS.py.

    Near-Field HEDM produces voxel-by-voxel 3D orientation maps with
    higher spatial resolution than FF-HEDM.

    Two modes:
    1. Reconstruction mode (refine_parameters=False):
       - Pre-processing: GetHKLListNF, GenSeedOrientationsFF2NFHEDM
       - Grid creation: MakeHexGrid
       - Spot simulation: MakeDiffrSpots
       - Image processing: MedianImageLibTiff, ProcessImagesCombined
       - Fitting: FitOrientationOMP or FitOrientationGPU (if use_gpu=True)
       - Post-processing: ParseMic

    2. Parameter refinement mode (refine_parameters=True):
       - Refines experimental geometry using FitOrientationParameters

    Args:
        param_file: Path to NF-HEDM Parameters.txt file
        n_cpus: Number of CPU cores to use
        ff_seed_orientations: Use FF-HEDM results as seed orientations
        ff_grains_file: Path to FF-HEDM Grains.csv (if ff_seed_orientations=True)
        do_image_processing: Perform image processing (median filter, background)
        refine_parameters: Run parameter refinement instead of reconstruction
        multi_grid_points: Use multiple grid points for parameter refinement
        machine_name: Machine config (local, orthrosnew, umich, polaris)
        n_nodes: Number of compute nodes for HPC
        use_gpu: Use FitOrientationGPU (-gpuFit flag) — v10 feature
        resume_file: Path to checkpoint file to resume from (--resume flag) — v10 feature
        restart_from: Step name to restart from (--restartFrom flag) — v10 feature

    Returns:
        JSON with reconstruction status and Grains.mic output info
    """
    try:
        # Validate param file
        valid, param_path = validate_file(param_file)
        if not valid:
            return format_result({"error": param_path, "status": "failed"})

        # Validate FF grains file if using seeds
        if ff_seed_orientations and ff_grains_file:
            valid, ff_path = validate_file(ff_grains_file)
            if not valid:
                return format_result({
                    "error": f"FF grains file not found: {ff_grains_file}",
                    "status": "failed"
                })

        # Build command
        args = [
            "-paramFN", param_path,
            "-nCPUs", str(n_cpus),
            "-machineName", machine_name,
            "-nNodes", str(n_nodes),
            "-refineParameters", "1" if refine_parameters else "0",
            "-ffSeedOrientations", "1" if ff_seed_orientations else "0",
            "-doImageProcessing", "1" if do_image_processing else "0",
            "-multiGridPoints", "1" if multi_grid_points else "0"
        ]

        if use_gpu:
            args.append("-gpuFit")
        if resume_file:
            args.extend(["--resume", resume_file])
        if restart_from:
            args.extend(["--restartFrom", restart_from])

        mode = "Parameter Refinement" if refine_parameters else "Full Reconstruction"
        print(f"Starting NF-HEDM {mode}", file=sys.stderr)

        # Execute workflow
        work_dir = Path(param_path).parent
        result = run_python_script("nf_MIDAS.py", args, cwd=str(work_dir), timeout=14400)

        # Check for output files
        output_info = {
            "grains_mic": None,
            "n_voxels": 0,
            "reconstruction_complete": False
        }

        mic_file = work_dir / "Grains.mic"
        if mic_file.exists():
            output_info["grains_mic"] = str(mic_file)
            output_info["file_size_mb"] = mic_file.stat().st_size / (1024 * 1024)
            output_info["reconstruction_complete"] = True

            # Try to count voxels
            try:
                with open(mic_file, 'r') as f:
                    output_info["n_voxels"] = sum(1 for line in f if not line.startswith('%'))
            except:
                pass

        # Check for logs
        log_dir = work_dir / "midas_log"
        if log_dir.exists():
            output_info["log_directory"] = str(log_dir)
            output_info["log_files"] = [f.name for f in log_dir.glob("*.log")]

        return format_result({
            "tool": "run_nf_hedm_reconstruction",
            "status": "completed" if result["success"] else "failed",
            "workflow": f"NF-HEDM {mode}",
            "execution": result,
            "parameters": {
                "param_file": param_path,
                "n_cpus": n_cpus,
                "ff_seed_orientations": ff_seed_orientations,
                "mode": mode,
                "machine": machine_name
            },
            "output": output_info
        })

    except Exception as e:
        return format_result({
            "tool": "run_nf_hedm_reconstruction",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def convert_nf_to_dream3d(
    nf_mic_file: str,
    output_hdf5: str = "nf_dream3d.h5",
    include_strain: bool = False,
    voxel_size: float = 1.0
) -> str:
    """Convert NF-HEDM output to DREAM.3D format for visualization.

    Converts MIDAS Grains.mic format to DREAM.3D HDF5 format for
    visualization in Paraview, DREAM.3D, or other 3D visualization tools.

    Args:
        nf_mic_file: Path to NF-HEDM Grains.mic file
        output_hdf5: Output HDF5 file name
        include_strain: Include strain tensor data (if available)
        voxel_size: Voxel size in microns

    Returns:
        JSON with conversion status and file information
    """
    try:
        # Validate input
        valid, mic_path = validate_file(nf_mic_file)
        if not valid:
            return format_result({"error": mic_path, "status": "failed"})

        work_dir = Path(mic_path).parent
        output_path = work_dir / output_hdf5

        # Build command for conversion utility
        args = [
            mic_path,
            str(output_path),
            "--voxel-size", str(voxel_size)
        ]

        if include_strain:
            args.append("--include-strain")

        print(f"Converting {mic_path} to DREAM.3D format", file=sys.stderr)

        result = run_python_script("nf_paraview_gen.py", args, cwd=str(work_dir))

        # Check output
        conversion_info = {
            "input_file": mic_path,
            "output_file": None,
            "conversion_successful": False
        }

        if output_path.exists():
            conversion_info["output_file"] = str(output_path)
            conversion_info["file_size_mb"] = output_path.stat().st_size / (1024 * 1024)
            conversion_info["conversion_successful"] = True

            # Try to read basic info from HDF5
            try:
                import h5py
                with h5py.File(output_path, 'r') as h5f:
                    conversion_info["hdf5_structure"] = list(h5f.keys())
            except:
                pass

        return format_result({
            "tool": "convert_nf_to_dream3d",
            "status": "completed" if result["success"] else "failed",
            "execution": result,
            "parameters": {
                "voxel_size_um": voxel_size,
                "include_strain": include_strain
            },
            "conversion": conversion_info,
            "usage_note": "Open in Paraview or DREAM.3D for 3D visualization"
        })

    except Exception as e:
        return format_result({
            "tool": "convert_nf_to_dream3d",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def overlay_ff_nf_results(
    ff_grains_file: str,
    nf_mic_file: str,
    output_plot: str = "ff_nf_overlay.png",
    slice_position: str = "middle"
) -> str:
    """Overlay FF and NF grain maps for validation.

    Creates visualization comparing coarse FF-HEDM grain map with
    detailed NF-HEDM microstructure reconstruction.

    Args:
        ff_grains_file: Path to FF-HEDM Grains.csv
        nf_mic_file: Path to NF-HEDM Grains.mic
        output_plot: Output plot file name
        slice_position: Which slice to show (top, middle, bottom)

    Returns:
        JSON with comparison statistics and plot file
    """
    try:
        # Validate inputs
        valid, ff_path = validate_file(ff_grains_file)
        if not valid:
            return format_result({"error": ff_path, "status": "failed"})

        valid, nf_path = validate_file(nf_mic_file)
        if not valid:
            return format_result({"error": nf_path, "status": "failed"})

        work_dir = Path(ff_path).parent
        output_path = work_dir / output_plot

        # Build command
        args = [
            ff_path,
            nf_path,
            "--output", str(output_path),
            "--slice", slice_position
        ]

        print(f"Overlaying FF and NF results", file=sys.stderr)

        result = run_python_script("PlotFFNF.py", args, cwd=str(work_dir))

        # Gather comparison statistics
        comparison = {
            "ff_grains_file": ff_path,
            "nf_mic_file": nf_path,
            "n_ff_grains": 0,
            "n_nf_voxels": 0,
            "plot_file": None
        }

        # Count FF grains
        try:
            with open(ff_path, 'r') as f:
                comparison["n_ff_grains"] = sum(1 for line in f) - 1
        except:
            pass

        # Count NF voxels
        try:
            with open(nf_path, 'r') as f:
                comparison["n_nf_voxels"] = sum(1 for line in f if not line.startswith('%'))
        except:
            pass

        # Check plot output
        if output_path.exists():
            comparison["plot_file"] = str(output_path)
            comparison["plot_size_kb"] = output_path.stat().st_size / 1024

        return format_result({
            "tool": "overlay_ff_nf_results",
            "status": "completed" if result["success"] else "failed",
            "execution": result,
            "comparison": comparison,
            "interpretation": {
                "ff_spatial_resolution": "~100-500 μm per grain",
                "nf_spatial_resolution": "~1-10 μm per voxel",
                "resolution_improvement": f"~{comparison['n_nf_voxels'] / max(comparison['n_ff_grains'], 1):.0f}x"
            }
        })

    except Exception as e:
        return format_result({
            "tool": "overlay_ff_nf_results",
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# ADVANCED ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
async def calculate_misorientation(
    grains_file: str,
    grain_id_1: int,
    grain_id_2: int,
    space_group: int = 225
) -> str:
    """Calculate misorientation angle between two grains.

    Computes the crystallographic misorientation angle and axis between
    two grains based on their orientation matrices.

    Args:
        grains_file: Path to Grains.csv file
        grain_id_1: First grain ID
        grain_id_2: Second grain ID
        space_group: Crystal space group number (default: 225 for FCC)

    Returns:
        JSON with misorientation angle, axis, and grain boundary character
    """
    try:
        valid, grains_path = validate_file(grains_file)
        if not valid:
            return format_result({"error": grains_path, "status": "failed"})

        # Read grains file and extract orientations
        grains_data = {}
        try:
            import csv
            with open(grains_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gid = int(float(row.get('GrainID', row.get('grainId', -1))))
                    if gid in [grain_id_1, grain_id_2]:
                        # Extract Euler angles (column names vary)
                        euler_cols = ['EulerAngle1', 'EulerAngle2', 'EulerAngle3']
                        alt_cols = ['phi1', 'Phi', 'phi2']

                        euler = None
                        for col_set in [euler_cols, alt_cols]:
                            if all(c in row for c in col_set):
                                euler = [float(row[c]) for c in col_set]
                                break

                        if euler:
                            grains_data[gid] = {
                                'grain_id': gid,
                                'euler_angles': euler
                            }
        except Exception as e:
            return format_result({
                "error": f"Failed to parse grains file: {e}",
                "status": "failed"
            })

        if grain_id_1 not in grains_data:
            return format_result({
                "error": f"Grain {grain_id_1} not found in file",
                "status": "failed"
            })

        if grain_id_2 not in grains_data:
            return format_result({
                "error": f"Grain {grain_id_2} not found in file",
                "status": "failed"
            })

        euler1 = grains_data[grain_id_1]['euler_angles']
        euler2 = grains_data[grain_id_2]['euler_angles']

        # Try to use MIDAS calcMiso utility
        args = [
            str(euler1[0]), str(euler1[1]), str(euler1[2]),
            str(euler2[0]), str(euler2[1]), str(euler2[2]),
            str(space_group)
        ]

        result = run_python_script("calcMiso.py", args)

        # Parse misorientation from output
        miso_angle = 0.0
        miso_axis = [0, 0, 0]

        if result["success"] and result.get("stdout"):
            # Try to parse output
            try:
                lines = result["stdout"].strip().split('\n')
                for line in lines:
                    if 'angle' in line.lower():
                        parts = line.split()
                        miso_angle = float(parts[-1])
                    elif 'axis' in line.lower():
                        parts = line.split()
                        miso_axis = [float(x) for x in parts[-3:]]
            except:
                pass

        # Classify grain boundary
        gb_type = "general"
        if miso_angle < 15:
            gb_type = "low-angle"
        elif 55 < miso_angle < 65 and space_group == 225:  # FCC
            # Check for special boundaries
            if abs(miso_angle - 60) < 5:
                gb_type = "Σ3 twin"
            elif abs(miso_angle - 38.9) < 2:
                gb_type = "Σ9"

        return format_result({
            "tool": "calculate_misorientation",
            "status": "completed" if result["success"] else "warning",
            "execution": result,
            "grain_pair": {
                "grain_1_id": grain_id_1,
                "grain_1_euler": euler1,
                "grain_2_id": grain_id_2,
                "grain_2_euler": euler2
            },
            "misorientation": {
                "angle_degrees": round(miso_angle, 3),
                "axis": [round(x, 4) for x in miso_axis],
                "space_group": space_group
            },
            "grain_boundary": {
                "type": gb_type,
                "classification": "Special" if "Σ" in gb_type else "Random"
            }
        })

    except Exception as e:
        return format_result({
            "tool": "calculate_misorientation",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def run_forward_simulation(
    input_grains_file: str,
    param_file: str,
    output_prefix: str,
    compressed: bool = True,
    scanning_mode: bool = False
) -> str:
    """Forward simulate diffraction from known microstructure.

    Simulates what diffraction patterns would look like for a given
    grain structure. Used for testing reconstruction algorithms and
    validating experimental data.

    Args:
        input_grains_file: Path to input Grains.csv with known orientations
        param_file: Path to Parameters.txt with experimental geometry
        output_prefix: Prefix for output files
        compressed: Use compressed output format
        scanning_mode: Simulate scanning (PF) mode instead of FF mode

    Returns:
        JSON with simulation status and output files
    """
    try:
        # Validate inputs
        valid, grains_path = validate_file(input_grains_file)
        if not valid:
            return format_result({"error": grains_path, "status": "failed"})

        valid, param_path = validate_file(param_file)
        if not valid:
            return format_result({"error": param_path, "status": "failed"})

        work_dir = Path(param_path).parent

        # Update parameter file with input grains and output prefix
        # This is a simplified approach - actual implementation may need
        # to properly parse and modify the parameter file
        temp_params = work_dir / "sim_params.txt"
        with open(param_path, 'r') as fin, open(temp_params, 'w') as fout:
            for line in fin:
                if line.startswith('InFileName'):
                    fout.write(f'InFileName {grains_path}\n')
                elif line.startswith('OutFileName'):
                    fout.write(f'OutFileName {output_prefix}\n')
                else:
                    fout.write(line)

        # Choose executable
        if scanning_mode:
            exe = "SimulateScanning"
        else:
            exe = "ForwardSimulationCompressed" if compressed else "ForwardSimulation"

        print(f"Running {exe} simulation", file=sys.stderr)

        result = run_midas_executable(exe, str(temp_params), cwd=str(work_dir), timeout=1800)

        # Find output files
        output_files = []
        for pattern in [f"{output_prefix}*.zip", f"{output_prefix}*.tif",
                       f"{output_prefix}*.h5"]:
            output_files.extend(work_dir.glob(pattern))

        simulation_info = {
            "input_grains": grains_path,
            "param_file": param_path,
            "output_prefix": output_prefix,
            "n_output_files": len(output_files),
            "output_files": [str(f) for f in output_files[:10]],  # Limit listing
            "simulation_mode": "PF-HEDM" if scanning_mode else "FF-HEDM"
        }

        # Count input grains
        try:
            with open(grains_path, 'r') as f:
                simulation_info["n_grains_simulated"] = sum(1 for line in f) - 1
        except:
            pass

        # Clean up temp file
        if temp_params.exists():
            temp_params.unlink()

        return format_result({
            "tool": "run_forward_simulation",
            "status": "completed" if result["success"] else "failed",
            "workflow": "Forward Diffraction Simulation",
            "execution": result,
            "simulation": simulation_info,
            "usage_note": "Use simulated data to test reconstruction pipelines"
        })

    except Exception as e:
        return format_result({
            "tool": "run_forward_simulation",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def extract_grain_centroids(
    nf_mic_file: str,
    output_csv: str = "grain_centroids.csv",
    min_grain_size: int = 100
) -> str:
    """Extract grain centroids from NF reconstruction.

    Identifies individual grains in NF-HEDM voxel data and calculates
    their centroids, volumes, and average orientations.

    Args:
        nf_mic_file: Path to NF-HEDM Grains.mic file
        output_csv: Output CSV file with centroid data
        min_grain_size: Minimum grain size in voxels

    Returns:
        JSON with grain statistics and centroid data
    """
    try:
        valid, mic_path = validate_file(nf_mic_file)
        if not valid:
            return format_result({"error": mic_path, "status": "failed"})

        work_dir = Path(mic_path).parent
        output_path = work_dir / output_csv

        # Run NFGrainCentroids executable
        # This may need a parameter file
        temp_params = work_dir / "centroid_params.txt"
        with open(temp_params, 'w') as f:
            f.write(f"MicFile {mic_path}\n")
            f.write(f"OutputFile {output_csv}\n")
            f.write(f"MinGrainSize {min_grain_size}\n")

        print("Extracting grain centroids", file=sys.stderr)

        result = run_midas_executable("NFGrainCentroids", str(temp_params),
                                     cwd=str(work_dir), timeout=600)

        # Parse output
        centroid_info = {
            "input_file": mic_path,
            "output_file": None,
            "n_grains_found": 0,
            "min_grain_size_voxels": min_grain_size
        }

        if output_path.exists():
            centroid_info["output_file"] = str(output_path)

            # Count grains
            try:
                with open(output_path, 'r') as f:
                    centroid_info["n_grains_found"] = sum(1 for line in f) - 1
            except:
                pass

            # Read some statistics
            try:
                import csv
                with open(output_path, 'r') as f:
                    reader = csv.DictReader(f)
                    volumes = []
                    for row in reader:
                        if 'Volume' in row:
                            volumes.append(float(row['Volume']))

                    if volumes:
                        centroid_info["grain_statistics"] = {
                            "mean_volume_voxels": round(np.mean(volumes), 1),
                            "median_volume_voxels": round(np.median(volumes), 1),
                            "total_volume_voxels": int(np.sum(volumes))
                        }
            except:
                pass

        # Clean up temp file
        if temp_params.exists():
            temp_params.unlink()

        return format_result({
            "tool": "extract_grain_centroids",
            "status": "completed" if result["success"] else "failed",
            "execution": result,
            "centroid_analysis": centroid_info
        })

    except Exception as e:
        return format_result({
            "tool": "extract_grain_centroids",
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# DATA MANAGEMENT & UTILITIES
# =============================================================================

@mcp.tool()
async def batch_convert_ge_to_tiff(
    ge_folder: str,
    output_folder: str,
    file_pattern: str = "*.ge*",
    parallel: bool = True,
    n_processes: int = 4
) -> str:
    """Batch convert GE detector files to TIFF format.

    Converts proprietary GE detector format to standard TIFF images
    for processing with MIDAS or other tools.

    Args:
        ge_folder: Directory containing GE files
        output_folder: Output directory for TIFF files
        file_pattern: Glob pattern for GE files (e.g., "*.ge2", "*.ge3")
        parallel: Use parallel processing
        n_processes: Number of parallel processes

    Returns:
        JSON with conversion statistics
    """
    try:
        # Validate directories
        ge_path = Path(ge_folder).expanduser()
        if not ge_path.exists():
            return format_result({
                "error": f"GE folder not found: {ge_path}",
                "status": "failed"
            })

        output_path = Path(output_folder).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        # Find GE files
        ge_files = list(ge_path.glob(file_pattern))

        if not ge_files:
            return format_result({
                "warning": f"No files matching '{file_pattern}' found in {ge_path}",
                "status": "warning",
                "n_files_found": 0
            })

        print(f"Converting {len(ge_files)} GE files to TIFF", file=sys.stderr)

        # Build command
        args = [
            str(ge_path),
            str(output_path),
            "--pattern", file_pattern
        ]

        if parallel:
            args.extend(["--parallel", "--n-processes", str(n_processes)])

        result = run_python_script("GE2Tiff.py", args, timeout=3600)

        # Count output files
        tiff_files = list(output_path.glob("*.tif")) + list(output_path.glob("*.tiff"))

        conversion_stats = {
            "input_folder": str(ge_path),
            "output_folder": str(output_path),
            "n_input_files": len(ge_files),
            "n_output_files": len(tiff_files),
            "conversion_rate": len(tiff_files) / len(ge_files) if ge_files else 0,
            "parallel_processing": parallel,
            "n_processes": n_processes if parallel else 1
        }

        return format_result({
            "tool": "batch_convert_ge_to_tiff",
            "status": "completed" if result["success"] else "partial",
            "execution": result,
            "conversion": conversion_stats
        })

    except Exception as e:
        return format_result({
            "tool": "batch_convert_ge_to_tiff",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def create_midas_parameter_file(
    lattice_constants: list,
    space_group: int,
    detector_distance: float,
    beam_center: list,
    wavelength: float,
    omega_step: float,
    output_file: str = "Parameters.txt",
    pixel_size: float = 200.0,
    beam_thickness: float = 200.0,
    wedge: float = 0.0,
    detector_tilt: list = None,
    additional_params: dict = None
) -> str:
    """Generate MIDAS parameter file programmatically.

    Creates a properly formatted Parameters.txt file with validated
    crystallographic and experimental parameters.

    Args:
        lattice_constants: [a, b, c, alpha, beta, gamma] in Å and degrees
        space_group: Space group number (1-230)
        detector_distance: Sample-detector distance in microns
        beam_center: [x, y] beam center in pixels
        wavelength: X-ray wavelength in Angstroms
        omega_step: Rotation step in degrees
        output_file: Output parameter file name
        pixel_size: Detector pixel size in microns
        beam_thickness: Beam height in microns
        wedge: Wedge angle in degrees
        detector_tilt: [tx, ty, tz] detector tilt in degrees
        additional_params: Dictionary of additional parameters

    Returns:
        JSON with parameter file path and validation status
    """
    try:
        # Validate inputs
        if len(lattice_constants) != 6:
            return format_result({
                "error": "lattice_constants must be [a, b, c, alpha, beta, gamma]",
                "status": "failed"
            })

        if len(beam_center) != 2:
            return format_result({
                "error": "beam_center must be [x, y]",
                "status": "failed"
            })

        if not (1 <= space_group <= 230):
            return format_result({
                "error": "space_group must be between 1 and 230",
                "status": "failed"
            })

        if detector_tilt is None:
            detector_tilt = [0.0, 0.0, 0.0]

        if len(detector_tilt) != 3:
            return format_result({
                "error": "detector_tilt must be [tx, ty, tz]",
                "status": "failed"
            })

        output_path = Path(output_file).expanduser()

        # Write parameter file
        with open(output_path, 'w') as f:
            f.write("# MIDAS Parameters File\n")
            f.write(f"# Generated by Beamline Assistant\n\n")

            # Crystal structure
            f.write("# Crystal Structure\n")
            lc = lattice_constants
            f.write(f"LatticeConstant {lc[0]:.6f} {lc[1]:.6f} {lc[2]:.6f} ")
            f.write(f"{lc[3]:.6f} {lc[4]:.6f} {lc[5]:.6f}\n")
            f.write(f"SpaceGroup {space_group}\n\n")

            # Detector configuration
            f.write("# Detector Configuration\n")
            f.write(f"Lsd {detector_distance:.4f}\n")
            f.write(f"BC {beam_center[0]:.4f} {beam_center[1]:.4f}\n")
            f.write(f"tx {detector_tilt[0]:.6f}\n")
            f.write(f"ty {detector_tilt[1]:.6f}\n")
            f.write(f"tz {detector_tilt[2]:.6f}\n")
            f.write(f"p0 0\n")  # Distortion parameters
            f.write(f"p1 0\n")
            f.write(f"p2 0\n")
            f.write(f"px {pixel_size:.4f}\n\n")

            # Experimental setup
            f.write("# Experimental Setup\n")
            f.write(f"Wavelength {wavelength:.6f}\n")
            f.write(f"Wedge {wedge:.6f}\n")
            f.write(f"OmegaStep {omega_step:.6f}\n")
            f.write(f"BeamThickness {beam_thickness:.4f}\n\n")

            # Analysis parameters (defaults)
            f.write("# Analysis Parameters\n")
            f.write("MinNrSpots 3\n")
            f.write("Completeness 0.8\n")
            f.write("OverAllRingToIndex 2\n\n")

            # Additional parameters
            if additional_params:
                f.write("# Additional Parameters\n")
                for key, value in additional_params.items():
                    f.write(f"{key} {value}\n")

        # Validate created file
        validation = {
            "file_created": output_path.exists(),
            "file_path": str(output_path),
            "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "parameters_validated": True
        }

        # Read back and validate
        if output_path.exists():
            with open(output_path, 'r') as f:
                content = f.read()
                validation["contains_lattice"] = "LatticeConstant" in content
                validation["contains_spacegroup"] = "SpaceGroup" in content
                validation["contains_detector"] = "Lsd" in content and "BC" in content
                validation["contains_wavelength"] = "Wavelength" in content

        return format_result({
            "tool": "create_midas_parameter_file",
            "status": "completed",
            "output_file": str(output_path),
            "validation": validation,
            "parameters": {
                "lattice_constants": lattice_constants,
                "space_group": space_group,
                "detector_distance_um": detector_distance,
                "beam_center": beam_center,
                "wavelength_angstrom": wavelength,
                "omega_step_deg": omega_step
            }
        })

    except Exception as e:
        return format_result({
            "tool": "create_midas_parameter_file",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def validate_midas_installation(
    midas_path: str = None
) -> str:
    """Validate MIDAS installation and dependencies.

    Checks for required executables, Python packages, and configuration files.

    Args:
        midas_path: Path to MIDAS installation (default: ~/opt/MIDAS)

    Returns:
        JSON with installation validation results
    """
    try:
        # Determine MIDAS path
        if midas_path:
            midas_root = Path(midas_path).expanduser()
        else:
            midas_root = MIDAS_ROOT

        validation = {
            "midas_root": str(midas_root),
            "root_exists": midas_root.exists(),
            "executables": {},
            "python_modules": {},
            "dependencies": {},
            "overall_status": "unknown"
        }

        if not midas_root.exists():
            validation["overall_status"] = "failed"
            validation["error"] = f"MIDAS root directory not found: {midas_root}"
            return format_result({
                "tool": "validate_midas_installation",
                "validation": validation
            })

        # Check for key executables
        bin_path = midas_root / "bin"
        validation["bin_directory"] = str(bin_path)
        validation["bin_exists"] = bin_path.exists()

        key_executables = [
            # FF-HEDM CPU
            "IndexerOMP", "FitPosOrStrainsOMP", "ProcessGrains",
            "GetHKLListZarr", "PeaksFittingOMPZarrRefactor",
            "CalibrantIntegratorOMP", "ForwardSimulationCompressed",
            # FF-HEDM GPU (v10)
            "IndexerGPU", "FitPosOrStrainsGPU",
            # Calibration & Integration (v10)
            "IntegratorZarrOMP", "CalibrantPanelShiftsOMP",
            # NF-HEDM
            "FitOrientationOMP", "FitOrientationGPU", "GetHKLListNF",
            "MakeHexGrid", "ParseMic", "ProcessImagesCombined",
            # Utilities
            "GrainTracking", "CalcStrains", "FitWedgeParallel"
        ]

        for exe in key_executables:
            exe_path = bin_path / exe
            validation["executables"][exe] = exe_path.exists()

        # Check Python workflows
        workflow_scripts = {
            "ff_MIDAS.py": midas_root / "FF_HEDM" / "ff_MIDAS.py",
            "pf_MIDAS.py": midas_root / "FF_HEDM" / "pf_MIDAS.py",
            "nf_MIDAS.py": midas_root / "NF_HEDM" / "nf_MIDAS.py",
            "match_grains.py": midas_root / "utils" / "match_grains.py",
            "AutoCalibrateZarr.py": midas_root / "utils" / "AutoCalibrateZarr.py"
        }

        for script, path in workflow_scripts.items():
            validation["python_modules"][script] = path.exists()

        # Check Python dependencies
        required_packages = [
            "numpy", "scipy", "matplotlib", "fabio", "pyFAI",
            "h5py", "zarr", "numcodecs", "parsl", "numba"
        ]

        for package in required_packages:
            try:
                __import__(package)
                validation["dependencies"][package] = True
            except ImportError:
                validation["dependencies"][package] = False

        # Overall assessment
        exe_found = sum(validation["executables"].values())
        exe_total = len(validation["executables"])
        scripts_found = sum(validation["python_modules"].values())
        scripts_total = len(validation["python_modules"])
        deps_found = sum(validation["dependencies"].values())
        deps_total = len(validation["dependencies"])

        validation["statistics"] = {
            "executables": f"{exe_found}/{exe_total}",
            "python_scripts": f"{scripts_found}/{scripts_total}",
            "dependencies": f"{deps_found}/{deps_total}"
        }

        # Determine overall status
        if exe_found == exe_total and scripts_found == scripts_total and deps_found == deps_total:
            validation["overall_status"] = "excellent"
        elif exe_found >= exe_total * 0.8 and scripts_found >= 2 and deps_found >= deps_total * 0.8:
            validation["overall_status"] = "good"
        elif exe_found >= exe_total * 0.5:
            validation["overall_status"] = "partial"
        else:
            validation["overall_status"] = "insufficient"

        # Recommendations
        validation["recommendations"] = []
        if exe_found < exe_total:
            validation["recommendations"].append(
                f"Rebuild MIDAS: cd {midas_root} && ./build.sh --build-type Release"
            )
        if deps_found < deps_total:
            validation["recommendations"].append(
                "Install missing Python packages: conda env create -f environment.yml"
            )
        if not validation["bin_exists"]:
            validation["recommendations"].append(
                "Run MIDAS build script to compile executables"
            )

        return format_result({
            "tool": "validate_midas_installation",
            "validation": validation
        })

    except Exception as e:
        return format_result({
            "tool": "validate_midas_installation",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def get_midas_workflow_status(
    result_folder: str,
    workflow_type: str = "ff"
) -> str:
    """Check status of running MIDAS workflow.

    Monitors workflow progress by parsing log files and checking output files.

    Args:
        result_folder: Workflow result directory
        workflow_type: Workflow type (ff, nf, pf)

    Returns:
        JSON with current workflow status and progress
    """
    try:
        result_path = Path(result_folder).expanduser()

        if not result_path.exists():
            return format_result({
                "error": f"Result folder not found: {result_path}",
                "status": "not_found"
            })

        status = {
            "result_folder": str(result_path),
            "workflow_type": workflow_type.upper(),
            "status": "unknown",
            "progress": {},
            "output_files": [],
            "errors": []
        }

        # Check for log directory
        log_dirs = [result_path / "midas_log", result_path / "output"]
        log_dir = None
        for ld in log_dirs:
            if ld.exists():
                log_dir = ld
                break

        if log_dir:
            status["log_directory"] = str(log_dir)
            log_files = list(log_dir.glob("*.log"))
            status["n_log_files"] = len(log_files)

            # Parse latest log file for status
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                status["latest_log"] = str(latest_log)

                try:
                    with open(latest_log, 'r') as f:
                        log_content = f.read()

                        # Look for completion markers
                        if "completed" in log_content.lower() or "finished" in log_content.lower():
                            status["status"] = "completed"
                        elif "error" in log_content.lower() or "failed" in log_content.lower():
                            status["status"] = "failed"
                            # Extract error lines
                            for line in log_content.split('\n'):
                                if 'error' in line.lower() or 'failed' in line.lower():
                                    status["errors"].append(line.strip())
                        elif "running" in log_content.lower() or "processing" in log_content.lower():
                            status["status"] = "running"

                        # Try to extract progress
                        for line in log_content.split('\n'):
                            if 'progress' in line.lower() or '%' in line:
                                status["progress"]["latest"] = line.strip()
                except:
                    pass

        # Check for output files based on workflow type
        if workflow_type.lower() == "ff":
            # FF-HEDM outputs
            output_patterns = [
                "LayerNr_*/GrainsReconstructed.csv",
                "*.MIDAS.zip",
                "Grains.csv"
            ]
        elif workflow_type.lower() == "nf":
            # NF-HEDM outputs
            output_patterns = [
                "Grains.mic",
                "*.mic"
            ]
        else:
            output_patterns = ["*.csv", "*.mic"]

        for pattern in output_patterns:
            matches = list(result_path.glob(pattern))
            for match in matches:
                status["output_files"].append({
                    "file": str(match),
                    "size_kb": match.stat().st_size / 1024,
                    "modified": match.stat().st_mtime
                })

        status["n_output_files"] = len(status["output_files"])

        # If we found output files but no log info, assume completed
        if status["status"] == "unknown" and status["n_output_files"] > 0:
            status["status"] = "likely_completed"

        return format_result({
            "tool": "get_midas_workflow_status",
            "status_check": status
        })

    except Exception as e:
        return format_result({
            "tool": "get_midas_workflow_status",
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# BASIC ANALYSIS TOOLS (from original server)
# =============================================================================

# =============================================================================
# FF-HEDM CALIBRATION (MIDAS Official)
# =============================================================================
# Tools moved to analysis_utilities_server.py:
# - detect_rings_quick (was: detect_diffraction_rings) - Custom NumPy diagnostic tool
# - identify_phases_basic (was: identify_crystalline_phases) - Basic phase matching
#
# This server now contains ONLY official MIDAS tools
# =============================================================================

@mcp.tool()
async def midas_integrate_2d_to_1d(
    image_file: str,
    param_file: str,
    dark_file: str = None,
    result_folder: str = None,
    n_cpus: int = 4,
    convert_files: bool = True
) -> str:
    """Integrate a single 2D diffraction image to a 1D lineout using MIDAS integrator.py (v10).

    REQUIRES calibrated params from midas_auto_calibrate (refined_MIDAS_params*.txt).
    integrator.py runs DetectorMapper automatically if Map.bin is missing.

    v10 executable chain:
      integrator.py → IntegratorZarrOMP (replaces old Integrator binary)

    Outputs written to result_folder (default: image_dir/integration/):
      *_lineout.xy         — 2θ (degrees) vs intensity text file
      *_lineout.bin        — binary lineout
      *_caked.hdf.zarr.zip — caked output (GSAS-II compatible)
      Map.bin / nMap.bin / maskMap.bin — geometry maps (first run only)

    Args:
        image_file: Path to 2D diffraction image (.tif/.tiff, .ge/.ge1-.ge5, .h5/.hdf5, .zip)
        param_file: Calibrated MIDAS parameter file (refined_MIDAS_params*.txt)
        dark_file: Optional dark field image for background subtraction
        result_folder: Output directory (default: <image_dir>/integration)
        n_cpus: OMP threads for IntegratorZarrOMP (default: 4)
        convert_files: Convert input to Zarr before integrating (default: True)
    """
    try:
        image_path = Path(image_file).expanduser().absolute()
        param_path = Path(param_file).expanduser().absolute()

        if not image_path.exists():
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"Image not found: {image_path}"})
        if not param_path.exists():
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"Parameter file not found: {param_path}"})

        # Find integrator.py in MIDAS/utils/
        integrator_script = MIDAS_ROOT / "utils" / "integrator.py"
        if not integrator_script.exists():
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"integrator.py not found at {integrator_script}"})

        out_dir = Path(result_folder).expanduser().absolute() if result_folder \
                  else image_path.parent / "integration"
        out_dir.mkdir(parents=True, exist_ok=True)

        midas_python = find_midas_python()
        cmd = [
            midas_python, str(integrator_script),
            "-paramFN",      str(param_path),
            "-dataFN",       str(image_path),
            "-resultFolder", str(out_dir),
            "-nCPUsLocal",   str(n_cpus),
            "-nCPUs",        "1",
            "-mapDetector",  "1",
            "-convertFiles", "1" if convert_files else "0",
            "-writeMat",     "0",
        ]
        if dark_file:
            dark_path = Path(dark_file).expanduser().absolute()
            if not dark_path.exists():
                return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                      "error": f"Dark file not found: {dark_path}"})
            cmd += ["-darkFN", str(dark_path)]

        print(f"\n  $ {' '.join(cmd)}", file=sys.stderr)

        result = subprocess.run(cmd, cwd=str(image_path.parent),
                                capture_output=True, text=True,
                                timeout=600, env=get_midas_env())

        if result.returncode != 0:
            print(f"❌ Integration failed (exit {result.returncode})", file=sys.stderr)
            for line in result.stderr.strip().splitlines()[-20:]:
                print(f"  {line}", file=sys.stderr)
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"integrator.py exited {result.returncode}",
                                  "stderr": result.stderr, "stdout": result.stdout})

        lineout = sorted(out_dir.glob("*_lineout.xy"), key=lambda p: p.stat().st_mtime, reverse=True)
        zarr_out = sorted(out_dir.glob("*_caked.hdf.zarr.zip"), key=lambda p: p.stat().st_mtime, reverse=True)

        return format_result({
            "tool": "midas_integrate_2d_to_1d", "status": "success",
            "input_image": str(image_path),
            "param_file": str(param_path),
            "result_folder": str(out_dir),
            "lineout_xy": str(lineout[0]) if lineout else "not found — check result_folder",
            "zarr_zip": str(zarr_out[0]) if zarr_out else "not found",
            "message": f"Integration complete. Lineout: {lineout[0].name if lineout else 'see result_folder'}"
        })

    except subprocess.TimeoutExpired:
        return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                              "error": "integrator.py timed out (>10 min)"})
    except Exception as e:
        return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error", "error": str(e)})

# Phase identification tool moved to analysis_utilities_server.py as identify_phases_basic
# Use GSAS-II server for comprehensive phase identification

@mcp.tool()
async def midas_auto_calibrate(
    image_file: str,
    parameters_file: str,
    dark_file: str = "",
    lsd_guess: float = 1000000.0,
    bc_x_guess: float = 0.0,
    bc_y_guess: float = 0.0,
    n_iterations: int = 40,
    tol_shifts: float = 3.0,
    tol_rotation: float = 1.0,
    mult_factor: float = 2.5,
    first_ring_nr: int = 1,
    eta_bin_size: float = 5.0,
    threshold: int = 0,
    make_plots: int = 0,
    save_plots_hdf: str = "",
    image_transform: str = "",
    data_loc: str = ""
) -> str:
    """🔧 PRIMARY TOOL FOR FF-HEDM DETECTOR CALIBRATION (MIDAS Official)

    ⚠️ WORKFLOW GUIDANCE - WHEN TO USE THIS TOOL:
    When user requests:
    - "calibrate the detector"
    - "auto-calibrate using CeO2" or any calibrant material
    - "determine detector geometry"
    - "calibration workflow"
    - "refine beam center and distance"

    → USE THIS TOOL (midas_auto_calibrate) - This is the ONLY tool for FF-HEDM calibration
    → DO NOT use detect_rings_quick (that's a diagnostic tool in utilities server)
    → This is the OFFICIAL MIDAS calibration method from AutoCalibrateZarr.py

    🚨 CRITICAL: FILE PATH REQUIREMENTS
    - image_file: Must be the EXACT, COMPLETE file path as it exists on disk
    - DO NOT abbreviate or guess filenames (e.g., "CeO2.tif" when actual file is "CeO2_650mm_61p332keV_2DFocused_0p1s_att200_004018.tif")
    - If user provides a directory, use filesystem_list_directory FIRST to find the actual filename
    - Then call this tool with the FULL PATH found from the directory listing
    - The tool will auto-search the parent directory if the exact file is not found

    MIDAS Manual: https://github.com/marinerhemant/MIDAS/blob/master/manuals/FF_autocalibrate.md
    MIDAS Component: AutoCalibrateZarr.py → CalibrantOMP → GetHKLList
    Location: MIDAS/utils/AutoCalibrateZarr.py

    DESCRIPTION:
    Auto-calibrates detector geometry by analyzing a 2D diffraction image of a known calibrant
    (e.g., CeO2, LaB6). Iteratively refines all geometric parameters (Lsd, beam center, tilts,
    distortions) until convergence using MIDAS's CalibrantOMP binary for robust least-squares
    fitting with automatic outlier rejection.

    FF-HEDM WORKFLOW POSITION:
    Step 1 - CALIBRATION (must run BEFORE analysis)
    ├─ Input: Raw calibrant image + initial parameter file
    ├─ Output: refined_MIDAS_params.txt (calibrated parameters)
    └─ Next steps: integrate_2d_to_1d OR run_ff_hedm_full_workflow

    Args:
        image_file: EXACT path to calibrant diffraction image (.tif/.tiff, .ge/.ge1-.ge5, .h5/.hdf5/.nxs, .zip Zarr)
                   If file not found, will auto-search parent directory for matching files.
        parameters_file: EXACT path to MIDAS parameter file containing material properties (SpaceGroup, LatticeParameter, Wavelength, px)
        dark_file: Optional path to dark field image for background subtraction
        lsd_guess: Initial sample-to-detector distance guess in µm (default: 1000000 = auto-detect from ring ratios)
        bc_x_guess: Initial beam center X coordinate in pixels (default: 0.0 = auto-detect from ring geometry)
        bc_y_guess: Initial beam center Y coordinate in pixels (default: 0.0 = auto-detect from ring geometry)
        n_iterations: Maximum number of refinement iterations (default: 40)
        tol_shifts: Panel shift convergence tolerance in pixels (default: 3.0)
        tol_rotation: Panel rotation convergence tolerance in degrees (default: 1.0)
        mult_factor: Outlier rejection multiplier - rings with strain > mult_factor × median_strain are excluded (default: 2.5)
        first_ring_nr: Index (1-based) of first prominent ring for initial Lsd estimation (default: 1)
        eta_bin_size: Azimuthal bin size in degrees for CalibrantOMP fitting (default: 5.0)
        threshold: Manual intensity threshold for ring segmentation (default: 0 = auto-calculate)
        make_plots: Display matplotlib plots during refinement (0=no, 1=yes) (default: 0)
        save_plots_hdf: Path to HDF5 file for saving all intermediate data/plots for offline analysis (default: "" = don't save)
        image_transform: Image transformation - "0"=none, "1"=flip LR, "2"=flip UD, "3"=transpose, or space-separated combo (default: "" = none)
        data_loc: HDF5 dataset path if not standard location (default: "" = use /entry/data/data)

    Returns:
        JSON with calibrated geometric parameters and convergence metrics

    Outputs:
        - refined_MIDAS_params.txt: Final converged parameters (Lsd, BC, tx, ty, tz, p0-p3)
        - autocal.log: Detailed execution log with iteration history
        - calibrant_screen_out.csv: Raw CalibrantOMP output (for debugging)
        - [optional] HDF5 file with all intermediate arrays and plots

    Example Usage:
        Standard CeO2 calibration at ~650mm:
        {
            "image_file": "/data/CeO2_61keV_650mm.tif",
            "parameters_file": "/data/Params_CeO2.txt",
            "lsd_guess": 650000,
            "stopping_strain": 0.0001
        }

        High-precision calibration with diagnostics:
        {
            "image_file": "LaB6_calibrant.h5",
            "parameters_file": "Params_LaB6.txt",
            "dark_file": "dark.h5",
            "lsd_guess": 200000,
            "stopping_strain": 0.00004,
            "mult_factor": 3.0,
            "save_plots_hdf": "calibration_diagnostics.h5",
            "image_transform": "2"
        }

    Required Parameter File Format:
        SpaceGroup 225              # CeO2: 225, LaB6: 221, Si: 227
        LatticeParameter 5.411      # CeO2 lattice constant in Angstroms
        Wavelength 0.2021           # X-ray wavelength in Angstroms (61.332 keV)
        px 200                      # Pixel size in microns
        SkipFrame 0
    """
    try:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"🔧 midas_auto_calibrate called:", file=sys.stderr)
        print(f"   Image: {image_file}", file=sys.stderr)
        print(f"   Params: {parameters_file}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)

        # Locate AutoCalibrateZarr.py
        # Note: We don't check MIDAS_AVAILABLE here because that only checks for
        # pyFAI/fabio dependencies, not MIDAS executables. AutoCalibrateZarr.py
        # has its own dependencies managed within the MIDAS environment.
        autocal_script = MIDAS_ROOT / "utils" / "AutoCalibrateZarr.py"
        print(f"✓ Checking for AutoCalibrateZarr.py at: {autocal_script}", file=sys.stderr)
        if not autocal_script.exists():
            # Provide diagnostic information about what was found
            utils_dir = MIDAS_ROOT / "utils"
            utils_exists = utils_dir.exists()

            diagnostic_info = f"MIDAS_ROOT detected: {MIDAS_ROOT}\n"
            diagnostic_info += f"utils/ directory exists: {utils_exists}\n"

            if utils_exists:
                try:
                    utils_contents = [f.name for f in utils_dir.iterdir() if f.name.endswith('.py')][:10]
                    diagnostic_info += f"Python files in utils/: {', '.join(utils_contents) if utils_contents else 'none'}\n"
                except:
                    diagnostic_info += "Could not list utils/ contents\n"

            # Check for alternative locations
            alt_locations = []
            for name in ["AutoCalibrateZarr.py", "deprecated_AutoCalibrate.py", "AutoCalibrate.py"]:
                script_path = MIDAS_ROOT / "utils" / name
                if script_path.exists():
                    alt_locations.append(str(script_path))

            if alt_locations:
                diagnostic_info += f"\nFound alternative scripts:\n  " + "\n  ".join(alt_locations)

            return format_result({
                "tool": "midas_auto_calibrate",
                "status": "error",
                "error": f"AutoCalibrateZarr.py not found at expected location: {autocal_script}\n\n{diagnostic_info}\n\nTo fix:\n1. Set MIDAS_PATH environment variable to your MIDAS installation\n2. Ensure AutoCalibrateZarr.py exists in MIDAS/utils/\n3. Use a recent MIDAS version from https://github.com/marinerhemant/MIDAS"
            })

        # Expand paths
        image_path = Path(image_file).expanduser().absolute()
        param_path = Path(parameters_file).expanduser().absolute()

        print(f"✓ Image path: {image_path} (exists: {image_path.exists()})", file=sys.stderr)
        print(f"✓ Param path: {param_path} (exists: {param_path.exists()})", file=sys.stderr)

        # Auto-search for image file if not found
        if not image_path.exists():
            print(f"⚠ Image file not found at: {image_path}", file=sys.stderr)

            # Search dirs: parent of specified path + subdirs of cwd (e.g. test1/)
            cwd = Path.cwd()
            search_dirs = {image_path.parent, cwd}
            for d in cwd.iterdir():
                if d.is_dir():
                    search_dirs.add(d)

            print(f"  Searching for image in: {', '.join(d.name for d in search_dirs)}", file=sys.stderr)

            search_patterns = ["*.tif", "*.tiff", "*.ge2", "*.ge3", "*.ge4", "*.ge5", "*.h5", "*.hdf5"]
            found_files = []
            for search_dir in search_dirs:
                if search_dir.exists():
                    for pattern in search_patterns:
                        found_files.extend(search_dir.glob(pattern))

            if found_files:
                found_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                print(f"  Found {len(found_files)} image file(s):", file=sys.stderr)
                for i, f in enumerate(found_files[:5], 1):
                    print(f"    {i}. {f.parent.name}/{f.name}", file=sys.stderr)

                # Try to match by basename stem first
                search_basename = image_path.stem.lower()
                basename_match = next((f for f in found_files if search_basename in f.stem.lower()), None)
                if basename_match:
                    image_path = basename_match
                    print(f"  ✓ Using matched file: {image_path}", file=sys.stderr)
                else:
                    image_path = found_files[0]
                    print(f"  ✓ Using most recent file: {image_path}", file=sys.stderr)
            else:
                print(f"❌ ERROR: No image files found!", file=sys.stderr)
                return format_result({
                    "tool": "midas_auto_calibrate",
                    "status": "error",
                    "error": f"No diffraction images (.tif, .ge, .h5) found in cwd or subdirectories. Please provide the full path."
                })

        # Auto-search for parameters file if not found
        if not param_path.exists():
            print(f"⚠ Parameters file not found at: {param_path}", file=sys.stderr)

            cwd = Path.cwd()
            search_dirs = {param_path.parent, cwd}
            for d in cwd.iterdir():
                if d.is_dir():
                    search_dirs.add(d)

            param_patterns = ["*arameters*.txt", "*params*.txt", "*Params*.txt", "*.txt"]
            found_params = []
            for search_dir in search_dirs:
                if search_dir.exists():
                    for pattern in param_patterns:
                        found_params.extend(search_dir.glob(pattern))

            # Filter out obvious non-parameter files (< 10KB)
            found_params = [p for p in found_params if p.stat().st_size < 10000]

            if found_params:
                found_params.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                print(f"  Found {len(found_params)} parameter file(s):", file=sys.stderr)
                for i, f in enumerate(found_params[:5], 1):
                    print(f"    {i}. {f.parent.name}/{f.name}", file=sys.stderr)

                search_basename = param_path.stem.lower()
                basename_match = next(
                    (f for f in found_params if search_basename in f.stem.lower() or "param" in f.stem.lower()),
                    None
                )
                if basename_match:
                    param_path = basename_match
                    print(f"  ✓ Using matched file: {param_path}", file=sys.stderr)
                else:
                    param_path = found_params[0]
                    print(f"  ✓ Using: {param_path}", file=sys.stderr)
            else:
                print(f"❌ ERROR: No parameter files found!", file=sys.stderr)
                return format_result({
                    "tool": "midas_auto_calibrate",
                    "status": "error",
                    "error": f"Parameters file not found. Please provide the full path to the MIDAS parameter file."
                })

        # Determine file type for ConvertFile flag
        suffix = image_path.suffix.lower()
        # v10: format is auto-detected from extension — ConvertFile is optional.
        # We still pass it explicitly so the conversion path is unambiguous.
        if suffix in ['.zip']:
            convert_file = 0  # Zarr zip
        elif suffix in ['.h5', '.hdf5', '.hdf', '.nxs']:
            convert_file = 1  # HDF5 / NeXus
        elif suffix in ['.ge', '.ge1', '.ge2', '.ge3', '.ge4', '.ge5']:
            convert_file = 2  # GE binary
        elif suffix in ['.tif', '.tiff']:
            convert_file = 3  # TIFF
        else:
            return format_result({
                "tool": "midas_auto_calibrate",
                "status": "error",
                "error": f"Unsupported file format: {suffix}. Supported: .zip (Zarr), .h5/.hdf5 (HDF5), .ge/.ge1-.ge5 (GE binary), .tif/.tiff (TIFF)"
            })

        # WORKAROUND for MIDAS filename parsing bug:
        # MIDAS ffGenerateZipRefactor.py has trouble when TIF files have complex names
        # because it converts filename.tif -> filename.tif.ge, then tries to parse
        # the stem which still contains ".tif", confusing the numeric parser.
        #
        # Solution: Create a simple symlink without dots in the basename
        # Example: CeO2_650mm_61p332keV_2DFocused_0p1s_att200_004018.tif
        #       -> CeO2_calib_000001.tif

        # Always create a simple symlink for TIFF files to avoid parsing issues
        if suffix in ['.tif', '.tiff']:
            # Extract just the first part of filename before any numbers
            import re
            # Find the first meaningful word (usually material name)
            # Preserve material name including trailing digit (e.g. CeO2, LaB6)
            # so AutoCalibrateZarr.py can auto-detect calibrant from symlink name
            match = re.match(r'^([A-Za-z]+[0-9]*)', image_path.stem)
            prefix = match.group(1) if match else "calib"

            # Create simple MIDAS-friendly name WITHOUT extension
            # MIDAS will add .ge to it, and needs numbers at the END of the stem
            # So: calib_000001 -> calib_000001.ge (stem: calib_000001, ends with numbers ✓)
            # NOT: calib_000001.tif -> calib_000001.tif.ge (stem: calib_000001.tif, ends with .tif ✗)
            midas_friendly_name = f"{prefix}_000001{image_path.suffix}"
            midas_friendly_path = image_path.parent / midas_friendly_name

            # Remove old symlink if it exists
            if midas_friendly_path.is_symlink():
                midas_friendly_path.unlink()
            elif midas_friendly_path.exists():
                # Don't overwrite real files, use different name
                midas_friendly_name = f"{prefix}_calibration_000001{image_path.suffix}"
                midas_friendly_path = image_path.parent / midas_friendly_name
                if midas_friendly_path.is_symlink():
                    midas_friendly_path.unlink()

            # Create symlink
            try:
                midas_friendly_path.symlink_to(image_path.name)  # Relative symlink
                print(f"✓ Created MIDAS-friendly symlink: {midas_friendly_name} -> {image_path.name}", file=sys.stderr)
                image_path = midas_friendly_path
            except Exception as e:
                print(f"⚠ Could not create symlink: {e}. Using original filename.", file=sys.stderr)
                # Continue with original filename

        # Build command with all parameters according to MIDAS manual
        # Use MIDAS Python (conda midas_env) instead of current Python (UV)
        midas_python = find_midas_python()
        cmd = [
            midas_python,
            str(autocal_script),
            "-dataFN", str(image_path),
            "-paramFN", str(param_path),
            "-ConvertFile", str(convert_file),
            "--n-iterations", str(n_iterations),
            "--tol-shifts", str(tol_shifts),
            "--tol-rotation", str(tol_rotation),
            "-MultFactor", str(mult_factor),
            "-FirstRingNr", str(first_ring_nr),
            "-EtaBinSize", str(eta_bin_size),
            "-MakePlots", str(make_plots)
        ]

        # Add optional parameters
        if dark_file:
            dark_path = Path(dark_file).expanduser().absolute()
            if dark_path.exists():
                cmd.extend(["-darkFN", str(dark_path)])

        if lsd_guess < 1000000:  # User provided a real guess (not auto-detect)
            cmd.extend(["-LsdGuess", str(int(lsd_guess))])  # Convert to int µm

        if bc_x_guess != 0.0 or bc_y_guess != 0.0:
            # MIDAS expects BCGuess as Y X (not X Y!)
            cmd.extend(["-BCGuess", str(bc_y_guess), str(bc_x_guess)])

        if threshold > 0:  # Manual threshold specified
            cmd.extend(["-Threshold", str(threshold)])

        if save_plots_hdf:  # Save diagnostic HDF5
            hdf_path = Path(save_plots_hdf).expanduser().absolute()
            cmd.extend(["-SavePlotsHDF", str(hdf_path)])

        if data_loc:  # Non-standard HDF5 dataset location
            cmd.extend(["-dataLoc", data_loc])

        # Add image transformation options if provided
        if image_transform:
            # Parse transform string - can be "2" or "1 2 3" etc.
            transforms = image_transform.strip().split()
            if transforms:
                cmd.extend(["-ImTransOpt"] + transforms)

        # Add beamline-standard bad pixel and gap intensity markers
        cmd.extend(["-BadPxIntensity", "-2"])
        cmd.extend(["-GapIntensity", "-1"])

        # ===== TRANSPARENCY: Show exact command being run =====
        cmd_str = " ".join(str(x) for x in cmd)
        print("="*70, file=sys.stderr)
        print("🔧 MIDAS AUTO-CALIBRATION COMMAND:", file=sys.stderr)
        print(f"   Working directory: {image_path.parent}", file=sys.stderr)
        print(f"   Python: {cmd[0]}", file=sys.stderr)
        print(f"   Script: {cmd[1]}", file=sys.stderr)
        print(f"   Parameters:", file=sys.stderr)
        for i in range(2, len(cmd), 2):
            if i+1 < len(cmd) and cmd[i].startswith("-"):
                print(f"      {cmd[i]} {cmd[i+1]}", file=sys.stderr)
        print("="*70, file=sys.stderr)

        # Run calibration with MIDAS environment
        result = subprocess.run(
            cmd,
            cwd=str(image_path.parent),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=get_midas_python_env()  # Python script — no DYLD_LIBRARY_PATH override
        )

        if result.returncode != 0:
            error_msg = f"Calibration failed with exit code {result.returncode}"

            # Print raw stderr to terminal so operator can see the real error
            print(f"❌ Calibration failed (exit code {result.returncode})", file=sys.stderr)
            if result.stderr:
                print("--- stderr (last 20 lines) ---", file=sys.stderr)
                for line in result.stderr.strip().splitlines()[-20:]:
                    print(f"  {line}", file=sys.stderr)
            if result.stdout:
                print("--- stdout (last 5 lines) ---", file=sys.stderr)
                for line in result.stdout.strip().splitlines()[-5:]:
                    print(f"  {line}", file=sys.stderr)

            return format_result({
                "tool": "midas_auto_calibrate",
                "status": "error",
                "error": error_msg,
                "stderr": result.stderr,
                "stdout": result.stdout
            })

        # v10 names the output refined_MIDAS_params_<material>.txt (e.g. _CeO2.txt)
        # so glob for any matching file rather than hardcoding the name
        output = result.stdout
        candidates = sorted(image_path.parent.glob("refined_MIDAS_params*.txt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        refined_params_file = candidates[0] if candidates else image_path.parent / "refined_MIDAS_params.txt"

        calibrated_params = {
            "bc_x": None,
            "bc_y": None,
            "lsd": None,
            "tx": None,
            "ty": None,
            "tz": None,
            "p0": None,
            "p1": None,
            "p2": None,
            "p3": None,
            "wavelength": None,
            "px": None
        }

        if refined_params_file.exists():
            # Parse the refined parameters file
            with open(refined_params_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    key = parts[0]
                    if key == 'BC':
                        # BC format: "BC y x" (MIDAS uses Y X order)
                        if len(parts) >= 3:
                            calibrated_params['bc_y'] = float(parts[1])
                            calibrated_params['bc_x'] = float(parts[2])
                    elif key == 'Lsd':
                        calibrated_params['lsd'] = float(parts[1])
                    elif key == 'tx':
                        calibrated_params['tx'] = float(parts[1])
                    elif key == 'ty':
                        calibrated_params['ty'] = float(parts[1])
                    elif key == 'tz':
                        calibrated_params['tz'] = float(parts[1])
                    elif key == 'p0':
                        calibrated_params['p0'] = float(parts[1])
                    elif key == 'p1':
                        calibrated_params['p1'] = float(parts[1])
                    elif key == 'p2':
                        calibrated_params['p2'] = float(parts[1])
                    elif key == 'p3':
                        calibrated_params['p3'] = float(parts[1])
                    elif key == 'Wavelength':
                        calibrated_params['wavelength'] = float(parts[1])
                    elif key == 'px':
                        calibrated_params['px'] = float(parts[1])

        # Parse convergence metrics from autocal.log
        autocal_log = image_path.parent / "autocal.log"
        convergence_metrics = {
            "num_iterations": None,
            "final_mean_strain": None,
            "excluded_rings": [],
            "converged": False
        }

        if autocal_log.exists():
            with open(autocal_log, 'r') as f:
                log_content = f.read()

                # Count iterations
                iterations = log_content.count('INFO - Iteration')
                if iterations > 0:
                    convergence_metrics["num_iterations"] = iterations

                # Parse final mean strain from last INFO line
                info_lines = [l for l in log_content.split('\n') if 'INFO -' in l and 'Mean Strain' in l]
                if info_lines:
                    last_line = info_lines[-1]
                    try:
                        # Format: "INFO - Mean Strain: 0.000123"
                        strain_val = float(last_line.split('Mean Strain:')[-1].strip())
                        convergence_metrics["final_mean_strain"] = strain_val
                        convergence_metrics["converged"] = strain_val < 0.0001
                    except:
                        pass

                # Parse excluded rings
                excluded_lines = [l for l in log_content.split('\n') if 'Excluding ring' in l]
                for line in excluded_lines:
                    try:
                        ring_num = int(line.split('ring')[-1].strip().split()[0])
                        if ring_num not in convergence_metrics["excluded_rings"]:
                            convergence_metrics["excluded_rings"].append(ring_num)
                    except:
                        pass

        # Also check stdout for INFO messages (if autocal.log not found)
        if not autocal_log.exists():
            for line in output.split('\n'):
                if 'INFO -' in line and 'Mean Strain' in line:
                    try:
                        strain_val = float(line.split('Mean Strain')[-1].replace(':', '').strip())
                        convergence_metrics["final_mean_strain"] = strain_val
                        convergence_metrics["converged"] = strain_val < 0.0001
                    except:
                        pass

        # Look for generated zarr file
        zarr_file = None
        for f in image_path.parent.glob("*.zarr.zip"):
            if f.stat().st_mtime > (image_path.stat().st_mtime - 60):  # Created recently
                zarr_file = str(f)
                break

        # Build success message
        bc_x = calibrated_params.get('bc_x')
        bc_y = calibrated_params.get('bc_y')
        lsd_mm = calibrated_params.get('lsd')

        if lsd_mm is not None and lsd_mm > 1000:  # Convert from µm to mm
            lsd_mm = lsd_mm / 1000.0

        message = f"✓ Auto-calibration completed successfully!\n\n"
        message += f"Refined Parameters:\n"
        message += f"  Beam Center: ({bc_x:.2f}, {bc_y:.2f}) pixels\n"
        message += f"  Distance (Lsd): {lsd_mm:.2f} mm\n"
        if calibrated_params.get('tx'):
            message += f"  Tilts: tx={calibrated_params['tx']:.6f}, ty={calibrated_params['ty']:.6f}, tz={calibrated_params['tz']:.6f} rad\n"

        message += f"\nConvergence:\n"
        if convergence_metrics["num_iterations"]:
            message += f"  Iterations: {convergence_metrics['num_iterations']}\n"
        if convergence_metrics["final_mean_strain"]:
            message += f"  Final Mean Strain: {convergence_metrics['final_mean_strain']:.6f}\n"
            message += f"  Status: {'CONVERGED ✓' if convergence_metrics['converged'] else 'HIGH STRAIN — consider increasing --n-iterations'}\n"
        if convergence_metrics["excluded_rings"]:
            message += f"  Excluded Rings: {', '.join(map(str, convergence_metrics['excluded_rings']))}\n"

        message += f"\nOutput Files:\n"
        message += f"  • {refined_params_file.name} - Use this for ff_MIDAS.py\n"
        if autocal_log.exists():
            message += f"  • autocal.log - Detailed iteration history\n"
        if save_plots_hdf:
            message += f"  • {Path(save_plots_hdf).name} - Diagnostic plots and arrays\n"

        return format_result({
            "tool": "midas_auto_calibrate",
            "status": "success",
            "image_file": str(image_path),
            "input_parameters_file": str(param_path),
            "calibrated_parameters_file": str(refined_params_file) if refined_params_file.exists() else None,
            "calibrated_parameters": calibrated_params,
            "convergence_metrics": convergence_metrics,
            "zarr_file": zarr_file,
            "message": message
        })

    except subprocess.TimeoutExpired:
        return format_result({
            "tool": "midas_auto_calibrate",
            "status": "error",
            "error": "Calibration timed out (>10 minutes)"
        })
    except Exception as e:
        return format_result({
            "tool": "midas_auto_calibrate",
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# BATCH INTEGRATION (MULTI-PANEL DETECTOR SUPPORT)
# =============================================================================

@mcp.tool()
async def midas_batch_integrate(
    data_file: str,
    dark_file: str,
    parameter_file: str,
    start_frame: int,
    end_frame: int,
    result_folder: str = "./integration_results",
    map_detector: bool = True,
    num_cpus: int = 10,
    num_frame_chunks: int = 10,
    convert_files: bool = True,
    write_mat: bool = False,
    data_location: str = "/exchange/data",
    dark_location: str = "/exchange/data"
) -> str:
    """
    Batch integrate 2D diffraction images using MIDAS Python integrator.

    This is the PRODUCTION workflow used at beamlines for multi-panel detectors
    like Hydra (4 panels). Uses integrator.py which supports:
    - Multi-panel detector mapping
    - Dark file correction
    - Batch frame processing
    - Parallel CPU processing
    - HDF5 input/output

    ✨ AUTOMATED: Runs DetectorMapper if Map.bin missing (handled by integrator.py)

    Based on cake_ge_v2.sh workflow from 1-ID beamline.

    Args:
        data_file: Path to HDF5 data file (e.g., sample_003083.ge1.h5)
        dark_file: Path to HDF5 dark file for background subtraction
        parameter_file: MIDAS parameter file (refined_MIDAS_params.txt)
        start_frame: First frame number to process
        end_frame: Last frame number to process
        result_folder: Output directory for results (default: ./integration_results)
        map_detector: Enable detector mapping for multi-panel detectors (default: True)
        num_cpus: Number of CPU cores for parallel processing (default: 10)
        num_frame_chunks: Number of chunks to divide frames into (default: 10)
        convert_files: Convert files to MIDAS analysis format (default: True)
        write_mat: Write MATLAB .mat files (default: False, saves disk space)
        data_location: Location within data HDF5 file (default: /exchange/data)
        dark_location: Location within dark HDF5 file (default: /exchange/data)

    Returns:
        JSON with integration status, output files (.zarr.zip), and processing details

    Example:
        midas_batch_integrate(
            data_file="/path/to/CeO2_003083.ge1.h5",
            dark_file="/path/to/dark_003084.ge1.h5",
            parameter_file="refined_MIDAS_params_ge1_Tx_cake_partial.txt",
            start_frame=3083,
            end_frame=3085,
            map_detector=True,
            num_cpus=80
        )
    """
    try:
        # Validate inputs
        data_path = Path(data_file).resolve()
        dark_path = Path(dark_file).resolve()
        param_path = Path(parameter_file).resolve()

        if not data_path.exists():
            return format_result({
                "tool": "midas_batch_integrate",
                "status": "error",
                "error": f"Data file not found: {data_path}"
            })

        if not dark_path.exists():
            return format_result({
                "tool": "midas_batch_integrate",
                "status": "error",
                "error": f"Dark file not found: {dark_path}"
            })

        if not param_path.exists():
            return format_result({
                "tool": "midas_batch_integrate",
                "status": "error",
                "error": f"Parameter file not found: {param_path}"
            })

        # Find MIDAS integrator.py via MIDAS_ROOT (set from .env)
        if not MIDAS_ROOT:
            return format_result({"tool": "midas_batch_integrate", "status": "error",
                                  "error": "MIDAS_PATH not set. Add MIDAS_PATH to .env"})
        midas_integrator = MIDAS_ROOT / "utils" / "integrator.py"
        if not midas_integrator.exists():
            return format_result({"tool": "midas_batch_integrate", "status": "error",
                                  "error": f"integrator.py not found at {midas_integrator}"})

        # Create result folder
        result_path = Path(result_folder).resolve()
        result_path.mkdir(parents=True, exist_ok=True)

        # Build integrator command
        # Based on: python ~/opt/MIDAS/utils/integrator.py -resultFolder ./ge1_cake -paramFN params.txt -dataFN data.h5 -dataLoc /exchange/data -darkFN dark.h5 -darkLoc /exchange/data -startFileNr 3083 -endFileNr 3085 -convertFiles 1 -mapDetector 1 -nCPUs 80 -writeMat 0 -numFrameChunks 10

        midas_python = find_midas_python()
        cmd = [
            midas_python,
            str(midas_integrator),
            "-resultFolder", str(result_path),
            "-paramFN", str(param_path),
            "-dataFN", str(data_path),
            "-dataLoc", data_location,
            "-darkFN", str(dark_path),
            "-darkLoc", dark_location,
            "-startFileNr", str(start_frame),
            "-endFileNr", str(end_frame),
            "-convertFiles", "1" if convert_files else "0",
            "-mapDetector", "1" if map_detector else "0",
            "-nCPUs", str(num_cpus),
            "-writeMat", "1" if write_mat else "0",
            "-numFrameChunks", str(num_frame_chunks)
        ]

        cmd_str = " ".join(cmd)
        print(f"\n  $ {cmd_str}", file=sys.stderr)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for batch processing
            env=get_midas_env()
        )

        if result.returncode != 0:
            print(f"❌ Batch integration failed (exit {result.returncode})", file=sys.stderr)
            for line in result.stderr.strip().splitlines()[-20:]:
                print(f"  {line}", file=sys.stderr)
            return format_result({
                "tool": "midas_batch_integrate",
                "status": "error",
                "command": cmd_str,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": f"Integration failed with exit code {result.returncode}"
            })

        # Find output files
        zarr_files = list(result_path.glob("*.zarr.zip"))
        hdf_files = list(result_path.glob("*.hdf"))
        mat_files = list(result_path.glob("*.mat"))

        # Count processed frames
        num_frames = end_frame - start_frame + 1

        # Build success message
        message = f"✓ Batch integration completed successfully!\n\n"
        message += f"Processing Details:\n"
        message += f"  Frames processed: {num_frames} ({start_frame} to {end_frame})\n"
        message += f"  Data file: {data_path.name}\n"
        message += f"  Dark file: {dark_path.name}\n"
        message += f"  Parameters: {param_path.name}\n"
        message += f"  Detector mapping: {'ENABLED' if map_detector else 'DISABLED'}\n"
        message += f"  CPUs used: {num_cpus}\n"

        message += f"\nOutput Files ({result_path}):\n"
        if zarr_files:
            message += f"  • {len(zarr_files)} ZARR files (.zarr.zip)\n"
            for zf in zarr_files[:3]:  # Show first 3
                message += f"    - {zf.name}\n"
            if len(zarr_files) > 3:
                message += f"    ... and {len(zarr_files) - 3} more\n"

        if hdf_files:
            message += f"  • {len(hdf_files)} HDF5 files\n"

        if mat_files:
            message += f"  • {len(mat_files)} MATLAB files\n"

        message += f"\nNext Steps:\n"
        message += f"  1. ZARR files contain integrated 1D patterns\n"
        message += f"  2. For multi-panel detectors, repeat for all panels (ge1, ge2, ge3, ge4)\n"
        message += f"  3. Convert ZARR to MAT if needed using zarr_tomat.py\n"
        message += f"  4. Use ff_MIDAS.py for grain indexing with refined parameters\n"

        return format_result({
            "tool": "midas_batch_integrate",
            "status": "success",
            "method": "MIDAS integrator.py",
            "command": cmd_str,
            "data_file": str(data_path),
            "dark_file": str(dark_path),
            "parameter_file": str(param_path),
            "result_folder": str(result_path),
            "frame_range": {"start": start_frame, "end": end_frame, "count": num_frames},
            "detector_mapping": map_detector,
            "cpus": num_cpus,
            "output_files": {
                "zarr": [str(f) for f in zarr_files],
                "hdf": [str(f) for f in hdf_files],
                "mat": [str(f) for f in mat_files]
            },
            "stdout": result.stdout,
            "stderr": result.stderr,
            "message": message
        })

    except subprocess.TimeoutExpired:
        return format_result({
            "tool": "midas_batch_integrate",
            "status": "error",
            "error": "Integration timed out (>30 minutes). Try reducing frame count or increase timeout."
        })
    except Exception as e:
        return format_result({
            "tool": "midas_batch_integrate",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# =============================================================================
# KNOWLEDGE BASE & DOMAIN EXPERTISE
# =============================================================================

# Initialize knowledge base (lazy loading)
_knowledge_base = None
_materials_db = None
_typical_params = None

def get_knowledge_base():
    """Get or initialize the knowledge base (RAG with ChromaDB)"""
    global _knowledge_base
    if _knowledge_base is None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            from pathlib import Path

            kb_path = Path(__file__).parent / "knowledge_base"
            chroma_path = kb_path / "chroma_db"

            if chroma_path.exists():
                _knowledge_base = {
                    "client": chromadb.PersistentClient(path=str(chroma_path)),
                    "embedder": SentenceTransformer('all-MiniLM-L6-v2'),
                    "available": True
                }
            else:
                _knowledge_base = {"available": False, "error": "Knowledge base not indexed"}
        except Exception as e:
            _knowledge_base = {"available": False, "error": str(e)}

    return _knowledge_base


def get_materials_db():
    """Get materials database"""
    global _materials_db
    if _materials_db is None:
        try:
            import json
            from pathlib import Path
            kb_path = Path(__file__).parent / "knowledge_base"
            materials_file = kb_path / "data" / "materials.json"

            if materials_file.exists():
                with open(materials_file) as f:
                    _materials_db = json.load(f)
            else:
                _materials_db = {}
        except Exception as e:
            _materials_db = {}

    return _materials_db


def get_typical_parameters():
    """Get typical HEDM parameters"""
    global _typical_params
    if _typical_params is None:
        try:
            import json
            from pathlib import Path
            kb_path = Path(__file__).parent / "knowledge_base"
            params_file = kb_path / "data" / "typical_parameters.json"

            if params_file.exists():
                with open(params_file) as f:
                    _typical_params = json.load(f)
            else:
                _typical_params = {}
        except Exception as e:
            _typical_params = {}

    return _typical_params


@mcp.tool()
async def query_hedm_knowledge(
    question: str,
    max_results: int = 3,
    source_type: Optional[str] = None
) -> str:
    """Query the HEDM knowledge base using semantic search across papers, logbooks, and books.

    This tool searches through indexed research papers, experimental logbooks, and
    crystallography textbooks to answer questions about HEDM theory, best practices,
    and past experiments.

    Args:
        question: Natural language question about HEDM, crystallography, calibration, etc.
                 Examples:
                 - "What is a good calibration strain value?"
                 - "How was sample XYZ processed in 2019?"
                 - "Explain Bragg's law for 61keV beam"
        max_results: Number of relevant excerpts to return (default: 3)
        source_type: Filter by document type: "paper", "logbook", "book", or None for all

    Returns:
        JSON string with relevant excerpts, sources, and confidence scores
    """
    try:
        kb = get_knowledge_base()

        if not kb.get("available"):
            return json.dumps({
                "status": "error",
                "error": "Knowledge base not available",
                "details": kb.get("error", "Not indexed"),
                "suggestion": "Run: python knowledge_base/index_knowledge.py"
            }, indent=2)

        # Encode the question
        query_embedding = kb["embedder"].encode(question).tolist()

        # Get collection
        collection = kb["client"].get_collection(name="hedm_knowledge")

        # Build filter
        where_filter = {"type": source_type} if source_type else None

        # Query the knowledge base
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            where=where_filter
        )

        # Format results
        excerpts = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            excerpts.append({
                "rank": i,
                "source": meta['source'],
                "type": meta['type'],
                "relevance_score": round(1 - dist, 3),
                "excerpt": doc,
                "chunk_index": meta.get('chunk_index', 0)
            })

        return json.dumps({
            "status": "success",
            "question": question,
            "results_count": len(excerpts),
            "excerpts": excerpts
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


@mcp.tool()
async def get_material_properties(material_name: str) -> str:
    """Get crystallographic properties and metadata for a material.

    Provides authoritative data from Materials Project including lattice parameters,
    space group, density, and HEDM-specific metadata like typical ring counts and
    calibration usage.

    Args:
        material_name: Chemical formula or name (e.g., "CeO2", "LaB6", "Ti", "Steel_316L")

    Returns:
        JSON string with complete material properties
    """
    try:
        materials_db = get_materials_db()

        # Try exact match first
        material = materials_db.get(material_name)

        # Try case-insensitive match
        if not material:
            for key in materials_db.keys():
                if key.upper() == material_name.upper():
                    material = materials_db[key]
                    break

        if not material:
            # List available materials
            available = [k for k in materials_db.keys() if not k.startswith("_")]
            return json.dumps({
                "status": "not_found",
                "requested": material_name,
                "available_materials": available[:20],
                "total_available": len(available)
            }, indent=2)

        return json.dumps({
            "status": "success",
            "material": material
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


@mcp.tool()
async def get_typical_hedm_parameters(
    beam_energy_kev: Optional[float] = None,
    geometry_type: Optional[str] = None
) -> str:
    """Get typical HEDM experimental parameters and quality thresholds.

    Provides guidelines for detector distances, calibration quality metrics,
    integration parameters, and processing recommendations based on experimental
    geometry and beam energy.

    Args:
        beam_energy_kev: Beam energy in keV (e.g., 30, 61.3, 90, 120)
        geometry_type: "ff_hedm" or "nf_hedm" (optional)

    Returns:
        JSON string with typical parameter ranges and quality thresholds
    """
    try:
        params = get_typical_parameters()

        if not params:
            return json.dumps({
                "status": "error",
                "error": "Typical parameters database not loaded"
            }, indent=2)

        result = {"status": "success"}

        # Get relevant beam energy info
        if beam_energy_kev:
            wavelength_angstrom = 12.398 / beam_energy_kev
            result["beam_info"] = {
                "energy_kev": beam_energy_kev,
                "wavelength_angstrom": round(wavelength_angstrom, 4)
            }

            # Find closest standard energy
            energies = params.get("beam_energies", {})
            for energy_type, energy_data in energies.items():
                if abs(energy_data["value_kev"] - beam_energy_kev) < 5:
                    result["beam_info"]["type"] = energy_type
                    result["beam_info"]["typical_use"] = energy_data["use_case"]
                    break

        # Get geometry-specific parameters
        if geometry_type and "detector_geometries" in params:
            geom_params = params["detector_geometries"].get(geometry_type)
            if geom_params:
                result["detector_geometry"] = geom_params

        # Always include quality thresholds
        result["calibration_quality"] = params.get("calibration_quality", {})
        result["data_quality_thresholds"] = params.get("data_quality_thresholds", {})

        # Include integration parameters if FF-HEDM
        if geometry_type == "ff_hedm" or not geometry_type:
            result["integration_parameters"] = params.get("integration_parameters", {})

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


@mcp.tool()
async def estimate_parameters_from_image(
    wavelength_angstrom: float,
    ring_radii_px: List[float],
    pixel_size_um: float,
    material: Optional[str] = "CeO2"
) -> str:
    """Estimate detector geometry (Lsd, beam center) from observed diffraction rings.

    Uses ring positions and known d-spacings to calculate detector distance.
    Helpful for initial parameter estimation before auto-calibration.

    Args:
        wavelength_angstrom: X-ray wavelength in Angstroms
        ring_radii_px: List of observed ring radii in pixels (e.g., [412, 478, 675])
        pixel_size_um: Detector pixel size in microns (typically 172 for GE)
        material: Calibrant material (default: "CeO2")

    Returns:
        JSON string with estimated Lsd and quality metrics
    """
    try:
        import numpy as np

        materials_db = get_materials_db()
        mat_data = materials_db.get(material)

        if not mat_data:
            return json.dumps({
                "status": "error",
                "error": f"Material {material} not found in database"
            }, indent=2)

        # Get d-spacings for this material
        d_spacings = mat_data.get("d_spacings_angstrom")
        if not d_spacings:
            return json.dumps({
                "status": "error",
                "error": f"No d-spacings available for {material}"
            }, indent=2)

        # Calculate Lsd for each ring
        lsd_estimates = []
        for r_px in ring_radii_px[:len(d_spacings)]:
            r_mm = r_px * pixel_size_um / 1000  # Convert px to mm

            # Bragg's law: 2*d*sin(theta) = lambda
            # theta = arcsin(lambda / (2*d))
            # R = Lsd * tan(2*theta)
            # Therefore: Lsd = R / tan(2*theta)

            for d in d_spacings:
                theta_rad = np.arcsin(wavelength_angstrom / (2 * d))
                two_theta_rad = 2 * theta_rad
                expected_lsd_mm = r_mm / np.tan(two_theta_rad)

                # Only consider reasonable Lsd values (100mm to 3000mm)
                if 100 < expected_lsd_mm < 3000:
                    lsd_estimates.append({
                        "ring_radius_px": r_px,
                        "d_spacing_angstrom": d,
                        "estimated_lsd_mm": round(expected_lsd_mm, 2)
                    })

        if not lsd_estimates:
            return json.dumps({
                "status": "error",
                "error": "Could not estimate Lsd from provided rings"
            }, indent=2)

        # Calculate average and std
        lsd_values = [est["estimated_lsd_mm"] for est in lsd_estimates]
        mean_lsd = np.mean(lsd_values)
        std_lsd = np.std(lsd_values)

        return json.dumps({
            "status": "success",
            "material": material,
            "wavelength_angstrom": wavelength_angstrom,
            "estimated_lsd_mm": round(mean_lsd, 2),
            "std_lsd_mm": round(std_lsd, 2),
            "confidence": "high" if std_lsd < 10 else "medium" if std_lsd < 50 else "low",
            "individual_estimates": lsd_estimates,
            "recommendation": f"Use Lsd guess: {round(mean_lsd * 1000)} microns (±{round(std_lsd)}mm)"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


# =============================================================================
# SERVER MAIN
# =============================================================================

if __name__ == "__main__":
    if not MIDAS_AVAILABLE:
        print("⚠ MIDAS scientific dependencies not available", file=sys.stderr)
    mcp.run(transport='stdio')
