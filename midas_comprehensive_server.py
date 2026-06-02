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
import xrayutilities as xu
import re
import subprocess
import asyncio
import logging
import traceback
from mcp.server.fastmcp import FastMCP

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

def _find_gsasii_conda_base() -> Optional[str]:
    """Find conda base directory across common locations."""
    conda_base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX")
    if not conda_base:
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            conda_base = str(Path(conda_exe).parent.parent)
    if not conda_base:
        for loc in [Path.home() / "miniconda3", Path.home() / "anaconda3",
                     Path.home() / "opt" / "miniconda3"]:
            if loc.exists():
                conda_base = str(loc)
                break
    return conda_base


def _find_gsasii_path() -> Optional[str]:
    """Auto-detect GSAS-II installation for subprocess env.

    Returns the parent directory of the GSASII package (the path to add to
    sys.path so that ``from GSASII import GSASIIscriptable`` works).

    Search order:
      1. GSASII_PATH environment variable
      2. Conda env named 'GSASII' with a GSAS-II checkout
      3. pip-installed GSASII in the current env
    """
    # 1. Explicit env var
    gsas_path = os.environ.get("GSASII_PATH")
    if gsas_path:
        p = Path(gsas_path)
        if (p / "GSASIIscriptable.py").exists():
            return str(p.parent)
        if (p / "GSASII" / "GSASIIscriptable.py").exists():
            return str(p)
        return str(p)

    # 2. Conda env named GSASII
    conda_base = _find_gsasii_conda_base()
    if conda_base:
        for sub in ["GSAS-II", "GSASII", "gsas2"]:
            candidate = Path(conda_base) / "envs" / "GSASII" / sub
            if (candidate / "GSASII" / "GSASIIscriptable.py").exists():
                return str(candidate)

    return None


def find_gsasii_python() -> Optional[str]:
    """Find Python interpreter from the GSASII conda env.

    GSAS-II binaries (pyspg, pypowder) are compiled against a specific
    Python version. The midas_env Python may be a different version, causing
    binary load failures. This function returns the GSASII env's own Python
    which is guaranteed to match the compiled binaries.

    Falls back to midas_env Python if no GSASII conda env exists (e.g.,
    pip-installed GSAS-II).
    """
    conda_base = _find_gsasii_conda_base()
    if conda_base:
        gsasii_python = Path(conda_base) / "envs" / "GSASII" / "bin" / "python"
        if gsasii_python.exists():
            return str(gsasii_python)
    return None


def get_midas_env() -> dict:
    """Get environment variables for all MIDAS operations (C++ and Python).

    MIDAS C++ binaries have @rpath set at build time so they find their own
    dylibs without DYLD_LIBRARY_PATH. Overriding DYLD_LIBRARY_PATH breaks
    h5py in midas_env (HDF5 symbol mismatch). Use this single function for
    all tools — both C++ executables and Python workflow scripts.
    """
    env = os.environ.copy()

    # Add MIDAS bin to PATH so C++ executables are found
    midas_bin_paths = [str(MIDAS_BIN), str(MIDAS_ROOT / "bin")]
    env["PATH"] = ":".join(midas_bin_paths + [env.get("PATH", "")])

    env["MIDAS_PATH"] = str(MIDAS_ROOT)

    # MIDAS/utils must be on PYTHONPATH for Python workflow scripts
    midas_utils = str(MIDAS_ROOT / "utils")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = midas_utils + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = midas_utils

    # Auto-detect GSAS-II so gsas_ii_refine.py can import GSASIIscriptable
    gsasii_path = _find_gsasii_path()
    if gsasii_path:
        env["GSASII_PATH"] = gsasii_path
        # Also add to PYTHONPATH so `from GSASII import ...` works directly
        env["PYTHONPATH"] = gsasii_path + ":" + env["PYTHONPATH"]

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
STRESS_RUNNER_SCRIPT = Path(__file__).parent / "_stress_runner.py"

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
    from scipy import ndimage
    from scipy.signal import find_peaks, peak_widths
    from scipy.optimize import curve_fit
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    MIDAS_AVAILABLE = True
except ImportError as e:
    MIDAS_AVAILABLE = False
    print(f"⚠ Scientific deps missing: {e}", file=sys.stderr)

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
            env=get_midas_env()
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
            "numpy", "scipy", "matplotlib", "fabio",
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

        # Check MIDAS Python packages (midas-params and midas-stress)
        validation["packages"] = {}
        midas_python = find_midas_python()
        for pkg_name, import_name in [
            ("midas_params", "midas_params"),
            ("midas_stress", "midas_stress"),
        ]:
            try:
                check = subprocess.run(
                    [midas_python, "-c", f"import {import_name}; print({import_name}.__version__)"],
                    capture_output=True, text=True, timeout=15,
                    env=get_midas_env(),
                )
                if check.returncode == 0:
                    validation["packages"][pkg_name] = {
                        "installed": True,
                        "version": check.stdout.strip(),
                    }
                else:
                    validation["packages"][pkg_name] = {"installed": False}
            except Exception:
                validation["packages"][pkg_name] = {"installed": False}

        pkgs_missing = [p for p, v in validation["packages"].items() if not v.get("installed")]
        if pkgs_missing:
            for pkg in pkgs_missing:
                validation["recommendations"].append(
                    f"Install {pkg}: pip install -e $MIDAS_PATH/packages/{pkg}"
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

def _read_param_value(param_path: Path, key: str) -> str | None:
    """Return the first whitespace-separated value for ``key`` in a MIDAS params file, or None."""
    try:
        for line in param_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split(None, 1)
            if len(parts) == 2 and parts[0] == key:
                return parts[1].split('#', 1)[0].strip()
    except Exception:
        return None
    return None


def _resolve_image_transform(image_path: Path, user_arg: str,
                             params_file: Path | None) -> tuple[str, str]:
    """Pick the right ImTransOpt for calibration.

    Per MIDAS manual (manuals/README.md §Image Transformation), ImTransOpt is
    detector-mount specific — there is no extension-based rule. Resolution order:

      1. Explicit ``user_arg`` wins (honour what the operator said).
      2. ``ImTransOpt`` from the parameter file if present (re-use prior calibration).
      3. Sibling ``parameters.txt`` / ``Parameters.txt`` next to the image (auto-pick).
      4. Fall back to ``"0"`` (no transform) and emit a warning.

    Returns ``(transform_str, source)``. ``transform_str`` is "" when no override
    should be passed (i.e. params already has it). ``source`` is a short tag for
    diagnostics ("user" / "params" / "sibling" / "default-warned").
    """
    if user_arg:
        return user_arg.strip(), "user"

    # Check the explicit params file we're about to use
    if params_file and params_file.exists():
        v = _read_param_value(params_file, "ImTransOpt")
        if v is not None:
            print(f"  ImTransOpt={v} (from params {params_file.name})", file=sys.stderr)
            return "", "params"  # already in params, do not pass on CLI

    # Check sibling params files in the image directory
    for sibling_name in ("parameters.txt", "Parameters.txt", "params.txt"):
        sibling = image_path.parent / sibling_name
        if sibling.exists():
            v = _read_param_value(sibling, "ImTransOpt")
            if v is not None:
                print(f"  ImTransOpt={v} (auto-detected from {sibling.name})", file=sys.stderr)
                return v, "sibling"

    print("  ⚠ ImTransOpt not specified and not found in params/siblings — defaulting to 0 (no transform).", file=sys.stderr)
    print("    If calibration fails or geometry looks wrong, verify ImTransOpt against a", file=sys.stderr)
    print("    physical fiducial (beam-stop wire, fiducial marker). See MIDAS manuals/README.md.", file=sys.stderr)
    return "0", "default-warned"


def _strip_empty_value_lines(param_path: Path):
    """Remove lines with a key but no value (e.g. 'Dark \\n') that crash ffGenerateZipRefactor.py."""
    try:
        lines = param_path.read_text().splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                cleaned.append(line)
                continue
            parts = stripped.split(None, 1)
            if len(parts) == 1 and not stripped[0].isdigit():
                continue
            cleaned.append(line)
        param_path.write_text('\n'.join(cleaned) + '\n')
    except Exception:
        pass


@mcp.tool()
async def midas_integrate_2d_to_1d(
    image_file: str,
    calibration_file: str = None,
    dark_file: str = None,
    result_folder: str = None,
    n_cpus: int = 4,
    convert_files: bool = True,
    bright_file: str = None,
    csv_output: bool = False,
    out_name: str = None,
    short_names: bool = True,
    r_min: float = None,
    r_max: float = None,
    r_bin_size: float = None,
    eta_min: float = None,
    eta_max: float = None,
    eta_bin_size: float = None,
) -> str:
    """Integrate a single 2D diffraction image to a 1D lineout using MIDAS integrator.py (v11).

    calibration_file is optional — if omitted, auto-searches the image directory for
    refined_MIDAS_params*.txt produced by midas_auto_calibrate.

    v11 executable chain:
      integrator.py (FF_HEDM/workflows/) → IntegratorZarrOMP

    Outputs written to result_folder (default: image_dir/integration/):
      *_lineout.xy   — 2θ (degrees) vs intensity text file
      *_lineout.bin  — binary lineout
      <stem>.zarr.zip — caked output (GSAS-II compatible; short-name default)
      *_lineouts.csv / *_REtaMap.csv — when csv_output=True
      Map.bin / nMap.bin / maskMap.bin — geometry maps (generated once)

    Args:
        image_file: Path to 2D diffraction image (.tif/.tiff, .ge/.ge1-.ge5, .h5/.hdf5, .zip)
        calibration_file: MIDAS parameter file (refined_MIDAS_params*.txt). Auto-detected if omitted.
        dark_file: Optional dark field image for background subtraction
        result_folder: Output directory (default: <image_dir>/integration)
        n_cpus: OMP threads for IntegratorZarrOMP (default: 4)
        convert_files: Convert input to Zarr before integrating (default: True)
        bright_file: Optional bright/flat-field image; integrated profiles embedded under processed/bright/ in zarr (v11, issue #20)
        csv_output: Also export per-frame lineouts and REtaMap as CSVs (v11, issue #23)
        out_name: Override the output zarr stem (single-file only — clobbers on multi-file)
        short_names: Use short v11 output naming (<stem>.zarr.zip). False = legacy suffix-stacking
        r_min, r_max, r_bin_size: Radial integration range overrides (pixels). Defaults: from params file.
        eta_min, eta_max, eta_bin_size: Azimuthal range overrides (degrees). Defaults: from params file.

    The agent SHOULD prompt the user to confirm R/eta ranges and result_folder before
    invoking this tool — those values control what 2θ window and azimuthal slice get
    integrated, which depends on what the user wants to study (specific rings, full eta,
    sub-sector, etc.). See midas-integrate SKILL.md for the recommended prompting flow.
    """
    try:
        image_path = Path(image_file).expanduser().absolute()

        if not image_path.exists():
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"Image not found: {image_path}"})

        # Auto-find calibration file if not provided
        if calibration_file is None:
            candidates = sorted(image_path.parent.glob("refined_MIDAS_params*.txt"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                      "error": "No refined_MIDAS_params*.txt found in image directory. "
                                               "Run midas_auto_calibrate first, or pass calibration_file explicitly."})
            param_path = candidates[0]
            print(f"  Auto-detected calibration file: {param_path.name}", file=sys.stderr)
        else:
            param_path = Path(calibration_file).expanduser().absolute()
            if not param_path.exists():
                return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                      "error": f"Parameter file not found: {param_path}"})

        # Validate param file is actually a MIDAS parameter file (not CSV/JSON/etc.)
        if param_path.suffix.lower() in ('.csv', '.json', '.log', '.bin', '.h5', '.hdf', '.tif'):
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"Invalid parameter file: {param_path.name} (suffix {param_path.suffix}). "
                                           f"Use refined_MIDAS_params*.txt from midas_auto_calibrate."})
        try:
            with open(param_path) as f:
                first_500 = f.read(500)
            if not any(kw in first_500 for kw in ["Lsd", "Wavelength", "NrPixels", "BC "]):
                return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                      "error": f"Parameter file {param_path.name} is missing critical keys "
                                               f"(Lsd, Wavelength, NrPixels, BC). "
                                               f"Use refined_MIDAS_params*.txt from midas_auto_calibrate."})
        except UnicodeDecodeError:
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"Parameter file {param_path.name} is a binary file, not a text param file."})

        # Strip empty-value lines that crash ffGenerateZipRefactor.py
        _strip_empty_value_lines(param_path)

        # ── Native engine attempt (Strategy C: PyTorch compute + dual-format) ──
        # OPT-IN ONLY: subprocess (integrator.py) is the default path; set
        # APEXA_USE_NATIVE_MIDAS=1 to try the native pip-package path first.
        # Native path is also skipped when bright_file is set (no flat-field
        # support yet) or csv_output is requested (no export-csv sidecars yet).
        if (os.environ.get("APEXA_USE_NATIVE_MIDAS") == "1"
                and not bright_file and not csv_output):
            try:
                from apexa_midas_native import (
                    native_integrate_2d_to_1d, MidasEngineUnavailable,
                )
                print("[engine] trying native midas_integrate first…",
                      file=sys.stderr)
                _out = (str(Path(result_folder).expanduser().absolute())
                        if result_folder
                        else str(image_path.parent / "integration"))
                result_dict = native_integrate_2d_to_1d(
                    image_file=str(image_path),
                    calibration_file=str(param_path),
                    dark_file=dark_file or "",
                    result_folder=_out,
                    out_name=out_name,
                    r_min=r_min, r_max=r_max, r_bin_size=r_bin_size,
                    eta_min=eta_min, eta_max=eta_max, eta_bin_size=eta_bin_size,
                )
                return format_result(result_dict)
            except MidasEngineUnavailable as e:
                print(f"[engine] native unavailable: {e.install_hint}",
                      file=sys.stderr)
                print("[engine] falling back to subprocess (integrator.py)",
                      file=sys.stderr)
            except Exception as e:
                print(f"[engine] native call raised {type(e).__name__}: {e}",
                      file=sys.stderr)
                print("[engine] falling back to subprocess (integrator.py)",
                      file=sys.stderr)

        # Find integrator.py — v10 location: FF_HEDM/workflows/
        integrator_script = MIDAS_ROOT / "FF_HEDM" / "workflows" / "integrator.py"
        if not integrator_script.exists():
            return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                  "error": f"integrator.py not found at {integrator_script}"})

        out_dir = Path(result_folder).expanduser().absolute() if result_folder \
                  else image_path.parent / "integration"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract file number from basename only (not full path) to avoid
        # matching numbers in parent directories (e.g. /Users/b324240/ → 324240)
        basename = image_path.stem
        file_nr_match = re.search(r'(\d{6})', basename)
        file_nr = int(file_nr_match.group(1)) if file_nr_match else None

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
            "-shortNames",   "1" if short_names else "0",
            "-csvOutput",    "1" if csv_output else "0",
        ]
        if file_nr is not None:
            cmd += ["-startFileNr", str(file_nr), "-endFileNr", str(file_nr)]
        if dark_file:
            dark_path = Path(dark_file).expanduser().absolute()
            if not dark_path.exists():
                return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                      "error": f"Dark file not found: {dark_path}"})
            cmd += ["-darkFN", str(dark_path)]
        if bright_file:
            bright_path = Path(bright_file).expanduser().absolute()
            if not bright_path.exists():
                return format_result({"tool": "midas_integrate_2d_to_1d", "status": "error",
                                      "error": f"Bright file not found: {bright_path}"})
            cmd += ["-brightFN", str(bright_path)]
        if out_name:
            cmd += ["-outName", str(out_name)]

        # Trailing parameter overrides — integrator.py treats unknown KEY VALUE pairs
        # at end of argv as parameter-file overrides written to a temp params copy.
        for key, val in (("RMin", r_min), ("RMax", r_max), ("RBinSize", r_bin_size),
                         ("EtaMin", eta_min), ("EtaMax", eta_max),
                         ("EtaBinSize", eta_bin_size)):
            if val is not None:
                cmd += [key, str(val)]

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
        # v11 short-name default is <stem>.zarr.zip; legacy is *_caked.hdf.zarr.zip
        zarr_out = sorted(out_dir.glob("*.zarr.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        csv_files = sorted(out_dir.glob("*_lineouts.csv"), key=lambda p: p.stat().st_mtime, reverse=True) \
                    if csv_output else []

        return format_result({
            "tool": "midas_integrate_2d_to_1d", "status": "success",
            "input_image": str(image_path),
            "calibration_file": str(param_path),
            "result_folder": str(out_dir),
            "lineout_xy": str(lineout[0]) if lineout else "not found — check result_folder",
            "zarr_zip": str(zarr_out[0]) if zarr_out else "not found",
            "csv_files": [str(p) for p in csv_files] if csv_files else None,
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
    parameters_file: str = "",
    dark_file: str = "",
    output_dir: str = "",
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
        parameters_file: OPTIONAL path to MIDAS parameter file. If omitted, AutoCalibrateZarr.py auto-detects
                        calibrant (CeO2/LaB6 from filename), energy (from keV in filename), distance (from mm in filename),
                        and pixel size (from detector shape). Only needed if filename lacks these hints.
        dark_file: Optional path to dark field image for background subtraction
        output_dir: Optional output directory for calibration results (refined_MIDAS_params*.txt,
                   autocal.log, calibrant_screen_out.csv). Created if it does not exist.
                   Default: "" = write results alongside the source image file (original behaviour).
                   Use this when calibrating multiple files in a batch so results from each file
                   land in their own directory instead of clobbering each other.
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

        # ── Native engine attempt ──────────────────────────────────────────
        # OPT-IN ONLY: subprocess (AutoCalibrateZarr.py) is the default path;
        # set APEXA_USE_NATIVE_MIDAS=1 to try the native pip-package path
        # first. Native path is also skipped when parameters_file is empty
        # (no filename-based auto-detect yet) or when image_transform is set.
        if (os.environ.get("APEXA_USE_NATIVE_MIDAS") == "1"
                and parameters_file and not image_transform):
            try:
                from apexa_midas_native import (
                    native_autocalibrate, MidasEngineUnavailable,
                )
                print("[engine] trying native midas_calibrate first…",
                      file=sys.stderr)
                result_dict = native_autocalibrate(
                    image_file=image_file,
                    parameters_file=parameters_file,
                    dark_file=dark_file,
                    n_iterations=n_iterations,
                )
                return format_result(result_dict)
            except MidasEngineUnavailable as e:
                print(f"[engine] native unavailable: {e.install_hint}",
                      file=sys.stderr)
                print("[engine] falling back to subprocess (AutoCalibrateZarr.py)",
                      file=sys.stderr)
            except Exception as e:
                # Any other native failure → fall back; don't lose the user.
                print(f"[engine] native call raised {type(e).__name__}: {e}",
                      file=sys.stderr)
                print("[engine] falling back to subprocess (AutoCalibrateZarr.py)",
                      file=sys.stderr)

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
        print(f"✓ Image path: {image_path} (exists: {image_path.exists()})", file=sys.stderr)

        param_path = None
        if parameters_file:
            param_path = Path(parameters_file).expanduser().absolute()
            print(f"✓ Param path: {param_path} (exists: {param_path.exists()})", file=sys.stderr)
        else:
            print(f"  No parameter file provided — AutoCalibrateZarr.py will auto-detect from filename", file=sys.stderr)

        # Auto-search for image file if not found
        if not image_path.exists():
            print(f"⚠ Image file not found at: {image_path}", file=sys.stderr)

            # Search ONLY in the directory the user specified, not all CWD subdirs.
            # This prevents picking wrong files from test2/ when user meant test1/.
            specified_dir = image_path.parent
            search_dirs = [specified_dir]
            # Only expand to CWD if specified_dir doesn't exist (user gave bare filename)
            if not specified_dir.exists() or specified_dir == Path.cwd():
                search_dirs = [Path.cwd()]

            print(f"  Searching in: {', '.join(str(d) for d in search_dirs)}", file=sys.stderr)

            search_patterns = ["*.tif", "*.tiff", "*.ge", "*.ge1", "*.ge2", "*.ge3", "*.ge4", "*.ge5", "*.h5", "*.hdf5", "*.nxs"]
            found_files = []
            for search_dir in search_dirs:
                if search_dir.exists():
                    for pattern in search_patterns:
                        found_files.extend(search_dir.glob(pattern))

            # Filter symlinks to avoid duplicates
            found_files = [f for f in found_files if not f.is_symlink()]

            if found_files:
                # Try exact stem match first, then partial match
                search_stem = image_path.stem.lower()
                exact_match = next((f for f in found_files if f.stem.lower() == search_stem), None)
                if exact_match:
                    image_path = exact_match
                    print(f"  ✓ Exact match: {image_path}", file=sys.stderr)
                else:
                    partial_match = next((f for f in found_files if search_stem in f.stem.lower()), None)
                    if partial_match:
                        image_path = partial_match
                        print(f"  ✓ Partial match: {image_path}", file=sys.stderr)
                    else:
                        # List what was found so the user can pick
                        file_list = "\n".join(f"  - {f.parent.name}/{f.name}" for f in found_files[:10])
                        return format_result({
                            "tool": "midas_auto_calibrate",
                            "status": "error",
                            "error": f"File '{image_path.name}' not found. Files in {specified_dir}:\n{file_list}\n\nPlease provide the exact filename."
                        })
            else:
                return format_result({
                    "tool": "midas_auto_calibrate",
                    "status": "error",
                    "error": f"No diffraction images found in {specified_dir}. Please provide the full absolute path to the image file."
                })

        # Auto-search for parameters file if specified but not found
        if param_path and not param_path.exists():
            print(f"⚠ Parameters file not found at: {param_path}", file=sys.stderr)

            # Search in image directory and cwd
            search_dirs = {param_path.parent, image_path.parent, Path.cwd()}
            param_patterns = ["*arameters*.txt", "*params*.txt", "*Params*.txt", "refined_MIDAS_params*.txt"]
            found_params = []
            for search_dir in search_dirs:
                if search_dir.exists():
                    for pattern in param_patterns:
                        found_params.extend(search_dir.glob(pattern))

            # Filter out obvious non-parameter files (< 10KB)
            found_params = [p for p in found_params if p.stat().st_size < 10000]

            if found_params:
                found_params.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                search_basename = param_path.stem.lower()
                basename_match = next(
                    (f for f in found_params if search_basename in f.stem.lower() or "param" in f.stem.lower()),
                    None
                )
                param_path = basename_match if basename_match else found_params[0]
                print(f"  ✓ Using: {param_path}", file=sys.stderr)
            else:
                # No param file found — let AutoCalibrateZarr.py auto-detect
                print(f"  No parameter file found — falling back to auto-detection from filename", file=sys.stderr)
                param_path = None

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

        # Save original filename before symlink creation (for energy extraction)
        original_filename = image_path.name
        if image_path.is_symlink():
            original_filename = image_path.resolve().name

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

            # Skip symlink if the friendly name is already the same as the input
            # (would create a self-referential symlink CeO2_000001.tif -> CeO2_000001.tif)
            if midas_friendly_name == image_path.name:
                print(f"  Input already has a MIDAS-friendly name: {image_path.name}", file=sys.stderr)
                # Resolve the real target if the input is itself a symlink
                if image_path.is_symlink():
                    real_target = image_path.resolve()
                    print(f"  Symlink target: {real_target.name}", file=sys.stderr)
                    # Rebuild the symlink to point to the real file
                    image_path.unlink()
                    image_path.symlink_to(real_target)
                    print(f"  ✓ Refreshed symlink: {image_path.name} -> {real_target}", file=sys.stderr)
            else:
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
                    midas_friendly_path.symlink_to(image_path.resolve())  # Absolute symlink
                    print(f"✓ Created MIDAS-friendly symlink: {midas_friendly_name} -> {image_path.resolve()}", file=sys.stderr)
                    image_path = midas_friendly_path
                except Exception as e:
                    print(f"⚠ Could not create symlink: {e}. Using original filename.", file=sys.stderr)
                    # Continue with original filename

        # Extract energy/wavelength from original filename if symlink lost it
        # Uses same regex as MIDAS AutoCalibrateZarr.py (handles 61p332keV, 61.332keV, 30keV)
        _energy_from_filename = None
        original_stem = Path(original_filename).stem
        energy_match = re.search(
            r'(?:^|[_\-])([\d]+(?:[p.][\d]+)?)keV(?:[_\-.]|$)',
            original_stem, re.IGNORECASE
        )
        if energy_match:
            energy_kev = float(energy_match.group(1).replace('p', '.'))
            if energy_kev > 0:
                from apexa_units import kev_to_angstrom
                _energy_from_filename = kev_to_angstrom(energy_kev)
                print(f"✓ Extracted energy from original filename: {energy_kev} keV → λ = {_energy_from_filename:.6f} Å", file=sys.stderr)

        # Extract Lsd guess from original filename if present (e.g. 650mm, 210mm)
        lsd_match = re.search(
            r'(?:^|[_\-])([\d]+(?:[p.][\d]+)?)mm(?:[_\-.]|$)',
            original_stem, re.IGNORECASE
        )
        if lsd_match and lsd_guess >= 1000000:  # Only if user didn't provide one
            dist_mm = float(lsd_match.group(1).replace('p', '.'))
            lsd_from_filename = int(dist_mm * 1000)  # mm → µm
            print(f"✓ Extracted Lsd from original filename: {dist_mm} mm → {lsd_from_filename} µm", file=sys.stderr)

        # Build command with all parameters according to MIDAS manual
        # Use MIDAS Python (conda midas_env) instead of current Python (UV)
        midas_python = find_midas_python()
        cmd = [
            midas_python,
            str(autocal_script),
            "-dataFN", str(image_path),
        ]
        if param_path:
            cmd.extend(["-paramFN", str(param_path)])

        # Pass wavelength extracted from original filename if no param file
        if _energy_from_filename and not param_path:
            cmd.extend(["--wavelength", f"{_energy_from_filename:.6f}"])

        # Pass Lsd from original filename if no user-provided guess
        if lsd_match and lsd_guess >= 1000000 and not param_path:
            cmd.extend(["-LsdGuess", str(lsd_from_filename)])

        cmd.extend([
            "-ConvertFile", str(convert_file),
            "--n-iterations", str(n_iterations),
            "--tol-shifts", str(tol_shifts),
            "--tol-rotation", str(tol_rotation),
            "-MultFactor", str(mult_factor),
            "-FirstRingNr", str(first_ring_nr),
            "-EtaBinSize", str(eta_bin_size),
            "-MakePlots", str(make_plots)
        ])

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

        # Image transformation: explicit user_arg → params file → sibling params → 0+warn
        # (per MIDAS manuals/README.md, ImTransOpt is detector-mount specific — no extension rule)
        resolved_transform, transform_source = _resolve_image_transform(
            image_path, image_transform, param_path
        )
        if resolved_transform:
            transforms = resolved_transform.strip().split()
            if transforms:
                cmd.extend(["-ImTransOpt"] + transforms)

        # Add beamline-standard bad pixel and gap intensity markers
        cmd.extend(["-BadPxIntensity", "-2"])
        cmd.extend(["-GapIntensity", "-1"])

        # ── Output directory handling ─────────────────────────────────────────
        # AutoCalibrateZarr.py writes all outputs (refined_MIDAS_params*.txt,
        # autocal.log, calibrant_screen_out.csv) into the working directory
        # (cwd). When output_dir is specified we:
        #   1. Create the directory.
        #   2. Symlink the image (and param/dark files if present) into it so
        #      MIDAS can find them via relative paths.
        #   3. Set cwd = output_dir so all outputs land there.
        # This lets the caller batch-calibrate multiple files, one per subdir,
        # without results clobbering each other.
        work_dir = image_path.parent   # default: same dir as image

        if output_dir:
            out_path = Path(output_dir).expanduser().absolute()
            out_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory: {out_path}", file=sys.stderr)

            # Symlink image into output dir so MIDAS can read it from cwd.
            img_link = out_path / image_path.name
            if not img_link.exists() and not img_link.is_symlink():
                img_link.symlink_to(image_path.resolve())
                print(f"  ↳ symlink: {image_path.name} → {image_path.resolve()}", file=sys.stderr)

            # Update the image path the command receives so MIDAS uses the link
            cmd_image_path = img_link
            # Replace the -dataFN argument already appended above
            datafn_idx = cmd.index("-dataFN")
            cmd[datafn_idx + 1] = str(cmd_image_path)

            # Symlink param file if present
            if param_path and param_path.exists():
                p_link = out_path / param_path.name
                if not p_link.exists() and not p_link.is_symlink():
                    p_link.symlink_to(param_path.resolve())

            # Symlink dark file if present
            if dark_file:
                dark_path_sym = Path(dark_file).expanduser().absolute()
                if dark_path_sym.exists():
                    d_link = out_path / dark_path_sym.name
                    if not d_link.exists() and not d_link.is_symlink():
                        d_link.symlink_to(dark_path_sym.resolve())

            work_dir = out_path

        # ===== TRANSPARENCY: Show exact command being run =====
        cmd_str = " ".join(str(x) for x in cmd)
        print("="*70, file=sys.stderr)
        print("🔧 MIDAS AUTO-CALIBRATION COMMAND:", file=sys.stderr)
        print(f"   Working directory: {work_dir}", file=sys.stderr)
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
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=get_midas_env()
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
        # so glob for any matching file rather than hardcoding the name.
        # Search in work_dir (= output_dir if specified, else image_path.parent).
        output = result.stdout
        candidates = sorted(work_dir.glob("refined_MIDAS_params*.txt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        refined_params_file = candidates[0] if candidates else work_dir / "refined_MIDAS_params.txt"

        # AutoCalibrateZarr.py writes "Dark " (key with no value) when no dark file
        # is provided. ffGenerateZipRefactor.py crashes on empty-value lines.
        # Strip them from the output file so integration doesn't fail.
        if refined_params_file.exists():
            lines = refined_params_file.read_text().splitlines()
            cleaned = [ln for ln in lines if len(ln.split()) >= 2 or not ln.split()]
            if len(cleaned) != len(lines):
                refined_params_file.write_text("\n".join(cleaned) + "\n")
                removed = [ln for ln in lines if ln not in cleaned]
                print(f"  Cleaned {len(removed)} empty-value line(s) from params: {removed}", file=sys.stderr)

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
        message += f"  Beam Center: BC_Y={bc_y:.2f}, BC_X={bc_x:.2f} pixels\n"
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
    dark_location: str = "/exchange/data",
    bright_file: str = None,
    csv_output: bool = False,
    short_names: bool = True,
    r_min: float = None,
    r_max: float = None,
    r_bin_size: float = None,
    eta_min: float = None,
    eta_max: float = None,
    eta_bin_size: float = None,
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
        bright_file: Optional bright/flat-field image (v11, issue #20) — embedded under processed/bright/ in zarr
        csv_output: Also export per-frame lineouts and REtaMap as CSVs (v11, issue #23)
        short_names: Use short v11 output naming (<stem>.zarr.zip). False = legacy suffix-stacking
        r_min, r_max, r_bin_size: Radial integration range overrides (pixels). Defaults: from params file.
        eta_min, eta_max, eta_bin_size: Azimuthal range overrides (degrees). Defaults: from params file.

    The agent SHOULD show the params-file R/eta values back to the user and confirm
    result_folder before invoking this tool. See midas-integrate SKILL.md.

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

        if param_path.suffix.lower() in ('.csv', '.json', '.log', '.bin', '.h5', '.hdf', '.tif'):
            return format_result({"tool": "midas_batch_integrate", "status": "error",
                                  "error": f"Invalid parameter file: {param_path.name} (suffix {param_path.suffix}). "
                                           f"Use refined_MIDAS_params*.txt from midas_auto_calibrate."})

        _strip_empty_value_lines(param_path)

        # Find MIDAS integrator.py via MIDAS_ROOT (set from .env)
        if not MIDAS_ROOT:
            return format_result({"tool": "midas_batch_integrate", "status": "error",
                                  "error": "MIDAS_PATH not set. Add MIDAS_PATH to .env"})
        midas_integrator = MIDAS_ROOT / "FF_HEDM" / "workflows" / "integrator.py"
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
            "-numFrameChunks", str(num_frame_chunks),
            "-shortNames", "1" if short_names else "0",
            "-csvOutput", "1" if csv_output else "0",
        ]
        if bright_file:
            bright_path = Path(bright_file).expanduser().resolve()
            if not bright_path.exists():
                return format_result({"tool": "midas_batch_integrate", "status": "error",
                                      "error": f"Bright file not found: {bright_path}"})
            cmd += ["-brightFN", str(bright_path)]

        # Trailing parameter overrides (R/Eta range — see integrator.py override syntax)
        for key, val in (("RMin", r_min), ("RMax", r_max), ("RBinSize", r_bin_size),
                         ("EtaMin", eta_min), ("EtaMax", eta_max),
                         ("EtaBinSize", eta_bin_size)):
            if val is not None:
                cmd += [key, str(val)]

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
                model_name = os.environ.get("APEXA_EMBED_MODEL",
                                            "nomic-ai/nomic-embed-text-v1.5")
                _knowledge_base = {
                    "client": chromadb.PersistentClient(path=str(chroma_path)),
                    "embedder": SentenceTransformer(model_name, trust_remote_code=True),
                    "model_name": model_name,
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

        # Encode the question. Some embedders (Nomic) require a task prefix;
        # most others don't. Keep this in sync with EMBED_PREFIXES in
        # knowledge_base/index_knowledge.py.
        _query_prefix_map = {
            "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
            "nomic-ai/nomic-embed-text-v1":   "search_query: ",
        }
        q_pfx = _query_prefix_map.get(kb.get("model_name", ""), "")
        query_embedding = kb["embedder"].encode(f"{q_pfx}{question}").tolist()

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

        # Format results — include citation, page, DOI for proper referencing
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
                "citation": meta.get('citation', meta['source']),
                "bibkey": meta.get('bibkey', ''),
                "title": meta.get('title', ''),
                "authors": meta.get('authors', ''),
                "year": meta.get('year', ''),
                "journal": meta.get('journal', ''),
                "doi": meta.get('doi', ''),
                "page": meta.get('page', 0),
                "similarity": round(max(0.0, 1 - dist), 3),
                "excerpt": doc,
                "chunk_index": meta.get('chunk_index', 0)
            })

        # Compact reference list (deduped by source) for the LLM to cite cleanly
        seen = set()
        references = []
        for ex in excerpts:
            if ex['source'] in seen:
                continue
            seen.add(ex['source'])
            references.append({
                "source": ex['source'],
                "citation": ex['citation'],
                "bibkey": ex['bibkey'],
                "doi": ex['doi'],
            })

        return json.dumps({
            "status": "success",
            "question": question,
            "results_count": len(excerpts),
            "references": references,
            "excerpts": excerpts,
            "instruction_to_assistant": (
                "When answering, cite each excerpt inline using the 'citation' field "
                "(e.g. 'Sharma et al. (2012). J. Appl. Cryst. 45:693–704. DOI:...') and, "
                "when available, reference the page number. Use get_bibtex(source) to "
                "retrieve the BibTeX entry for any cited source."
            )
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


@mcp.tool()
async def get_bibtex(source: str) -> str:
    """Return the BibTeX entry for a paper in the knowledge base.

    Reads the sibling .bib sidecar (e.g. HEDM-I.pdf -> HEDM-I.bib) for the
    requested source. Use this after query_hedm_knowledge to cite a result
    in a manuscript or report.

    Args:
        source: PDF filename returned by query_hedm_knowledge (e.g. "HEDM-I.pdf"),
                or the bare stem ("HEDM-I"), or a bibkey ("Sharma2012a").

    Returns:
        JSON with the BibTeX entry text, or an error if unavailable.
    """
    try:
        from pathlib import Path
        papers_dir = Path(__file__).parent / "knowledge_base" / "papers"

        # Normalize: try as filename, stem, then bibkey-search
        candidates = []
        s = source.strip()
        if s.endswith(".pdf"):
            candidates.append(papers_dir / (s[:-4] + ".bib"))
        if s.endswith(".bib"):
            candidates.append(papers_dir / s)
        candidates.append(papers_dir / f"{s}.bib")

        bib_path = next((c for c in candidates if c.exists()), None)

        if bib_path is None:
            # Fallback: scan all .bib files for matching bibkey
            for bib in papers_dir.glob("*.bib"):
                text = bib.read_text(encoding="utf-8", errors="ignore")
                if f"{{{s}," in text or f"{{ {s}," in text:
                    bib_path = bib
                    break

        if bib_path is None:
            return json.dumps({
                "status": "error",
                "error": f"No BibTeX entry found for '{source}'",
                "available": sorted(p.stem for p in papers_dir.glob("*.bib")),
            }, indent=2)

        return json.dumps({
            "status": "success",
            "source": source,
            "bib_file": str(bib_path.name),
            "bibtex": bib_path.read_text(encoding="utf-8"),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
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
            from apexa_units import kev_to_angstrom
            wavelength_angstrom = kev_to_angstrom(beam_energy_kev)
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
# MIDAS VIEWER LAUNCHER
# =============================================================================

# Map of viewer short names → relative paths in MIDAS repo
_VIEWER_SCRIPTS = {
    "plot_calibrant_results":   "gui/viewers/plot_calibrant_results.py",
    "plot_lineout_results":     "gui/viewers/plot_lineout_results.py",
    "plot_lineout_comparison":  "gui/viewers/plot_lineout_comparison.py",
    "plot_integrator_peaks":    "gui/viewers/plot_integrator_peaks.py",
    "plot_caked_peaks":         "gui/viewers/plot_caked_peaks.py",
    "live_viewer":              "gui/viewers/live_viewer.py",
    "interactiveFFplotting":    "gui/viewers/interactiveFFplotting.py",
    "ff_asym_qt":               "gui/ff_asym_qt.py",
    "nf_qt":                    "gui/nf_qt.py",
}

@mcp.tool()
async def run_midas_viewer(
    viewer: str,
    data_file: str,
    param_file: str = "",
    extra_args: str = "",
) -> str:
    """Launch a MIDAS viewer/plotting script on a data file.

    Handles all path resolution internally — finds MIDAS installation,
    midas_env Python, and builds the full command automatically.

    Available viewers:
    - plot_calibrant_results: Plot calibration fit (*_corr.csv)
    - plot_lineout_results: Plot 1D lineout (*_lineout.xy)
    - plot_lineout_comparison: Compare lineouts with ring overlay
    - plot_integrator_peaks: Interactive caked data viewer (*_caked.hdf.zarr.zip)
    - plot_caked_peaks: Peak fitting viewer (*_caked_peaks.h5)
    - live_viewer: Real-time GPU streaming viewer (*_lineout.bin)
    - interactiveFFplotting: FF-HEDM grain viewer (Grains.csv + .zarr)
    - ff_asym_qt: Raw 2D diffraction image viewer
    - nf_qt: NF-HEDM microstructure viewer (.mic/.map)

    Args:
        viewer: Viewer name (e.g. "plot_calibrant_results", "plot_lineout_results")
        data_file: Path to the data file to visualize
        param_file: Optional parameter file (enables 2θ/Q axes in lineout viewers)
        extra_args: Optional extra command-line arguments (e.g. "--nRBins 2000")

    Returns:
        JSON with command executed, stdout/stderr, and status
    """
    try:
        # Resolve viewer script
        rel_path = _VIEWER_SCRIPTS.get(viewer)
        if not rel_path:
            return format_result({
                "tool": "run_midas_viewer",
                "status": "error",
                "error": f"Unknown viewer: '{viewer}'",
                "available_viewers": list(_VIEWER_SCRIPTS.keys()),
            })

        script_path = MIDAS_ROOT / rel_path
        if not script_path.exists():
            return format_result({
                "tool": "run_midas_viewer",
                "status": "error",
                "error": f"Viewer script not found: {script_path}",
            })

        # Resolve data file
        data_path = Path(data_file).expanduser().absolute()
        if not data_path.exists():
            return format_result({
                "tool": "run_midas_viewer",
                "status": "error",
                "error": f"Data file not found: {data_path}",
            })

        # Find the right Python (midas_env conda Python)
        midas_python = find_midas_python()

        # Build command
        cmd = [midas_python, str(script_path), str(data_path)]

        if param_file:
            param_path = Path(param_file).expanduser().absolute()
            if param_path.exists():
                cmd.extend(["--paramFN", str(param_path)])

        if extra_args:
            cmd.extend(extra_args.split())

        env = get_midas_env()

        print(f"  Launching viewer: {viewer} → {data_path.name}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

        return format_result({
            "tool": "run_midas_viewer",
            "status": "success" if result.returncode == 0 else "error",
            "viewer": viewer,
            "data_file": str(data_path),
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        })

    except subprocess.TimeoutExpired:
        return format_result({
            "tool": "run_midas_viewer",
            "status": "error",
            "error": "Viewer timed out after 120s",
        })
    except Exception as e:
        return format_result({
            "tool": "run_midas_viewer",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


# =============================================================================
# GSAS-II REFINEMENT & LIVE ANALYSIS
# =============================================================================

def _extract_instprm_from_zarr(zarr_path: str, output_dir: str) -> Optional[str]:
    """Extract instrument parameters from a MIDAS .zarr.zip and write a .instprm file.

    GSAS-II .xye import requires an instrument parameter file. MIDAS stores
    these in the zarr under InstrumentParameters/ (Distance, Lam, U, V, W, etc.).
    """
    try:
        instprm_path = Path(output_dir) / "_midas_extracted.instprm"

        # Use the GSASII or midas_env Python to read the zarr (handles compression)
        extract_script = (
            "import sys, json\n"
            "data_file = sys.argv[1]\n"
            "try:\n"
            "    import zarr\n"
            "    try:\n"
            "        fp = zarr.open(data_file, mode='r')\n"
            "    except Exception:\n"
            "        import asyncio\n"
            "        async def _o():\n"
            "            s = await zarr.storage.ZipStore.open(data_file, mode='r')\n"
            "            return zarr.open_group(s, mode='r')\n"
            "        fp = asyncio.run(_o())\n"
            "    import numpy as np\n"
            "    ip = fp['InstrumentParameters']\n"
            "    params = {}\n"
            "    for k in ['Lam','Polariz','U','V','W','X','Y','Z','SH_L','Distance']:\n"
            "        if k in ip:\n"
            "            params[k] = float(np.array(ip[k]).flat[0])\n"
            "    print(json.dumps(params))\n"
            "except Exception as e:\n"
            "    print(json.dumps({'error': str(e)}))\n"
        )

        gsasii_python = find_gsasii_python()
        python_exe = gsasii_python or find_midas_python()
        env = get_midas_env()

        result = subprocess.run(
            [python_exe, "-c", extract_script, zarr_path],
            capture_output=True, text=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            return None

        params = json.loads(result.stdout.strip())
        if "error" in params or "Lam" not in params:
            return None

        lines = [
            "#GSAS-II instrument parameter file; do not add/delete items!",
            "Type:PXC",
            f"Lam:{params.get('Lam', 0.2):.6f}",
            "Zero:0.0",
            f"Polariz.:{params.get('Polariz', 0.99):.4f}",
            f"U:{params.get('U', 1.163):.6f}",
            f"V:{params.get('V', -0.126):.6f}",
            f"W:{params.get('W', 0.063):.6f}",
            f"X:{params.get('X', 0.0):.6f}",
            f"Y:{params.get('Y', 0.0):.6f}",
            f"Z:{params.get('Z', 0.0):.6f}",
            f"SH/L:{params.get('SH_L', 0.002):.6f}",
            "Azimuth:0.0",
            "Bank:1",
        ]
        instprm_path.write_text("\n".join(lines) + "\n")
        return str(instprm_path)
    except Exception as e:
        print(f"  Warning: Could not extract instprm from zarr: {e}", file=sys.stderr)
        return None

@mcp.tool()
async def run_gsas_refinement(
    data_file: str,
    cif_files: List[str],
    output_dir: str = "refinement",
    bkg_terms: int = 6,
    two_theta_limits: Optional[List[float]] = None,
    no_atoms: bool = False,
    no_export: bool = False,
    n_cpus: int = 8,
    instprm_file: Optional[str] = None,
    robust: bool = True,
    wavelength_A: Optional[float] = None,
    experimental_a: Optional[float] = None,
    engine: str = "gsas2",
) -> str:
    """Run Rietveld refinement on MIDAS caked output via GSAS-II, MAUD, or both.

    The `engine` parameter selects the backend; the historical name
    `run_gsas_refinement` is retained for backward compatibility but the tool
    now dispatches to one of two drivers (with `engine="both"` running both
    engines and writing a cross-validation summary). When `robust=False` the
    engine selector is ignored and the legacy MIDAS `gsas_ii_refine.py` script
    is invoked exactly as before.

    By default (robust=True, engine="gsas2") APEXA's robust GSAS-II driver
    `apexa_gsas_robust.py` is invoked instead of the raw MIDAS
    `gsas_ii_refine.py`. The robust driver fixes two real-data failure modes
    encountered on the cross-detector benchmark (see
    benchmark/detector_zoo/ground_truth.json):

      * NaN-safe extraction — pixel-array detectors with module gaps (Pilatus,
        Eiger) leave NaN bins in the lineout. The MIDAS extractor passes them
        through and they bias the residual; we drop them per-slice. On Pilatus
        this took |Δa| from 63 mAangstrom to 0.56 mAangstrom and recovered all
        360/360 slices.

      * Data-aware starting cell — Materials Project DFT-relaxed CIFs sit
        ~50 mAangstrom above experimental lattice constants. For low-statistics
        single-frame data (GE) the per-slice landscape does not pull the cell
        across that gap, locking refinement near the wrong minimum. The robust
        driver substitutes a NIST experimental cell when the CIF names a known
        calibrant (CeO2 / LaB6 / Si / Al2O3), or uses the user-supplied
        ``experimental_a``. On GE this took |Δa| from 65 mAangstrom to
        0.07 mAangstrom and recovered 81/81 slices.

    Set robust=False to fall back to the legacy MIDAS script (n_cpus parallelism
    is only available in the legacy path; the robust path runs serially per
    histogram but completes a 360-slice run in ~10 min).

    Stages: Background+Scale → Cell → U,V,W → X,Y,SH/L → (Atoms, optional).

    Prerequisites:
    - GSAS-II installed (conda install gsas2pkg -c briantoby)
    - zarr==2.18.3 in midas_env

    Args:
        data_file: Path to MIDAS .zarr.zip caked output file
        cif_files: One or more CIF files defining crystallographic phase(s)
        output_dir: Output directory for .gpx projects and exports (default: refinement/)
        bkg_terms: Number of Chebyshev background coefficients (default: 6)
        two_theta_limits: Optional 2θ limits in degrees as [LOW, HIGH] (default: full range)
        no_atoms: Skip atomic position / thermal parameter refinement (default: False)
        no_export: Skip CIF and CSV exports after refinement (default: False)
        n_cpus: Number of parallel workers (legacy path only; default: 8)
        instprm_file: Optional .instprm file for GSAS-II instrument parameters
        robust: Use APEXA robust driver with NaN filter + calibrant cell fix (default: True)
        wavelength_A: X-ray wavelength in Angstroms — used by robust driver to
            estimate starting cell from lineout peak positions when CIF cell
            disagrees with data by >20 mAangstrom (default: None, no estimation)
        experimental_a: Explicit override for starting cell a parameter in
            Angstroms (robust driver only). Overrides auto-calibrant detection.
        engine: Rietveld engine to use when robust=True. One of:
            - "gsas2" (default): GSAS-II via `apexa_gsas_robust.py`.
            - "maud":  MAUD via `apexa_maud_milk.py` (MILK Python wrapper;
                      requires MAUD + a JDK on PATH; install hint surfaced
                      automatically when missing).
            - "both":  run gsas2 then maud, write `refinement_crossvalidation.json`
                      reporting |Δa_engines|, the Rwp ratio, and an agreement
                      verdict. If MAUD is unavailable, the call silently falls
                      back to gsas2-only and tags the response.
            Ignored when robust=False (the legacy MIDAS path runs unchanged).

    Returns:
        JSON with refinement summary: Rwp values, lattice parameters, output file paths.
        When robust=True, includes "extraction_stats", "cif_preparation", and
        "robust_features_applied" diagnostics. When engine="both", also includes
        a top-level "crossvalidation" object.
    """
    try:
        data_path = Path(data_file).expanduser().absolute()
        if not data_path.exists():
            return format_result({"tool": "run_gsas_refinement", "status": "error",
                                  "error": f"Data file not found: {data_path}"})

        if not str(data_path).endswith(".zarr.zip"):
            return format_result({"tool": "run_gsas_refinement", "status": "error",
                                  "error": f"Expected .zarr.zip file, got: {data_path.name}"})

        # Validate CIF files
        resolved_cifs = []
        for cif in cif_files:
            cif_path = Path(cif).expanduser().absolute()
            if not cif_path.exists():
                return format_result({"tool": "run_gsas_refinement", "status": "error",
                                      "error": f"CIF file not found: {cif_path}"})
            resolved_cifs.append(str(cif_path))

        # === Engine + driver-script resolution ================================
        # `engine` selects between the two robust drivers; `robust=False`
        # forces the legacy MIDAS path regardless of `engine`.
        if engine not in ("gsas2", "maud", "both"):
            return format_result({"tool": "run_gsas_refinement", "status": "error",
                                  "error": f"Unknown engine={engine!r}. Use 'gsas2', 'maud', or 'both'."})

        apexa_root = Path(__file__).resolve().parent
        gsas2_script  = apexa_root / "apexa_gsas_robust.py"
        maud_script   = apexa_root / "apexa_maud_milk.py"
        legacy_script = MIDAS_ROOT / "utils" / "gsas_ii_refine.py"

        if not robust and not legacy_script.exists():
            return format_result({"tool": "run_gsas_refinement", "status": "error",
                                  "error": f"Legacy gsas_ii_refine.py not found at {legacy_script}. "
                                           "Requires MIDAS v11+."})
        if robust and engine in ("gsas2", "both") and not gsas2_script.exists():
            return format_result({"tool": "run_gsas_refinement", "status": "error",
                                  "error": f"GSAS-II robust driver missing at {gsas2_script}."})

        # MAUD-engine availability is a soft fail (degrade rather than error).
        # Discovery checks the filesystem; the script itself raises EngineUnavailable
        # at import time if MILK / JDK are missing.
        from apexa_engines import find_maud_installation, maud_install_hint, cross_validate, load_summary
        maud_available = maud_script.exists() and find_maud_installation() is not None
        maud_unavailable_msg = None
        if engine == "maud" and not maud_available:
            return format_result({
                "tool": "run_gsas_refinement",
                "status": "engine_unavailable",
                "engine": "maud",
                "install_hint": maud_install_hint(),
                "maud_script_exists": maud_script.exists(),
            })
        if engine == "both" and not maud_available:
            # Fall back to gsas2-only, but flag it in the response.
            maud_unavailable_msg = (
                "MAUD engine unavailable; cross-validation skipped. "
                + maud_install_hint()
            )

        out_path = Path(output_dir).expanduser().absolute()
        out_path.mkdir(parents=True, exist_ok=True)

        # Auto-extract instrument parameters from the zarr if no instprm provided
        generated_instprm = None
        if not instprm_file:
            generated_instprm = _extract_instprm_from_zarr(str(data_path), str(out_path))
            if generated_instprm:
                instprm_file = generated_instprm
                print(f"  Auto-extracted instrument params from zarr", file=sys.stderr)

        # Prefer GSASII env Python for the GSAS-II driver — its binaries match
        # the compiled .so ABI. The MAUD driver runs in the apexa env.
        gsasii_python = find_gsasii_python()
        python_for_gsas2 = gsasii_python or find_midas_python()
        python_for_maud  = sys.executable  # apexa env (MILK + JDK)
        python_for_legacy = python_for_gsas2

        # ------------------------------------------------------------------
        # Inner: build command + execute one driver, return a result dict.
        # The same shape is reused for gsas2, maud, and the legacy path.
        # ------------------------------------------------------------------
        def _invoke(script: Path, python_exe: str, sub_out: Path, label: str) -> dict:
            sub_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                python_exe, str(script),
                "--data", str(data_path),
                "--cif", *resolved_cifs,
                "--out", str(sub_out),
                "--bkg-terms", str(bkg_terms),
            ]
            if script == legacy_script:
                cmd.extend(["--nCPUs", str(n_cpus)])
            if two_theta_limits and len(two_theta_limits) == 2:
                cmd.extend(["--limits", str(two_theta_limits[0]), str(two_theta_limits[1])])
            # Robust drivers (gsas2, maud): atom refinement OFF by default; legacy: ON by default.
            if script != legacy_script:
                if not no_atoms:
                    cmd.append("--refine-atoms")
                if wavelength_A is not None:
                    cmd.extend(["--wavelength", str(wavelength_A)])
                if experimental_a is not None:
                    cmd.extend(["--experimental-a", str(experimental_a)])
            else:
                if no_atoms:
                    cmd.append("--no-atoms")
                if no_export:
                    cmd.append("--no-export")
            if instprm_file:
                instprm_path = Path(instprm_file).expanduser().absolute()
                if instprm_path.exists():
                    cmd.extend(["--instprm", str(instprm_path)])

            env = get_midas_env()
            print(f"  [{label}] {data_path.name} → {sub_out}", file=sys.stderr)
            print(f"  [{label}] python: {python_exe}", file=sys.stderr)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)

            # Read whichever summary file the driver writes.
            summary = {}
            for cand in ("summary.json", "refinement_summary.json"):
                f = sub_out / cand
                if f.exists():
                    try:
                        summary = json.load(open(f))
                        break
                    except Exception:
                        pass

            ok = proc.returncode == 0
            stderr = proc.stderr[-1500:] if proc.stderr else ""
            res = {
                "label": label,
                "driver": script.name,
                "status": "success" if ok else "error",
                "returncode": proc.returncode,
                "output_dir": str(sub_out),
                "summary": summary,
                "python_used": python_exe,
                "stderr": stderr if not ok else "",
                "stdout_tail": (proc.stdout or "")[-500:],
                "command": " ".join(cmd),
            }
            if not ok:
                hint = ""
                if "Cannot import GSASIIscriptable" in stderr or "ModuleNotFoundError" in stderr:
                    hint = (" GSAS-II import failed — install via: "
                            "conda install gsas2full -c briantoby -c conda-forge")
                if "milk" in stderr.lower() or "MAUD" in stderr or "JAVA_HOME" in stderr:
                    hint = " " + maud_install_hint()
                res["error"] = f"{script.name} failed (exit {proc.returncode}).{hint}"
            return res

        # === Dispatch =========================================================
        if not robust:
            res = _invoke(legacy_script, python_for_legacy, out_path, "legacy")
            return format_result({
                "tool": "run_gsas_refinement",
                "status": res["status"],
                "engine": "midas_legacy",
                "driver": "midas_gsas_ii_refine",
                "data_file": str(data_path),
                "cif_files": resolved_cifs,
                "output_dir": str(out_path),
                "summary": res["summary"],
                "python_used": res["python_used"],
                "stdout": res["stdout_tail"],
                "stderr": res.get("stderr", ""),
                **({"error": res["error"]} if "error" in res else {}),
            })

        if engine == "gsas2":
            res = _invoke(gsas2_script, python_for_gsas2, out_path, "gsas2")
            return format_result({
                "tool": "run_gsas_refinement",
                "status": res["status"],
                "engine": "gsas2",
                "driver": "apexa_gsas_robust",
                "data_file": str(data_path),
                "cif_files": resolved_cifs,
                "output_dir": str(out_path),
                "summary": res["summary"],
                "python_used": res["python_used"],
                "stdout": res["stdout_tail"],
                "stderr": res.get("stderr", ""),
                **({"error": res["error"]} if "error" in res else {}),
            })

        if engine == "maud":
            res = _invoke(maud_script, python_for_maud, out_path, "maud")
            return format_result({
                "tool": "run_gsas_refinement",
                "status": res["status"],
                "engine": "maud",
                "driver": "apexa_maud_milk",
                "data_file": str(data_path),
                "cif_files": resolved_cifs,
                "output_dir": str(out_path),
                "summary": res["summary"],
                "python_used": res["python_used"],
                "stdout": res["stdout_tail"],
                "stderr": res.get("stderr", ""),
                **({"error": res["error"]} if "error" in res else {}),
            })

        # engine == "both": run gsas2, then maud (if available), then cross-validate.
        gsas2_dir = out_path / "gsas2"
        maud_dir  = out_path / "maud"
        gsas2_res = _invoke(gsas2_script, python_for_gsas2, gsas2_dir, "gsas2")

        if not maud_available:
            return format_result({
                "tool": "run_gsas_refinement",
                "status": gsas2_res["status"],
                "engine": "both",
                "fallback_used": "gsas2",
                "fallback_reason": maud_unavailable_msg,
                "data_file": str(data_path),
                "cif_files": resolved_cifs,
                "output_dir": str(out_path),
                "gsas2": gsas2_res,
                "maud": None,
                "crossvalidation": None,
            })

        maud_res = _invoke(maud_script, python_for_maud, maud_dir, "maud")

        # Cross-validate when both engines wrote a usable summary.
        cv = None
        if gsas2_res["summary"] and maud_res["summary"]:
            try:
                g = load_summary(gsas2_dir / "summary.json", engine="gsas2") \
                    if (gsas2_dir / "summary.json").exists() \
                    else load_summary(gsas2_dir / "refinement_summary.json", engine="gsas2")
                m = load_summary(maud_dir / "summary.json", engine="maud") \
                    if (maud_dir / "summary.json").exists() \
                    else load_summary(maud_dir / "refinement_summary.json", engine="maud")
                cv = cross_validate(g, m)
                (out_path / "refinement_crossvalidation.json").write_text(
                    json.dumps(cv, indent=2)
                )
            except Exception as e:
                cv = {"verdict": "incomplete", "error": str(e)}

        overall_status = (
            "success" if gsas2_res["status"] == "success" and maud_res["status"] == "success"
            else "partial_success"
        )
        return format_result({
            "tool": "run_gsas_refinement",
            "status": overall_status,
            "engine": "both",
            "data_file": str(data_path),
            "cif_files": resolved_cifs,
            "output_dir": str(out_path),
            "gsas2": gsas2_res,
            "maud": maud_res,
            "crossvalidation": cv,
        })

    except subprocess.TimeoutExpired:
        return format_result({"tool": "run_gsas_refinement", "status": "error",
                              "error": "Refinement timed out after 1800s. Try fewer histograms with two_theta_limits or more n_cpus."})
    except Exception as e:
        return format_result({"tool": "run_gsas_refinement", "status": "error",
                              "error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
async def run_live_analysis(
    backend: str,
    cif_files: List[str],
    param_file: str,
    data_file: Optional[str] = None,
    folder: Optional[str] = None,
    pva: bool = False,
    pva_ip: Optional[str] = None,
    output_dir: str = "refinement",
    dark_file: Optional[str] = None,
    n_cpus: int = 4,
    bkg_terms: int = 6,
    two_theta_limits: Optional[List[float]] = None,
    no_atoms: bool = False,
    skip_integration: bool = False,
    zarr_file: Optional[str] = None,
    skip_refinement: bool = False,
    output_h5: Optional[str] = None,
    data_location: str = "exchange/data",
    dark_location: str = "exchange/dark",
) -> str:
    """Run combined MIDAS integration + GSAS-II refinement pipeline.

    Two-stage pipeline:
      Stage 1: Integration (batch CPU or GPU streaming) → .zarr.zip
      Stage 2: GSAS-II refinement on the caked output → .gpx + summary

    Backends:
    - "batch": CPU-based integrator.py (requires data_file + param_file)
    - "stream": GPU-based integrator_batch_process.py (requires folder or pva + param_file)

    Use --skip-integration with --zarr-file to refine an existing .zarr.zip.
    Use --skip-refinement to only integrate without GSAS-II fitting.

    Args:
        backend: Integration backend: "batch" (CPU) or "stream" (GPU)
        cif_files: CIF file(s) for the crystallographic phase(s)
        param_file: MIDAS parameter file (refined_MIDAS_params*.txt)
        data_file: Path to first data file (batch backend, HDF5 or image)
        folder: Source folder of images (stream backend)
        pva: Enable PVA live-streaming mode (stream backend)
        pva_ip: PVA server IP address (stream backend)
        output_dir: Output directory for refinement results (default: refinement/)
        dark_file: Dark field file for background subtraction
        n_cpus: Number of CPUs for integration and refinement (default: 4)
        bkg_terms: Chebyshev background terms for GSAS-II (default: 6)
        two_theta_limits: Optional 2θ limits as [LOW, HIGH]
        no_atoms: Skip atomic position refinement (default: False)
        skip_integration: Skip Stage 1, use existing zarr_file (default: False)
        zarr_file: Path to existing .zarr.zip (use with skip_integration)
        skip_refinement: Run only Stage 1, no GSAS-II (default: False)
        output_h5: HDF5 output filename for stream backend
        data_location: HDF5 dataset path for batch backend (default: exchange/data)
        dark_location: HDF5 dark dataset path (default: exchange/dark)

    Returns:
        JSON with pipeline results: integration output, refinement summary, file paths
    """
    try:
        # Validate backend
        if backend not in ("batch", "stream"):
            return format_result({"tool": "run_live_analysis", "status": "error",
                                  "error": f"Invalid backend '{backend}'. Use 'batch' or 'stream'."})

        # Find the pipeline script
        pipeline_script = MIDAS_ROOT / "FF_HEDM" / "workflows" / "integrate_and_refine.py"
        if not pipeline_script.exists():
            return format_result({"tool": "run_live_analysis", "status": "error",
                                  "error": f"integrate_and_refine.py not found at {pipeline_script}. "
                                           "Requires MIDAS v11+."})

        # Validate param file
        param_path = Path(param_file).expanduser().absolute()
        if not skip_integration and not param_path.exists():
            return format_result({"tool": "run_live_analysis", "status": "error",
                                  "error": f"Parameter file not found: {param_path}"})

        # Validate CIF files
        resolved_cifs = []
        if not skip_refinement:
            for cif in cif_files:
                cif_path = Path(cif).expanduser().absolute()
                if not cif_path.exists():
                    return format_result({"tool": "run_live_analysis", "status": "error",
                                          "error": f"CIF file not found: {cif_path}"})
                resolved_cifs.append(str(cif_path))

        out_path = Path(output_dir).expanduser().absolute()
        out_path.mkdir(parents=True, exist_ok=True)

        env = get_midas_env()
        midas_python = find_midas_python()
        integration_stdout = ""
        integration_stderr = ""
        zarr_output = None

        # ── Stage 1: Integration (midas_env Python — has diplib/skimage/h5py) ──
        if not skip_integration:
            int_cmd = [
                midas_python, str(pipeline_script),
                "--backend", backend,
                "-nCPUs", str(n_cpus),
                "--skip-refinement",
            ]

            if backend == "batch":
                if not data_file:
                    return format_result({"tool": "run_live_analysis", "status": "error",
                                          "error": "batch backend requires data_file"})
                data_path = Path(data_file).expanduser().absolute()
                if not data_path.exists():
                    return format_result({"tool": "run_live_analysis", "status": "error",
                                          "error": f"Data file not found: {data_path}"})
                int_cmd.extend(["-paramFN", str(param_path), "-dataFN", str(data_path)])
                int_cmd.extend(["-dataLoc", data_location, "-darkLoc", dark_location])
            elif backend == "stream":
                int_cmd.extend(["--param-file", str(param_path)])
                if pva:
                    int_cmd.append("--pva")
                    if pva_ip:
                        int_cmd.extend(["--pva-ip", pva_ip])
                elif folder:
                    folder_path = Path(folder).expanduser().absolute()
                    if not folder_path.exists():
                        return format_result({"tool": "run_live_analysis", "status": "error",
                                              "error": f"Folder not found: {folder_path}"})
                    int_cmd.extend(["--folder", str(folder_path)])
                else:
                    return format_result({"tool": "run_live_analysis", "status": "error",
                                          "error": "stream backend requires --folder or --pva"})
                if output_h5:
                    int_cmd.extend(["--output-h5", output_h5])

            if dark_file:
                dark_path = Path(dark_file).expanduser().absolute()
                if dark_path.exists():
                    if backend == "batch":
                        int_cmd.extend(["-darkFN", str(dark_path)])
                    else:
                        int_cmd.extend(["--dark", str(dark_path)])

            print(f"  Stage 1: Integration ({backend}): {param_path.name}", file=sys.stderr)
            int_result = subprocess.run(int_cmd, capture_output=True, text=True,
                                        timeout=1200, env=env)
            integration_stdout = int_result.stdout
            integration_stderr = int_result.stderr

            if int_result.returncode != 0:
                return format_result({
                    "tool": "run_live_analysis",
                    "status": "error",
                    "stage": "integration",
                    "error": f"Integration failed (exit {int_result.returncode})",
                    "stderr": integration_stderr[-1500:] if integration_stderr else "",
                    "command": " ".join(int_cmd),
                })

            # Find the produced .zarr.zip file
            data_dir = data_path.parent if backend == "batch" else Path(folder).expanduser().absolute()
            for zf in sorted(data_dir.rglob("*_caked.hdf.zarr.zip")):
                zarr_output = str(zf)
                break
            if not zarr_output:
                for zf in sorted(data_dir.rglob("*.zarr.zip")):
                    zarr_output = str(zf)
                    break
        else:
            # skip_integration: use provided zarr_file
            if zarr_file:
                zarr_path = Path(zarr_file).expanduser().absolute()
                if not zarr_path.exists():
                    return format_result({"tool": "run_live_analysis", "status": "error",
                                          "error": f"Zarr file not found: {zarr_path}"})
                zarr_output = str(zarr_path)

        # ── Stage 2: Refinement (GSASII env Python — has matching binaries) ──
        summary = {}
        refinement_stdout = ""
        if not skip_refinement and resolved_cifs and zarr_output:
            refine_script = MIDAS_ROOT / "utils" / "gsas_ii_refine.py"
            if not refine_script.exists():
                return format_result({"tool": "run_live_analysis", "status": "error",
                                      "error": f"gsas_ii_refine.py not found at {refine_script}"})

            gsasii_python = find_gsasii_python()
            refine_python = gsasii_python or midas_python

            # Extract instrument params from zarr for GSAS-II
            live_instprm = _extract_instprm_from_zarr(zarr_output, str(out_path))

            ref_cmd = [
                refine_python, str(refine_script),
                "--data", zarr_output,
                "--cif", *resolved_cifs,
                "--out", str(out_path),
                "--bkg-terms", str(bkg_terms),
                "--nCPUs", str(n_cpus),
            ]
            if live_instprm:
                ref_cmd.extend(["--instprm", live_instprm])
            if two_theta_limits and len(two_theta_limits) == 2:
                ref_cmd.extend(["--limits", str(two_theta_limits[0]), str(two_theta_limits[1])])
            if no_atoms:
                ref_cmd.append("--no-atoms")

            print(f"  Stage 2: GSAS-II refinement → {out_path}", file=sys.stderr)
            print(f"  Python: {refine_python}", file=sys.stderr)
            ref_result = subprocess.run(ref_cmd, capture_output=True, text=True,
                                        timeout=1800, env=env)
            refinement_stdout = ref_result.stdout

            if ref_result.returncode != 0:
                return format_result({
                    "tool": "run_live_analysis",
                    "status": "error",
                    "stage": "refinement",
                    "error": f"GSAS-II refinement failed (exit {ref_result.returncode})",
                    "stderr": ref_result.stderr[-1500:] if ref_result.stderr else "",
                    "zarr_file": zarr_output,
                    "python_used": refine_python,
                    "command": " ".join(ref_cmd),
                })

            summary_file = out_path / "refinement_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)

        mode_desc = "skip→refine" if skip_integration else f"{backend}→refine"
        if skip_refinement:
            mode_desc = f"{backend}→integrate-only"

        output_files = sorted(str(p) for p in out_path.iterdir() if p.is_file()) if out_path.exists() else []

        return format_result({
            "tool": "run_live_analysis",
            "status": "success",
            "mode": mode_desc,
            "backend": backend,
            "param_file": str(param_path),
            "output_dir": str(out_path),
            "zarr_file": zarr_output,
            "output_files": output_files[:30],
            "refinement_summary": summary,
            "stdout": (refinement_stdout or integration_stdout)[-500:],
        })

    except subprocess.TimeoutExpired:
        return format_result({"tool": "run_live_analysis", "status": "error",
                              "error": "Pipeline timed out after 1800s"})
    except Exception as e:
        return format_result({"tool": "run_live_analysis", "status": "error",
                              "error": str(e), "traceback": traceback.format_exc()})


# =============================================================================
# CIF FILE FETCHER (MATERIALS PROJECT)
# =============================================================================

@mcp.tool()
async def fetch_cif_from_mp(
    formula: str,
    output_dir: str = ".",
    mp_api_key: Optional[str] = None,
) -> str:
    """Fetch CIF files from the Materials Project database for a given chemical formula.

    Uses the Materials Project REST API (mp-api) to download crystallographic
    information files (CIF) for use with GSAS-II refinement or MIDAS workflows.

    The API key is read from (in order): mp_api_key parameter, MP_API_KEY env var,
    or ~/.config/.pmgrc.yaml (set by `pmg init`).

    Args:
        formula: Chemical formula (e.g., "CeO2", "LaB6", "Fe", "Ti-6Al-4V" → "Ti")
        output_dir: Directory to save CIF files (default: current directory)
        mp_api_key: Materials Project API key (optional if set in env or config)

    Returns:
        JSON with downloaded CIF file paths and material metadata
    """
    try:
        # Try importing mp_api
        try:
            from mp_api.client import MPRester
        except ImportError:
            return format_result({
                "tool": "fetch_cif_from_mp",
                "status": "error",
                "error": "mp-api not installed. Install with: uv pip install mp-api",
                "suggestion": "Run: uv sync --extra extra"
            })

        out_path = Path(output_dir).expanduser().absolute()
        out_path.mkdir(parents=True, exist_ok=True)

        # Resolve API key
        api_key = mp_api_key or os.environ.get("MP_API_KEY")

        if not api_key:
            return format_result({
                "tool": "fetch_cif_from_mp",
                "status": "error",
                "error": "No Materials Project API key provided.",
                "suggestion": "Set MP_API_KEY in .env or pass mp_api_key parameter. "
                              "Get a key at https://next-gen.materialsproject.org/api"
            })

        # Set env var so MPRester always finds it (some versions ignore the kwarg)
        os.environ["MP_API_KEY"] = api_key

        results = []
        with MPRester(api_key) as mpr:
            # Query for structures matching the formula
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure",
                         "symmetry", "energy_above_hull", "is_stable"]
            )

            if not docs:
                return format_result({
                    "tool": "fetch_cif_from_mp",
                    "status": "not_found",
                    "formula": formula,
                    "error": f"No structures found for '{formula}' in Materials Project"
                })

            # Sort: stable structures first, then by energy above hull
            docs_sorted = sorted(docs, key=lambda d: (
                0 if d.is_stable else 1,
                d.energy_above_hull or 999
            ))
            for doc in docs_sorted[:3]:
                # Convert to conventional standard cell with proper space group
                # (MP stores primitive P1 cells; GSAS-II needs conventional with symmetry)
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    from pymatgen.io.cif import CifWriter
                    sga = SpacegroupAnalyzer(doc.structure)
                    conv_struct = sga.get_conventional_standard_structure()
                    cif_str = str(CifWriter(conv_struct, symprec=0.1))
                except Exception:
                    cif_str = doc.structure.to(fmt="cif")
                mp_id = doc.material_id
                cif_filename = f"{doc.formula_pretty}_{mp_id}.cif"
                cif_path = out_path / cif_filename

                with open(cif_path, "w") as f:
                    f.write(cif_str)

                results.append({
                    "material_id": str(mp_id),
                    "formula": doc.formula_pretty,
                    "space_group": doc.symmetry.symbol if doc.symmetry else "unknown",
                    "crystal_system": str(doc.symmetry.crystal_system) if doc.symmetry else "unknown",
                    "energy_above_hull_eV": doc.energy_above_hull,
                    "is_stable": doc.is_stable,
                    "cif_file": str(cif_path),
                })

        return format_result({
            "tool": "fetch_cif_from_mp",
            "status": "success",
            "formula": formula,
            "structures_found": len(docs),
            "downloaded": len(results),
            "files": results,
        })

    except Exception as e:
        return format_result({
            "tool": "fetch_cif_from_mp",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# =============================================================================
# PARAMETER VALIDATION TOOLS (midas-params)
# =============================================================================

def _resolve_param_file(path_str: str) -> tuple[bool, str]:
    """Resolve a parameter file path — accepts a file OR a directory.

    If a directory is given, searches for parameter files in priority order:
    Parameters.txt, refined_MIDAS_params*.txt, ps_*.txt, *.txt (param-like).
    """
    path = Path(path_str).expanduser()
    if path.is_file():
        return True, str(path)

    if path.is_dir():
        candidates = []
        # Priority 1: Parameters.txt / parameters.txt (standard names)
        # Use os.listdir for case-exact matching (macOS FS is case-insensitive)
        try:
            actual_names = set(os.listdir(path))
        except OSError:
            actual_names = set()
        for name in ["Parameters.txt", "parameters.txt"]:
            if name in actual_names:
                return True, str(path / name)
        # Priority 2: refined_MIDAS_params*.txt (calibration output)
        candidates.extend(sorted(path.glob("refined_MIDAS_params*.txt"), reverse=True))
        # Priority 3: ps_*.txt (parameter set files)
        candidates.extend(sorted(path.glob("ps_*.txt")))
        if candidates:
            return True, str(candidates[0])
        # Priority 4: any .txt file that looks like a param file (has key=value lines)
        for txt in sorted(path.glob("*.txt")):
            try:
                with open(txt) as f:
                    first_lines = f.read(500)
                if any(kw in first_lines for kw in ["Lsd", "Wavelength", "NrPixels", "SpaceGroup"]):
                    return True, str(txt)
            except Exception:
                continue
        return False, f"No parameter file found in directory: {path}"

    return False, f"Path not found: {path}"


def _find_midas_params_cli() -> str:
    """Locate the midas-params CLI binary (installed alongside midas_env Python)."""
    midas_python = find_midas_python()
    cli_path = str(Path(midas_python).parent / "midas-params")
    if Path(cli_path).exists():
        return cli_path
    return None


def _run_midas_params(subcommand_args: list, timeout: int = 60) -> dict:
    """Run a midas-params CLI subcommand and return parsed JSON output."""
    cli = _find_midas_params_cli()
    if cli is None:
        midas_python = find_midas_python()
        cmd = [midas_python, "-c",
               "from midas_params.cli import main; import sys; sys.exit(main())",
               ] + subcommand_args
    else:
        cmd = [cli] + subcommand_args

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, env=get_midas_env(),
    )
    if result.stdout.strip():
        return json.loads(result.stdout)
    raise RuntimeError(result.stderr or "midas-params produced no output")


@mcp.tool()
async def validate_parameter_file(
    param_file: str,
    pipeline: str = "ff"
) -> str:
    """Validate a MIDAS parameter file against pipeline-specific rules.

    Checks required keys, value ranges (wavelength, LSD, beam center),
    omega consistency, 12 cross-field rules, and detects typos with
    edit-distance suggestions. Run this BEFORE any HEDM workflow to
    catch configuration errors early and save beamtime.

    Args:
        param_file: Path to MIDAS Parameters.txt file, OR a directory
                   (auto-finds Parameters.txt / refined_MIDAS_params*.txt)
        pipeline: Pipeline to validate against: ff, nf, pf, or ri

    Returns:
        JSON with validation status, error/warning counts, and issues
        with line numbers, severity, and fix suggestions
    """
    try:
        valid, param_path = _resolve_param_file(param_file)
        if not valid:
            return format_result({"tool": "validate_parameter_file",
                                  "status": "error", "error": param_path})

        if pipeline not in ("ff", "nf", "pf", "ri"):
            return format_result({"tool": "validate_parameter_file",
                                  "status": "error",
                                  "error": f"Invalid pipeline '{pipeline}'. Must be ff, nf, pf, or ri"})

        report = _run_midas_params(["validate", param_path, "--path", pipeline, "--json"])

        return format_result({
            "tool": "validate_parameter_file",
            "status": "valid" if report.get("ok") else "invalid",
            "param_file": param_path,
            "pipeline": pipeline,
            "errors": report.get("errors", 0),
            "warnings": report.get("warnings", 0),
            "issues": report.get("issues", []),
        })

    except FileNotFoundError:
        return format_result({"tool": "validate_parameter_file", "status": "error",
                              "error": "midas-params not installed. Run: pip install -e $MIDAS_PATH/packages/midas_params"})
    except Exception as e:
        return format_result({"tool": "validate_parameter_file", "status": "error", "error": str(e)})


@mcp.tool()
async def diagnose_parameter_file(
    param_file: str,
    pipeline: str = "ff",
    output_format: str = "json"
) -> str:
    """Generate LLM-optimized diagnosis of a MIDAS parameter file.

    Produces a structured payload with validation issues, pipeline primer
    context, parameter registry info, and actionable suggestions designed
    for AI-assisted debugging. Use this after validate_parameter_file
    reports errors to get detailed fix guidance.

    Args:
        param_file: Path to MIDAS Parameters.txt file, OR a directory
                   (auto-finds Parameters.txt / refined_MIDAS_params*.txt)
        pipeline: Pipeline to diagnose against: ff, nf, pf, or ri
        output_format: Output format — 'json' (structured) or 'prompt' (LLM-ready text)

    Returns:
        JSON or text with complete diagnosis including line-level issues,
        parameter specs, and fix suggestions
    """
    try:
        valid, param_path = _resolve_param_file(param_file)
        if not valid:
            return format_result({"tool": "diagnose_parameter_file",
                                  "status": "error", "error": param_path})

        if pipeline not in ("ff", "nf", "pf", "ri"):
            return format_result({"tool": "diagnose_parameter_file",
                                  "status": "error",
                                  "error": f"Invalid pipeline '{pipeline}'. Must be ff, nf, pf, or ri"})

        fmt = "json" if output_format not in ("json", "prompt") else output_format
        args = ["diagnose", param_path, "--path", pipeline, "--format", fmt]

        if fmt == "prompt":
            cli = _find_midas_params_cli()
            if cli is None:
                midas_python = find_midas_python()
                cmd = [midas_python, "-c",
                       "from midas_params.cli import main; import sys; sys.exit(main())"
                       ] + args
            else:
                cmd = [cli] + args
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=60, env=get_midas_env(),
            )
            return format_result({
                "tool": "diagnose_parameter_file",
                "status": "success",
                "param_file": param_path,
                "pipeline": pipeline,
                "format": "prompt",
                "diagnosis": result.stdout,
            })

        diagnosis = _run_midas_params(args)
        return format_result({
            "tool": "diagnose_parameter_file",
            "status": "success",
            "param_file": param_path,
            "pipeline": pipeline,
            "format": "json",
            "diagnosis": diagnosis,
        })

    except FileNotFoundError:
        return format_result({"tool": "diagnose_parameter_file", "status": "error",
                              "error": "midas-params not installed. Run: pip install -e $MIDAS_PATH/packages/midas_params"})
    except Exception as e:
        return format_result({"tool": "diagnose_parameter_file", "status": "error", "error": str(e)})


@mcp.tool()
async def inspect_dataset_file(
    dataset_file: str
) -> str:
    """Auto-extract parameters from a raw dataset file.

    Reads GE, HDF5, Zarr, or TIFF files and extracts detector geometry,
    beam energy, pixel size, image dimensions, and other parameters
    embedded in the file headers or metadata. Useful for pre-populating
    parameter files or verifying consistency.

    Args:
        dataset_file: Path to data file (GE2/GE3/GE5, HDF5, Zarr, TIFF)

    Returns:
        JSON with extracted parameters, confidence levels, and sources
    """
    try:
        valid, dataset_path = validate_file(dataset_file)
        if not valid:
            return format_result({"tool": "inspect_dataset_file",
                                  "status": "error", "error": dataset_path})

        report = _run_midas_params(["inspect", dataset_path, "--json"])

        return format_result({
            "tool": "inspect_dataset_file",
            "status": "success",
            "dataset_file": dataset_path,
            "extracted": report,
        })

    except FileNotFoundError:
        return format_result({"tool": "inspect_dataset_file", "status": "error",
                              "error": "midas-params not installed. Run: pip install -e $MIDAS_PATH/packages/midas_params"})
    except Exception as e:
        return format_result({"tool": "inspect_dataset_file", "status": "error", "error": str(e)})


@mcp.tool()
async def enumerate_bragg_rings(
    param_file: str
) -> str:
    """List Bragg rings that fall on the detector for a given crystal/geometry.

    Reads wavelength, detector distance, lattice parameters, and space group
    from a MIDAS parameter file and enumerates all Bragg rings with their
    hkl indices, d-spacing, 2theta, and detector radius. Helps verify correct
    RingThresh and OverAllRingToIndex settings.

    Args:
        param_file: Path to MIDAS Parameters.txt with crystal/geometry info,
                   OR a directory (auto-finds the param file)

    Returns:
        JSON with ring list (hkl, d-spacing, 2theta, radius, on-detector flag)
    """
    try:
        valid, param_path = _resolve_param_file(param_file)
        if not valid:
            return format_result({"tool": "enumerate_bragg_rings",
                                  "status": "error", "error": param_path})

        report = _run_midas_params(["rings", "--from", param_path, "--json"])

        return format_result({
            "tool": "enumerate_bragg_rings",
            "status": "success",
            "param_file": param_path,
            "rings": report,
        })

    except FileNotFoundError:
        return format_result({"tool": "enumerate_bragg_rings", "status": "error",
                              "error": "midas-params not installed. Run: pip install -e $MIDAS_PATH/packages/midas_params"})
    except Exception as e:
        return format_result({"tool": "enumerate_bragg_rings", "status": "error", "error": str(e)})


# =============================================================================
# STRESS / STRAIN ANALYSIS TOOLS (midas-stress)
# =============================================================================

def _run_stress_runner(subcommand_args: list, timeout: int = 120) -> dict:
    """Run a _stress_runner.py subcommand and return parsed JSON output."""
    midas_python = find_midas_python()
    cmd = [midas_python, str(STRESS_RUNNER_SCRIPT)] + subcommand_args

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, env=get_midas_env(),
    )
    if result.stdout.strip():
        return json.loads(result.stdout)
    raise RuntimeError(result.stderr or "_stress_runner.py produced no output")


@mcp.tool()
async def read_grains_summary(
    grains_file: str
) -> str:
    """Read a MIDAS Grains.csv or HDF5 file and return a statistical summary.

    Reports grain count, position ranges, orientation spread, strain tensor
    statistics, lattice parameters, and confidence distribution. A quick
    "what's in this file" diagnostic for post-reconstruction data.

    Args:
        grains_file: Path to Grains.csv or consolidated .h5 file

    Returns:
        JSON with grain population statistics
    """
    try:
        valid, grains_path = validate_file(grains_file)
        if not valid:
            return format_result({"tool": "read_grains_summary",
                                  "status": "error", "error": grains_path})

        output = _run_stress_runner(["read_grains", "--grains", grains_path])
        return format_result({"tool": "read_grains_summary", **output})

    except Exception as e:
        return format_result({"tool": "read_grains_summary", "status": "error", "error": str(e)})


@mcp.tool()
async def compute_grain_stress(
    grains_file: str,
    material: str,
    applied_stress: str = "",
    min_confidence: float = 0.0
) -> str:
    """Compute per-grain stress from HEDM reconstruction output.

    Reads grain orientations and strains from a MIDAS Grains.csv file,
    applies single-crystal Hooke's law with equilibrium correction, and
    returns stress decomposition (hydrostatic, deviatoric, von Mises) with
    uncertainty estimates.

    Supported materials: Au, Cu, Al, Fe, Ni, Ti, W, Si, CeO2

    Args:
        grains_file: Path to Grains.csv or consolidated .h5 file
        material: Material name (e.g., "Cu", "Fe", "Ti")
        applied_stress: Applied macroscopic stress as 6 comma-separated Voigt
                       components in GPa (e.g., "0.1,0,0,0,0,0" for uniaxial).
                       Empty string = free-standing sample.
        min_confidence: Minimum grain confidence for equilibrium averaging (0-1)

    Returns:
        JSON with von Mises statistics, hydrostatic shift (d0 proxy), and uncertainty
    """
    try:
        valid, grains_path = validate_file(grains_file)
        if not valid:
            return format_result({"tool": "compute_grain_stress",
                                  "status": "error", "error": grains_path})

        cmd_args = ["compute_stress", "--grains", grains_path,
                    "--material", material,
                    "--min-confidence", str(min_confidence)]
        if applied_stress:
            cmd_args.extend(["--applied-stress", applied_stress])

        output = _run_stress_runner(cmd_args)
        return format_result({"tool": "compute_grain_stress", **output})

    except Exception as e:
        return format_result({"tool": "compute_grain_stress", "status": "error", "error": str(e)})


@mcp.tool()
async def get_material_stiffness(
    material: str
) -> str:
    """Look up single-crystal elastic stiffness matrix and d0 sensitivity.

    Returns the full 6x6 stiffness matrix in Voigt-Mandel notation (GPa),
    independent elastic constants, bulk modulus, and d0 sensitivity (how
    much stress error results from a given d0 uncertainty).

    Supported materials: Au, Cu, Al, Fe, Ni, Ti, W, Si, CeO2

    Args:
        material: Material name (e.g., "Cu", "Fe", "Ti")

    Returns:
        JSON with stiffness matrix, elastic constants, d0 sensitivity, and
        list of all available materials
    """
    try:
        output = _run_stress_runner(["material_info", "--material", material])
        return format_result({"tool": "get_material_stiffness", **output})

    except Exception as e:
        return format_result({"tool": "get_material_stiffness", "status": "error", "error": str(e)})


@mcp.tool()
async def correct_d0_equilibrium(
    grains_file: str,
    material: str,
    applied_stress: str = "",
    min_confidence: float = 0.0
) -> str:
    """Apply two-step d0 correction to grain stress data.

    The d0 reference lattice parameter is the dominant systematic error
    in HEDM stress analysis. This tool applies:
    1. Strain-level correction: fits isotropic strain offset (eps_iso)
    2. Stress-level correction: enforces mechanical equilibrium

    Reports the correction magnitude, residual norms before/after, and
    corrected stress statistics.

    Args:
        grains_file: Path to Grains.csv or consolidated .h5 file
        material: Material name (e.g., "Cu", "Fe", "Ti")
        applied_stress: Applied macroscopic stress as 6 comma-separated Voigt
                       components in GPa. Empty = free-standing.
        min_confidence: Minimum grain confidence for equilibrium averaging (0-1)

    Returns:
        JSON with eps_iso correction (ppm), residual improvement, corrected stress stats
    """
    try:
        valid, grains_path = validate_file(grains_file)
        if not valid:
            return format_result({"tool": "correct_d0_equilibrium",
                                  "status": "error", "error": grains_path})

        cmd_args = ["correct_d0", "--grains", grains_path,
                    "--material", material,
                    "--min-confidence", str(min_confidence)]
        if applied_stress:
            cmd_args.extend(["--applied-stress", applied_stress])

        output = _run_stress_runner(cmd_args)
        return format_result({"tool": "correct_d0_equilibrium", **output})

    except Exception as e:
        return format_result({"tool": "correct_d0_equilibrium", "status": "error", "error": str(e)})


@mcp.tool()
async def analyze_slip_systems(
    grains_file: str,
    material: str,
    load_direction: str = "0,0,1",
    crss: float = 0.0
) -> str:
    """Compute Schmid factors, Taylor factor, and yield proximity for grains.

    Performs slip-system analysis from HEDM stress data:
    - Schmid factors for each grain and slip system
    - Dominant slip system per grain
    - Taylor factor (polycrystal average)
    - Yield proximity ranking (if CRSS provided)

    Automatically selects slip families by material (FCC, BCC, or HCP).

    Args:
        grains_file: Path to Grains.csv or consolidated .h5 file
        material: Material name (e.g., "Cu" for FCC, "Fe" for BCC, "Ti" for HCP)
        load_direction: Loading direction in sample frame as 3 comma-separated
                       values (e.g., "0,0,1" for z-axis loading)
        crss: Critical resolved shear stress in MPa. Set > 0 to compute
              yield proximity (which grains yield first). Default 0 = skip.

    Returns:
        JSON with Schmid factor statistics, Taylor factor, and yield proximity
    """
    try:
        valid, grains_path = validate_file(grains_file)
        if not valid:
            return format_result({"tool": "analyze_slip_systems",
                                  "status": "error", "error": grains_path})

        cmd_args = ["plasticity", "--grains", grains_path,
                    "--material", material,
                    "--load-direction", load_direction,
                    "--crss", str(crss)]

        output = _run_stress_runner(cmd_args)
        return format_result({"tool": "analyze_slip_systems", **output})

    except Exception as e:
        return format_result({"tool": "analyze_slip_systems", "status": "error", "error": str(e)})


# =============================================================================
# MIDAS v11 PYTORCH PIPELINE WRAPPERS
# (packages/midas_ff_pipeline, midas_fit_grain, midas_nf_preprocess)
# =============================================================================

def _run_midas_module(module: str, args: list, timeout: int = 7200) -> dict:
    """Invoke `python -m <module>` with MIDAS env. Returns dict with status/stdout/stderr."""
    midas_python = find_midas_python()
    cmd = [midas_python, "-m", module] + [str(a) for a in args]
    print(f"\n  $ {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=get_midas_env())
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"{module} timed out (>{timeout}s)"}
    if result.returncode != 0:
        for line in result.stderr.strip().splitlines()[-20:]:
            print(f"  {line}", file=sys.stderr)
        return {"status": "error", "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "error": f"{module} exited {result.returncode}"}
    return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}


@mcp.tool()
async def run_ff_pipeline(
    params: str,
    result: str,
    zarr: str = None,
    detectors: str = None,
    layers: str = "1-1",
    n_cpus: int = 16,
    device: str = "cuda",
    dtype: str = "auto",
    resume: str = "auto",
    resume_from: str = None,
    only: list = None,
    skip: list = None,
    solver: str = "lbfgs",
    loss: str = "pixel",
    mode: str = "",
    group_size: str = "auto",
    shard_gpus: str = "auto",
    pg_mode: str = "spot_aware",
    raw_dir: str = None,
    grains_file: str = None,
    nf_result_dir: str = None,
    batch: bool = False,
    generate_h5: bool = False,
    extra_args: list = None,
) -> str:
    """Run the differentiable PyTorch FF-HEDM pipeline (midas_ff_pipeline.run).

    Wraps the v11 ``midas-ff-pipeline run`` console script — the new PyTorch port
    of the FF-HEDM workflow with checkpoint/resume, multi-GPU sharding, and
    swappable solvers/losses. Drop-in replacement for legacy ``ff_MIDAS.py``.

    Args:
        params: Path to Parameters.txt (required).
        result: Result directory for this run (required).
        zarr: Override zarr.zip path (auto-derived if omitted).
        detectors: Detector spec (e.g. "1" or "1,2,3,4" for Hydra).
        layers: Layer range, e.g. "1-1" or "1-3".
        n_cpus: CPU thread count for non-GPU stages.
        device: "cuda" | "cpu" | "mps".
        dtype: "auto" | "float32" | "float64".
        resume: Resume policy — "none" | "auto" | "from".
        resume_from: Stage to resume from when resume="from".
        only: Stage allow-list (one per --only flag).
        skip: Stage skip-list (one per --skip flag).
        solver: "lbfgs" | "lm" | "nelder_mead" | "adam" | "lm_batched".
        loss: "pixel" | "angular" | "internal_angle".
        mode: Optional mode override.
        group_size: Per-grain batch group size or "auto".
        shard_gpus: GPU sharding spec or "auto".
        pg_mode: Peak-group mode (default "spot_aware").
        raw_dir: Raw data directory (when not in params).
        grains_file: Pre-computed grains file (skip indexing).
        nf_result_dir: Companion NF reconstruction directory (for FF↔NF cross-checks).
        batch: Enable batched execution.
        generate_h5: Emit consolidated HDF5 output.
        extra_args: Any additional argv to forward verbatim (advanced).
    """
    args = ["run", "--params", params, "--result", result, "--layers", layers,
            "--n-cpus", n_cpus, "--device", device, "--dtype", dtype,
            "--resume", resume, "--solver", solver, "--loss", loss,
            "--group-size", group_size, "--shard-gpus", shard_gpus,
            "--pg-mode", pg_mode]
    if zarr:           args += ["--zarr", zarr]
    if detectors:      args += ["--detectors", detectors]
    if resume_from:    args += ["--from", resume_from]
    if mode:           args += ["--mode", mode]
    if raw_dir:        args += ["--raw-dir", raw_dir]
    if grains_file:    args += ["--grains-file", grains_file]
    if nf_result_dir:  args += ["--nf-result-dir", nf_result_dir]
    if batch:          args += ["--batch"]
    if generate_h5:    args += ["--generate-h5"]
    for s in (only or []):  args += ["--only", s]
    for s in (skip or []):  args += ["--skip", s]
    if extra_args:     args += list(extra_args)

    out = _run_midas_module("midas_ff_pipeline.cli", args, timeout=14400)
    payload = {"tool": "run_ff_pipeline", "result_dir": str(Path(result).expanduser().absolute()), **out}
    return format_result(payload)


@mcp.tool()
async def refine_grain_lattice(
    param_file: str,
    block_nr: int = 0,
    num_blocks: int = 1,
    num_lines: int = 0,
    num_procs: int = 0,
    solver: str = "lbfgs",
    mode: str = None,
    loss: str = "pixel",
    device: str = None,
    dtype: str = None,
    max_iter: int = 200,
    ftol: float = 1e-7,
    xtol: float = 1e-9,
    csv: bool = False,
    verbose: int = 0,
) -> str:
    """Refine grain position/orientation/strain with the PyTorch fitter
    (midas_fit_grain — drop-in for FitPosOrStrainsOMP / FitPosOrStrainsGPU).

    Mirrors the legacy C-binary positional argv:
        param_file blockNr numBlocks numLines numProcs

    Args:
        param_file: paramstest.txt produced by FitSetupParamsAllZarr / ff_MIDAS.
        block_nr: 0-based block index.
        num_blocks: Total number of blocks.
        num_lines: Number of grain seeds in SpotsToIndex.csv (0 = read from disk).
        num_procs: CPU thread count for torch.set_num_threads (0 = auto).
        solver: "lbfgs" | "adam" | "lm" | "nelder_mead" | "lm_batched".
        mode: "iterative" | "all_at_once" (default: iterative if FitAllAtOnce=0).
        loss: "pixel" (C parity) | "angular" | "internal_angle".
        device: Override MIDAS_FIT_GRAIN_DEVICE ("cuda"|"mps"|"cpu").
        dtype: Override MIDAS_FIT_GRAIN_DTYPE ("float32"|"float64").
        max_iter: Outer-iteration cap per phase.
        ftol: Relative-loss convergence threshold.
        xtol: Parameter-delta convergence threshold.
        csv: Also dump human-readable FitBest.csv next to the binary.
        verbose: 0=warning, 1=info, 2=debug.
    """
    args = [param_file, block_nr, num_blocks, num_lines, num_procs,
            "--solver", solver, "--loss", loss,
            "--max-iter", max_iter, "--ftol", ftol, "--xtol", xtol]
    if mode:    args += ["--mode", mode]
    if device:  args += ["--device", device]
    if dtype:   args += ["--dtype", dtype]
    if csv:     args += ["--csv"]
    for _ in range(min(verbose, 2)):
        args += ["-v"]

    out = _run_midas_module("midas_fit_grain.cli", args, timeout=3600)
    payload = {"tool": "refine_grain_lattice",
               "param_file": str(Path(param_file).expanduser().absolute()),
               "block": f"{block_nr}/{num_blocks}",
               **out}
    return format_result(payload)


@mcp.tool()
async def preprocess_nf_data(
    subcommand: str,
    args: list = None,
) -> str:
    """Run the NF-HEDM PyTorch preprocessing umbrella CLI (midas_nf_preprocess).

    Subcommands (mirror the standalone modules):
      - hex-grid           : generate the voxel grid (port of MakeHexGrid)
      - tomo-filter        : mask the grid by a tomography image / bbox
      - diffr-spots        : forward-simulate diffraction spots (port of MakeDiffrSpots)
      - process-images     : raw TIFF -> SpotsInfo.bin (port of ProcessImagesCombined)
      - seed-orientations  : seed grain orientations

    Args:
        subcommand: Which subcommand to run (see list above).
        args: Argv list for the subcommand. Pass exactly as you would on the
              shell (e.g. ["--paramFN", "params.txt", "--output", "grid.bin"]).
              Each subcommand has its own flag set — call with
              args=["--help"] to discover them.
    """
    valid = {"hex-grid", "tomo-filter", "diffr-spots", "process-images", "seed-orientations"}
    if subcommand not in valid:
        return format_result({"tool": "preprocess_nf_data", "status": "error",
                              "error": f"Unknown subcommand '{subcommand}'. "
                                       f"Valid: {sorted(valid)}"})

    cli_args = [subcommand] + [str(a) for a in (args or [])]
    out = _run_midas_module("midas_nf_preprocess.cli", cli_args, timeout=3600)
    payload = {"tool": "preprocess_nf_data", "subcommand": subcommand, **out}
    return format_result(payload)


# =============================================================================
# SERVER MAIN
# =============================================================================

if __name__ == "__main__":
    # Pre-warm the knowledge base (loads ~700MB Nomic embedder + ChromaDB client).
    # Without this, the first query_hedm_knowledge call pays a ~6 s cold-start
    # cost while the user is waiting. Cost is paid once at server startup
    # (hidden behind APEXA's normal banner) instead of on the first query.
    try:
        import sys as _sys
        kb = get_knowledge_base()
        if kb.get("available"):
            print("[midas] knowledge base pre-warmed", file=_sys.stderr, flush=True)
    except Exception as _e:
        print(f"[midas] kb warmup skipped: {_e}", file=_sys.stderr, flush=True)

    mcp.run(transport='stdio')
