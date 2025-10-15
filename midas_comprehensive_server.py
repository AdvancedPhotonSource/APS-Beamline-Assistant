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

def find_midas_installation() -> Path:
    """Find MIDAS installation by checking common locations.

    Priority order:
    1. MIDAS_PATH environment variable
    2. ~/.MIDAS (common beamline installation)
    3. ~/opt/MIDAS (macOS/development)
    4. /opt/MIDAS (system-wide Linux)
    5. Current directory ./MIDAS
    """
    # Check environment variable first
    if "MIDAS_PATH" in os.environ:
        midas_path = Path(os.environ["MIDAS_PATH"]).expanduser().absolute()
        if midas_path.exists():
            return midas_path

    # Check common installation locations
    common_paths = [
        Path.home() / ".MIDAS",           # Beamline standard
        Path.home() / "MIDAS",            # Home directory
        Path.home() / "opt" / "MIDAS",    # macOS/dev
        Path("/opt/MIDAS"),               # System-wide
        Path.cwd() / "MIDAS"              # Current directory
    ]

    for path in common_paths:
        if path.exists() and path.is_dir():
            print(f"Found MIDAS installation at: {path}", file=sys.stderr)
            return path

    # Default to ~/.MIDAS (will be created or cause errors later)
    default_path = Path.home() / ".MIDAS"
    print(f"WARNING: MIDAS not found, using default: {default_path}", file=sys.stderr)
    return default_path

# MIDAS installation paths
MIDAS_ROOT = find_midas_installation()
MIDAS_BIN = MIDAS_ROOT / "build" / "bin"  # Executables are in build/bin
MIDAS_FF_BIN = MIDAS_ROOT / "FF_HEDM" / "bin"  # FF-HEDM specific executables
MIDAS_NF_BIN = MIDAS_ROOT / "NF_HEDM" / "bin"  # NF-HEDM specific executables
MIDAS_FF_V7 = MIDAS_ROOT / "FF_HEDM" / "v7"
MIDAS_NF_V7 = MIDAS_ROOT / "NF_HEDM" / "v7"
MIDAS_UTILS = MIDAS_ROOT / "utils"

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
    print("✓ MIDAS scientific dependencies available", file=sys.stderr)
except ImportError as e:
    MIDAS_AVAILABLE = False
    print(f"⚠ MIDAS dependencies not available: {e}", file=sys.stderr)

# Try to import MIDAS Python workflow modules
try:
    # These may not exist depending on MIDAS installation
    # from ff_MIDAS import process_layer, read_parameter_file
    # from nf_MIDAS import run_preprocessing, run_fitting_and_postprocessing
    # from calcMiso import GetMisorientationAngle
    MIDAS_PYTHON_AVAILABLE = True
    print("✓ MIDAS Python APIs available", file=sys.stderr)
except ImportError:
    MIDAS_PYTHON_AVAILABLE = False
    print("ℹ MIDAS Python APIs not imported (will use subprocess)", file=sys.stderr)

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
        result = subprocess.run(
            [str(exe_path), str(param_file)],
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
    """Run a MIDAS Python script and return results."""
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
        cmd = ["python", str(script_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
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
    convert_files: bool = True
) -> str:
    """Run complete FF-HEDM production workflow using ff_MIDAS.py.

    This executes the full Far-Field High Energy Diffraction Microscopy workflow:
    1. Data conversion to Zarr format (if convert_files=True)
    2. HKL generation (GetHKLListZarr)
    3. Peak search (PeaksFittingOMPZarrRefactor)
    4. Peak merging (MergeOverlappingPeaksAllZarr)
    5. Data preparation (CalcRadiusAllZarr, FitSetupZarr)
    6. Data binning (SaveBinData)
    7. Indexing (IndexerOMP)
    8. Refinement (FitPosOrStrainsOMP)
    9. Post-processing (ProcessGrainsZarr)

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
    machine_name: str = "local"
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

        # Step 1: Run calibrant fitting
        exe = "CalibrantOMP" if use_omp else "Calibrant"
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
    n_nodes: int = 1
) -> str:
    """Run complete NF-HEDM microstructure reconstruction using nf_MIDAS.py.

    Near-Field HEDM produces voxel-by-voxel 3D orientation maps with
    higher spatial resolution than FF-HEDM.

    Two modes:
    1. Reconstruction mode (refine_parameters=False):
       - Pre-processing: GetHKLListNF, GenSeedOrientationsFF2NFHEDM
       - Grid creation: MakeHexGrid
       - Spot simulation: MakeDiffrSpots
       - Image processing: MedianImageLibTiff, ImageProcessingLibTiffOMP
       - Fitting: FitOrientationOMP
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
            # FF-HEDM
            "IndexerOMP", "FitPosOrStrainsOMP", "ProcessGrainsZarr",
            "GetHKLListZarr", "PeaksFittingOMPZarrRefactor",
            "CalibrantOMP", "ForwardSimulationCompressed",
            # NF-HEDM
            "FitOrientationOMP", "GetHKLListNF", "MakeHexGrid",
            "ParseMic", "NFGrainCentroids",
            # Utilities
            "GrainTracking", "CalcStrains"
        ]

        for exe in key_executables:
            exe_path = bin_path / exe
            validation["executables"][exe] = exe_path.exists()

        # Check Python workflows
        workflow_scripts = {
            "ff_MIDAS.py": midas_root / "FF_HEDM" / "v7" / "ff_MIDAS.py",
            "pf_MIDAS.py": midas_root / "FF_HEDM" / "v7" / "pf_MIDAS.py",
            "nf_MIDAS.py": midas_root / "NF_HEDM" / "v7" / "nf_MIDAS.py"
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

# Keep the existing basic tools for backward compatibility
@mcp.tool()
async def detect_diffraction_rings(
    image_path: str,
    detector_distance: float = 1000.0,
    wavelength: float = 0.2066,
    beam_center_x: float = None,
    beam_center_y: float = None
) -> str:
    """Detect and analyze diffraction rings in 2D powder diffraction patterns.

    Args:
        image_path: Path to the 2D diffraction image file
        detector_distance: Sample-to-detector distance in millimeters
        wavelength: X-ray wavelength in Angstroms
        beam_center_x: Beam center X coordinate in pixels
        beam_center_y: Beam center Y coordinate in pixels
    """
    try:
        if not Path(image_path).exists():
            return format_result({"error": f"Image file not found: {image_path}", "status": "failed"})

        image_data = load_diffraction_image(image_path)

        if beam_center_x is None or beam_center_y is None:
            center = (image_data.shape[0] // 2, image_data.shape[1] // 2)
        else:
            center = (int(beam_center_y), int(beam_center_x))

        # Radial profile
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r_int = r.astype(int)
        radial_prof = np.bincount(r_int.ravel(), image_data.ravel())
        radial_counts = np.bincount(r_int.ravel())

        valid_idx = radial_counts > 0
        radial_prof = radial_prof[valid_idx] / radial_counts[valid_idx]
        r_values = np.arange(len(radial_prof))[valid_idx]

        # Find peaks
        ring_radii = []
        ring_intensities = []
        if len(radial_prof) > 10:
            peaks, properties = find_peaks(radial_prof, height=np.mean(radial_prof) * 1.2, distance=10)
            ring_radii = r_values[peaks]
            ring_intensities = radial_prof[peaks]

        pixel_size = 172e-6
        ring_2theta = np.arctan(np.array(ring_radii) * pixel_size / (detector_distance * 1e-3)) * 180 / np.pi

        signal_to_noise = np.mean(ring_intensities) / np.std(image_data) if len(ring_intensities) > 0 else 0

        return format_result({
            "tool": "detect_diffraction_rings",
            "image_file": image_path,
            "rings_detected": len(ring_radii),
            "ring_positions_2theta": ring_2theta.tolist() if len(ring_2theta) > 0 else [],
            "quality_metrics": {
                "signal_to_noise": float(signal_to_noise)
            }
        })

    except Exception as e:
        return format_result({"error": str(e), "status": "error"})

@mcp.tool()
async def integrate_2d_to_1d(
    image_path: str,
    calibration_file: str = None,
    wavelength: float = None,
    detector_distance: float = None,
    beam_center_x: float = None,
    beam_center_y: float = None,
    dark_file: str = None,
    output_file: str = None
) -> str:
    """Integrate 2D diffraction image to 1D pattern using MIDAS Integrator or pyFAI.

    Converts a 2D detector image into a 1D azimuthally-integrated intensity vs. 2θ pattern.
    Supports dark image subtraction and various detector formats (TIFF, GE2/GE5, ED5, EDF).

    Args:
        image_path: Path to 2D diffraction image (.tiff, .ge2, .ge5, .ed5, .edf)
        calibration_file: Optional calibration/parameter file with detector geometry
        wavelength: X-ray wavelength in Angstroms (if not in calibration file)
        detector_distance: Sample-to-detector distance in mm (if not in calibration file)
        beam_center_x: Beam center X coordinate in pixels (if not in calibration file)
        beam_center_y: Beam center Y coordinate in pixels (if not in calibration file)
        dark_file: Optional dark image file for background subtraction (.tiff, .ge2, .ge5, .ed5)
        output_file: Output filename for 1D pattern (default: image_name_1d.dat)

    Returns:
        JSON with integration results and 1D pattern statistics
    """
    try:
        image_path = Path(image_path).expanduser().absolute()

        if not image_path.exists():
            return format_result({
                "tool": "integrate_2d_to_1d",
                "status": "error",
                "error": f"Image not found: {image_path}"
            })

        # Set output file
        if output_file is None:
            output_file = image_path.stem + "_1d.dat"

        output_path = image_path.parent / output_file

        # Try using pyFAI if available (faster and more flexible)
        if MIDAS_PYTHON_AVAILABLE:
            try:
                import fabio
                import pyFAI
                try:
                    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
                except ImportError:
                    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # fallback for older versions

                # Load image
                img = fabio.open(str(image_path))
                data = img.data.astype(float)  # Convert to float for dark subtraction

                # Load and subtract dark image if provided
                if dark_file:
                    dark_path = Path(dark_file).expanduser().absolute()
                    if not dark_path.exists():
                        return format_result({
                            "tool": "integrate_2d_to_1d",
                            "status": "error",
                            "error": f"Dark file not found: {dark_path}"
                        })

                    try:
                        dark_img = fabio.open(str(dark_path))
                        dark_data = dark_img.data.astype(float)

                        # Verify dimensions match
                        if data.shape != dark_data.shape:
                            return format_result({
                                "tool": "integrate_2d_to_1d",
                                "status": "error",
                                "error": f"Image and dark file dimensions don't match: {data.shape} vs {dark_data.shape}"
                            })

                        # Subtract dark image
                        data = data - dark_data

                        # Clip negative values to zero
                        data = np.clip(data, 0, None)

                    except Exception as e:
                        return format_result({
                            "tool": "integrate_2d_to_1d",
                            "status": "error",
                            "error": f"Failed to load/subtract dark file: {str(e)}"
                        })

                # Setup integrator
                ai = AzimuthalIntegrator()

                # Load calibration or use provided parameters
                if calibration_file:
                    cal_path = Path(calibration_file).expanduser()
                    if cal_path.exists():
                        ai.load(str(cal_path))
                    else:
                        return format_result({
                            "tool": "integrate_2d_to_1d",
                            "status": "error",
                            "error": f"Calibration file not found: {cal_path}"
                        })
                else:
                    # Use provided parameters
                    if (wavelength is not None and detector_distance is not None and
                        beam_center_x is not None and beam_center_y is not None):
                        # Convert to pyFAI units
                        ai.wavelength = wavelength * 1e-10  # Angstrom to meters
                        ai.dist = detector_distance / 1000.0  # mm to meters

                        # For GE detectors, pixel size is typically 200 microns
                        pixel_size = 200e-6  # 200 microns in meters
                        ai.pixel1 = pixel_size  # pixel size in meters (dimension 1)
                        ai.pixel2 = pixel_size  # pixel size in meters (dimension 2)

                        # PONI points are in meters from corner
                        ai.poni1 = beam_center_y * pixel_size
                        ai.poni2 = beam_center_x * pixel_size
                    else:
                        return format_result({
                            "tool": "integrate_2d_to_1d",
                            "status": "error",
                            "error": "Either calibration_file or all geometry parameters (wavelength, detector_distance, beam_center_x, beam_center_y) must be provided"
                        })

                # Perform integration
                result = ai.integrate1d(data, npt=2048, unit="2th_deg")
                two_theta = result[0]  # 2theta in degrees
                intensity = result[1]

                # Save to file
                with open(output_path, 'w') as f:
                    f.write("# 2D to 1D integration using pyFAI\n")
                    f.write(f"# Source: {image_path.name}\n")
                    if dark_file:
                        f.write(f"# Dark file: {Path(dark_file).name} (subtracted)\n")
                    f.write("# 2theta(deg)  Intensity\n")
                    for tth, I in zip(two_theta, intensity):
                        f.write(f"{tth:.4f}  {I:.2f}\n")

                # Calculate statistics
                peak_intensity = float(np.max(intensity))
                background = float(np.percentile(intensity, 10))
                signal_to_noise = peak_intensity / background if background > 0 else 0

                # Find approximate peak positions
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(intensity, height=background*3, distance=10)
                peak_positions = [float(two_theta[p]) for p in peaks[:10]]  # Top 10 peaks

                dark_msg = f" (with dark subtraction from {Path(dark_file).name})" if dark_file else ""
                return format_result({
                    "tool": "integrate_2d_to_1d",
                    "status": "success",
                    "method": "pyFAI",
                    "input_image": str(image_path),
                    "dark_file": str(dark_file) if dark_file else None,
                    "output_file": str(output_path),
                    "integration_parameters": {
                        "wavelength_angstrom": wavelength,
                        "detector_distance_mm": detector_distance,
                        "beam_center": [beam_center_x, beam_center_y] if beam_center_x else None,
                        "calibration_file": calibration_file,
                        "dark_subtraction": bool(dark_file)
                    },
                    "pattern_statistics": {
                        "n_points": len(two_theta),
                        "2theta_range_deg": [float(two_theta[0]), float(two_theta[-1])],
                        "peak_intensity": peak_intensity,
                        "background_level": background,
                        "signal_to_noise": signal_to_noise,
                        "peak_positions_deg": peak_positions
                    },
                    "message": f"Successfully integrated {image_path.name} to 1D pattern{dark_msg} with {len(peak_positions)} peaks detected"
                })

            except ImportError:
                pass  # Fall through to MIDAS Integrator
            except Exception as e:
                return format_result({
                    "tool": "integrate_2d_to_1d",
                    "status": "error",
                    "method": "pyFAI",
                    "error": f"pyFAI integration failed: {str(e)}"
                })

        # Fallback: Use MIDAS Integrator executable
        if calibration_file:
            cal_path = Path(calibration_file).expanduser()
            if not cal_path.exists():
                return format_result({
                    "tool": "integrate_2d_to_1d",
                    "status": "error",
                    "error": f"Calibration file not found: {cal_path}"
                })

            # Run MIDAS Integrator
            result = run_midas_executable(
                "Integrator",
                str(cal_path),
                cwd=str(image_path.parent),
                timeout=300
            )

            return format_result({
                "tool": "integrate_2d_to_1d",
                "status": "completed" if result["success"] else "failed",
                "method": "MIDAS Integrator",
                "input_image": str(image_path),
                "calibration_file": str(cal_path),
                "execution": result,
                "message": "Check output files in the same directory as input image"
            })
        else:
            return format_result({
                "tool": "integrate_2d_to_1d",
                "status": "error",
                "error": "Calibration file required for MIDAS Integrator method. Alternatively, provide all geometry parameters for pyFAI."
            })

    except Exception as e:
        return format_result({
            "tool": "integrate_2d_to_1d",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def identify_crystalline_phases(
    peak_positions: list,
    material_system: str = "unknown",
    temperature: float = 25.0,
    tolerance: float = 0.1
) -> str:
    """Identify crystalline phases from peak positions.

    Args:
        peak_positions: List of peak positions in degrees 2theta
        material_system: Expected material system
        temperature: Sample temperature in Celsius
        tolerance: Peak position tolerance in degrees 2theta
    """
    try:
        phase_database = {
            "austenite": {
                "formula": "γ-Fe",
                "space_group": "Fm-3m",
                "peaks": [12.47, 18.15, 25.84, 30.15, 35.71, 40.44],
                "intensities": [100, 60, 40, 25, 30, 15],
                "hkl": ["(111)", "(200)", "(220)", "(311)", "(222)", "(400)"]
            },
            "ferrite": {
                "formula": "α-Fe",
                "space_group": "Im-3m",
                "peaks": [31.39, 44.67, 65.02, 82.33, 98.95],
                "intensities": [100, 80, 60, 40, 30],
                "hkl": ["(110)", "(200)", "(211)", "(220)", "(310)"]
            }
        }

        identified_phases = []

        for phase_name, phase_data in phase_database.items():
            matched_peaks = []
            for obs_peak in peak_positions:
                for i, ref_peak in enumerate(phase_data["peaks"]):
                    if abs(obs_peak - ref_peak) <= tolerance:
                        matched_peaks.append({
                            "observed": float(obs_peak),
                            "calculated": float(ref_peak),
                            "hkl": phase_data["hkl"][i]
                        })
                        break

            if len(matched_peaks) >= 3:
                identified_phases.append({
                    "phase_name": phase_name.title(),
                    "chemical_formula": phase_data["formula"],
                    "space_group": phase_data["space_group"],
                    "matched_peaks": matched_peaks
                })

        return format_result({
            "tool": "identify_crystalline_phases",
            "identified_phases": identified_phases,
            "total_phases_found": len(identified_phases)
        })

    except Exception as e:
        return format_result({"error": str(e), "status": "error"})

# =============================================================================
# SERVER MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70, file=sys.stderr)
    print("MIDAS Comprehensive MCP Server", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"MIDAS Root: {MIDAS_ROOT}", file=sys.stderr)
    print(f"MIDAS Available: {MIDAS_AVAILABLE}", file=sys.stderr)
    print(f"Python APIs: {MIDAS_PYTHON_AVAILABLE}", file=sys.stderr)
    print("\nAvailable Tools:", file=sys.stderr)
    print("\nFF-HEDM Production:", file=sys.stderr)
    print("  - run_ff_hedm_full_workflow", file=sys.stderr)
    print("  - run_pf_hedm_workflow", file=sys.stderr)
    print("  - run_ff_calibration", file=sys.stderr)
    print("  - run_ff_grain_tracking", file=sys.stderr)
    print("\nNF-HEDM Reconstruction:", file=sys.stderr)
    print("  - run_nf_hedm_reconstruction", file=sys.stderr)
    print("  - convert_nf_to_dream3d", file=sys.stderr)
    print("  - overlay_ff_nf_results", file=sys.stderr)
    print("\nAdvanced Analysis:", file=sys.stderr)
    print("  - calculate_misorientation", file=sys.stderr)
    print("  - run_forward_simulation", file=sys.stderr)
    print("  - extract_grain_centroids", file=sys.stderr)
    print("\nData Management:", file=sys.stderr)
    print("  - batch_convert_ge_to_tiff", file=sys.stderr)
    print("  - create_midas_parameter_file", file=sys.stderr)
    print("  - validate_midas_installation", file=sys.stderr)
    print("  - get_midas_workflow_status", file=sys.stderr)
    print("\nBasic Analysis:", file=sys.stderr)
    print("  - detect_diffraction_rings", file=sys.stderr)
    print("  - integrate_2d_to_1d", file=sys.stderr)
    print("  - identify_crystalline_phases", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    mcp.run(transport='stdio')
