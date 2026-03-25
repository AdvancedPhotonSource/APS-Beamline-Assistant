#!/usr/bin/env python3
"""
Beamline Core Operations MCP Server
Unified server combining filesystem, command execution, and X-ray utilities

This server consolidates:
- filesystem_server.py → File operations
- command_executor_server.py → Safe command execution
- analysis_utilities_server.py → X-ray calculations and utilities

Author: Beamline Assistant Team
Organization: Argonne National Laboratory
Created: 2025 (Unified architecture)
"""

from typing import Any, Optional, List, Dict
import json
import sys
import os
import logging
from pathlib import Path

# Suppress noisy third-party library startup messages
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
import subprocess
import stat
import time
import platform
import shutil
import numpy as np
import logging
from mcp.server.fastmcp import FastMCP

# Suppress verbose MCP server logging
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# =============================================================================
# INITIALIZATION
# =============================================================================

mcp = FastMCP("beamline-core")

# Try to import X-ray utilities (domain-specific package)
try:
    import xrayutilities as xu
    import xrayutilities.materials as xumaterials
    XRAYUTILITIES_AVAILABLE = True
except ImportError:
    XRAYUTILITIES_AVAILABLE = False
    print("⚠️ xrayutilities not available - install with: pip install xrayutilities", file=sys.stderr)

# Try to import optional dependencies for image analysis
try:
    import fabio
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy/fabio not available - some image analysis features limited", file=sys.stderr)

# =============================================================================
# SECURITY: ALLOWED COMMANDS
# =============================================================================

ALLOWED_COMMANDS = {
    # Basic system commands
    'ls', 'pwd', 'cat', 'head', 'tail', 'grep', 'find',
    'python', 'python3', 'pip', 'pip3', 'uv', 'git',
    'ps', 'df', 'du', 'whoami', 'env', 'echo', 'date',
    'which', 'file', 'wc', 'sort', 'uniq',

    # ── FF-HEDM CPU executables ──────────────────────────────────────────────
    'GetHKLListZarr',
    'PeaksFittingOMPZarrRefactor',
    'MergeOverlappingPeaksAllZarr',
    'CalcRadiusAllZarr',
    'FitSetupZarr',
    'SaveBinData',
    'SaveBinDataScanning',
    'IndexerOMP',
    'IndexerScanningOMP',
    'FitPosOrStrainsOMP',
    'FitOrStrainsScanningOMP',
    'FitPosOrStrainsDoubleDataset',
    'FitMultipleGrains',
    'FitWedgeParallel',
    'ProcessGrains',
    'ProcessGrainsScanningHEDM',
    'MatchGrains',
    'MergeMultipleScans',
    'mergeScansScanning',

    # ── FF-HEDM GPU executables (v10) ────────────────────────────────────────
    'IndexerGPU',
    'IndexerScanningGPU',
    'FitPosOrStrainsGPU',
    'FitOrStrainsScanningGPU',
    'IntegratorFitPeaksGPUStream',

    # ── FF-HEDM calibration & integration ───────────────────────────────────
    'CalibrantIntegratorOMP',       # primary calibrant integrator (v10)
    'CalibrantPanelShiftsOMP',      # multi-panel shift calibration (v10)
    'CalibrantOMP',                 # legacy (keep for compatibility)
    'FitTiltBCLsdSample',
    'IntegratorZarrOMP',            # primary integrator (v10)
    'IntegratorZarrOMP_f32',        # float32 variant
    'DetectorMapper',
    'MapDatasets',
    'ForwardSimulationCompressed',
    'findSingleSolutionPFRefactored',
    'findMultipleSolutionsPF',
    'FindSaturatedPixels',
    'CalcRadius',

    # ── NF-HEDM executables ──────────────────────────────────────────────────
    'GetHKLListNF',
    'MakeHexGrid',
    'MakeDiffrSpots',
    'MedianImageLibTiff',
    'ProcessImagesCombined',        # replaces ImageProcessingLibTiffOMP (v10)
    'MMapImageInfo',
    'FitOrientationOMP',
    'FitOrientationGPU',            # GPU NF reconstruction (v10)
    'FitOrientationParameters',
    'FitOrientationParametersMultiPoint',
    'GenSeedOrientationsFF2NFHEDM',
    'SimulateDiffractionSpots',
    'filterGridfromTomo',
    'ParseMic',
    'ParseDeconvOutput',
    'Mic2GrainsList',
    'NFGrainCentroids',
    'compareNF',
    'simulateNF',

    # ── TOMO executables ─────────────────────────────────────────────────────
    'GenMedianDark',

    # ── Python workflow drivers ──────────────────────────────────────────────
    'ff_MIDAS.py',
    'nf_MIDAS.py',
    'nf_MIDAS_Multiple_Resolutions.py',
    'pf_MIDAS.py',
    'integrator.py',
    'integrator_server.py',
    'integrator_batch_process.py',
    'AutoCalibrateZarr.py',
    'ffGenerateZipRefactor.py',
    'ffGenerateZip.py',
    'runDTrecon.py',

    # ── Python utilities (v10) ───────────────────────────────────────────────
    'match_grains.py',
    'phase_id.py',
    'validate_calibration.py',
    'generate_mask.py',
    'undistort_image.py',
    'updateZarrDset.py',
    'extract_lineouts.py',
    'fit_caked_peaks.py',
    'BatchCake.py',
    'runScanning.py',
    'evalScanning.py',
    'calcMiso.py',
    'simulatePeaks.py',
    'blobPeaksearch.py',
    'nf_paraview_gen.py',
    'nf_mic_to_grains.py',
    'NFGrainCentroids.py',
    'DL2FF.py',
    'GE2Tiff.py',
    'hdf_gen_nf.py',
    'PlotFFNF.py',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_result(result: dict) -> str:
    """Format results as JSON string"""
    return json.dumps(result, indent=2, default=str)

def format_file_info(path: Path) -> dict:
    """Get detailed file information"""
    try:
        stat_info = path.stat()
        return {
            "name": path.name,
            "path": str(path.absolute()),
            "size": stat_info.st_size,
            "modified": time.ctime(stat_info.st_mtime),
            "created": time.ctime(stat_info.st_ctime),
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "permissions": stat.filemode(stat_info.st_mode),
            "owner_readable": bool(stat_info.st_mode & stat.S_IRUSR),
            "owner_writable": bool(stat_info.st_mode & stat.S_IWUSR),
            "owner_executable": bool(stat_info.st_mode & stat.S_IXUSR)
        }
    except Exception as e:
        return {"error": str(e)}

def is_command_allowed(command: str) -> bool:
    """Check if command is in allowed list"""
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False
    base_command = cmd_parts[0].lower()
    return base_command in ALLOWED_COMMANDS

# =============================================================================
# SECTION 1: FILESYSTEM OPERATIONS
# =============================================================================

@mcp.tool()
async def list_directory(path: str = ".", show_hidden: bool = False, details: bool = False) -> str:
    """List contents of a directory.

    Args:
        path: Directory path to list (default: current directory)
        show_hidden: Include hidden files/directories starting with '.' (default: False)
        details: Show detailed file information (default: False)

    Returns:
        JSON with directory contents
    """
    try:
        dir_path = Path(path).expanduser()

        if not dir_path.exists():
            return format_result({"error": f"Directory does not exist: {path}"})

        if not dir_path.is_dir():
            return format_result({"error": f"Path is not a directory: {path}"})

        def human_size(n):
            for unit in ('B', 'K', 'M', 'G'):
                if n < 1024:
                    return f"{n:.0f}{unit}" if unit == 'B' else f"{n:.1f}{unit}"
                n /= 1024
            return f"{n:.1f}T"

        lines = []
        hidden_count = 0
        def sort_key(p):
            try:
                return (not p.is_symlink() and p.is_file(), p.name.lower())
            except OSError:
                return (True, p.name.lower())

        for item in sorted(dir_path.iterdir(), key=sort_key):
            if item.name.startswith('.'):
                if not show_hidden:
                    hidden_count += 1
                    continue
            if item.is_symlink():
                try:
                    target = os.readlink(item)
                    resolved = item.resolve()
                    if resolved.is_dir():
                        lines.append(f"  {item.name}/ -> {target}")
                    else:
                        sz = human_size(item.stat().st_size)
                        lines.append(f"  {item.name:<50}  {sz:>6}  -> {target}")
                except OSError:
                    lines.append(f"  {item.name:<50}  [broken symlink]")
            elif item.is_dir():
                lines.append(f"  {item.name}/")
            else:
                try:
                    sz = human_size(item.stat().st_size)
                except OSError:
                    sz = "?"
                lines.append(f"  {item.name:<50}  {sz:>6}")

        output = f"{str(dir_path.absolute())}\n"
        output += "\n".join(lines)
        if hidden_count:
            output += f"\n  ({hidden_count} hidden items not shown — use show_hidden=True)"

        return format_result({
            "tool": "list_directory",
            "path": str(dir_path.absolute()),
            "listing": output
        })

    except Exception as e:
        return format_result({"error": f"Error listing directory: {str(e)}"})

@mcp.tool()
async def read_file(file_path: str, encoding: str = "utf-8", max_size: int = 1024000) -> str:
    """Read contents of a text file.

    Args:
        file_path: Path to the file to read
        encoding: Text encoding (default: utf-8)
        max_size: Maximum file size to read in bytes (default: 1MB)

    Returns:
        JSON with file contents
    """
    try:
        path = Path(file_path).expanduser()

        if not path.exists():
            return format_result({"error": f"File does not exist: {file_path}"})

        if not path.is_file():
            return format_result({"error": f"Path is not a file: {file_path}"})

        file_size = path.stat().st_size
        if file_size > max_size:
            return format_result({"error": f"File too large ({file_size} bytes > {max_size} bytes limit)"})

        # Detect if file is binary
        with open(path, 'rb') as f:
            sample = f.read(1024)
            if b'\x00' in sample:
                return format_result({"error": f"File appears to be binary: {file_path}"})

        # Read text file
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()

        return format_result({
            "tool": "read_file",
            "file_path": str(path.absolute()),
            "size": file_size,
            "encoding": encoding,
            "line_count": content.count('\n') + 1,
            "content": content[:10000],
            "truncated": len(content) > 10000
        })

    except Exception as e:
        return format_result({"error": f"Error reading file: {str(e)}"})

@mcp.tool()
async def write_file(file_path: str, content: str, encoding: str = "utf-8", append: bool = False) -> str:
    """Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: Text encoding (default: utf-8)
        append: Append to file instead of overwriting (default: False)

    Returns:
        JSON with operation status
    """
    try:
        path = Path(file_path).expanduser()
        mode = 'a' if append else 'w'

        with open(path, mode, encoding=encoding) as f:
            f.write(content)

        return format_result({
            "tool": "write_file",
            "status": "success",
            "file_path": str(path.absolute()),
            "bytes_written": len(content.encode(encoding)),
            "mode": "append" if append else "overwrite"
        })

    except Exception as e:
        return format_result({"error": f"Error writing file: {str(e)}"})

@mcp.tool()
async def get_file_info(file_path: str) -> str:
    """Get detailed information about a file or directory.

    Args:
        file_path: Path to the file or directory

    Returns:
        JSON with detailed file information
    """
    try:
        path = Path(file_path).expanduser()

        if not path.exists():
            return format_result({"error": f"Path does not exist: {file_path}"})

        info = format_file_info(path)
        info["tool"] = "get_file_info"

        return format_result(info)

    except Exception as e:
        return format_result({"error": f"Error getting file info: {str(e)}"})

# =============================================================================
# SECTION 2: COMMAND EXECUTION
# =============================================================================

@mcp.tool()
async def run_command(command: str, working_dir: str = None, timeout: int = 30) -> str:
    """Execute a system command safely.

    Args:
        command: Command to execute (must be in allowed list)
        working_dir: Working directory for command execution
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        JSON with command output and status
    """
    try:
        if not is_command_allowed(command):
            cmd_name = command.split()[0] if command.split() else command
            return format_result({
                "error": f"Command not allowed: {cmd_name}",
                "allowed_commands": "Use check_environment to see allowed commands"
            })

        cwd = working_dir if working_dir and Path(working_dir).exists() else None

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )

        return format_result({
            "tool": "run_command",
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "working_dir": str(cwd) if cwd else str(Path.cwd())
        })

    except subprocess.TimeoutExpired:
        return format_result({"error": f"Command timed out after {timeout} seconds"})
    except Exception as e:
        return format_result({"error": f"Error executing command: {str(e)}"})

@mcp.tool()
async def check_environment() -> str:
    """Check system environment and available tools.

    Returns:
        JSON with system information and available MIDAS tools
    """
    try:
        info = {
            "tool": "check_environment",
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "architecture": platform.machine(),
                "hostname": platform.node()
            },
            "paths": {
                "current_dir": str(Path.cwd()),
                "home_dir": str(Path.home()),
                "python_executable": sys.executable
            },
            "available_commands": sorted(list(ALLOWED_COMMANDS)),
            "optional_libraries": {
                "scipy": SCIPY_AVAILABLE,
                "fabio": SCIPY_AVAILABLE
            }
        }

        # Check for MIDAS installation
        midas_path = os.environ.get("MIDAS_PATH")
        if midas_path and Path(midas_path).exists():
            info["midas"] = {
                "installed": True,
                "path": midas_path
            }
        else:
            info["midas"] = {
                "installed": False,
                "message": "Set MIDAS_PATH environment variable"
            }

        return format_result(info)

    except Exception as e:
        return format_result({"error": f"Error checking environment: {str(e)}"})

# =============================================================================
# SECTION 3: X-RAY UTILITIES & CALCULATIONS (using xrayutilities)
# =============================================================================

@mcp.tool()
async def xray_calculate(
    calculation_type: str,
    h: int = None,
    k: int = None,
    l: int = None,
    material: str = None,
    lattice_a: float = None,
    two_theta_degrees: float = None,
    wavelength_angstroms: float = None,
    energy_kev: float = None,
    d_spacing: float = None,
    measured_d: float = None,
    reference_d: float = None
) -> str:
    """Universal X-ray calculation tool using xrayutilities library.

    🎯 CRITICAL: This tool MUST be used for ALL X-ray calculations.
    Uses domain-specific xrayutilities package for accurate scientific calculations.

    Supported calculation types:
    - "d_from_hkl": Calculate d-spacing from Miller indices
    - "d_from_angle": Calculate d-spacing from 2θ (Bragg's law)
    - "angle_from_d": Calculate 2θ from d-spacing
    - "energy_to_wavelength": Convert energy (keV) to wavelength (Å)
    - "wavelength_to_energy": Convert wavelength (Å) to energy (keV)
    - "strain": Calculate strain from measured and reference d-spacings
    - "list_materials": List available materials in xrayutilities

    Args:
        calculation_type: Type of calculation (see above)
        h, k, l: Miller indices (for d_from_hkl)
        material: Material name (e.g., "Fe", "Si", "Al") from xrayutilities
        lattice_a: Lattice parameter in Å (for custom materials)
        two_theta_degrees: Diffraction angle 2θ in degrees
        wavelength_angstroms: X-ray wavelength in Å
        energy_kev: X-ray energy in keV
        d_spacing: d-spacing in Å
        measured_d: Measured d-spacing in Å (for strain)
        reference_d: Reference d-spacing in Å (for strain)

    Returns:
        JSON with calculation results

    Examples:
        xray_calculate("d_from_hkl", h=1, k=1, l=0, material="Fe")
        xray_calculate("d_from_angle", two_theta_degrees=12.5, wavelength_angstroms=0.202)
        xray_calculate("energy_to_wavelength", energy_kev=61.332)
    """
    try:
        if not XRAYUTILITIES_AVAILABLE:
            return format_result({
                "error": "xrayutilities library not available",
                "install": "pip install xrayutilities",
                "fallback": "Some basic calculations available without xrayutilities"
            })

        # D-SPACING FROM MILLER INDICES
        if calculation_type == "d_from_hkl":
            if h is None or k is None or l is None:
                return format_result({"error": "h, k, l Miller indices required"})

            if material:
                # Use xrayutilities materials database
                try:
                    mat = getattr(xumaterials, material)
                    d = mat.planeDistance(h, k, l)

                    return format_result({
                        "tool": "xray_calculate",
                        "library": "xrayutilities",
                        "calculation": "d_from_hkl",
                        "inputs": {
                            "miller_indices": f"({h}{k}{l})",
                            "material": material,
                            "lattice_parameters": {
                                "a": mat.a,
                                "b": mat.b if hasattr(mat, 'b') else mat.a,
                                "c": mat.c if hasattr(mat, 'c') else mat.a
                            }
                        },
                        "result": {
                            "d_spacing_angstroms": round(d, 6),
                            "d_spacing_nm": round(d / 10, 6)
                        }
                    })
                except AttributeError:
                    return format_result({
                        "error": f"Material '{material}' not found in xrayutilities database",
                        "suggestion": "Use calculation_type='list_materials' to see available materials"
                    })

            elif lattice_a:
                # Calculate for cubic system with custom lattice
                d = lattice_a / np.sqrt(h**2 + k**2 + l**2)
                return format_result({
                    "tool": "xray_calculate",
                    "calculation": "d_from_hkl",
                    "inputs": {"h": h, "k": k, "l": l, "lattice_a": lattice_a},
                    "result": {"d_spacing_angstroms": round(d, 6)},
                    "note": "Cubic system assumed"
                })
            else:
                return format_result({"error": "Provide either 'material' or 'lattice_a'"})

        # D-SPACING FROM ANGLE (Bragg's law)
        elif calculation_type == "d_from_angle":
            if two_theta_degrees is None or wavelength_angstroms is None:
                return format_result({"error": "two_theta_degrees and wavelength_angstroms required"})

            theta_rad = np.radians(two_theta_degrees / 2.0)
            d = wavelength_angstroms / (2.0 * np.sin(theta_rad))

            return format_result({
                "tool": "xray_calculate",
                "library": "xrayutilities",
                "calculation": "d_from_angle",
                "formula": "d = λ / (2 * sin(θ))",
                "inputs": {
                    "two_theta_degrees": two_theta_degrees,
                    "wavelength_angstroms": wavelength_angstroms
                },
                "result": {"d_spacing_angstroms": round(d, 6)}
            })

        # ANGLE FROM D-SPACING
        elif calculation_type == "angle_from_d":
            if d_spacing is None or wavelength_angstroms is None:
                return format_result({"error": "d_spacing and wavelength_angstroms required"})

            sin_theta = wavelength_angstroms / (2.0 * d_spacing)
            if sin_theta > 1.0:
                return format_result({
                    "error": "No diffraction possible",
                    "reason": f"λ/(2d) = {sin_theta:.3f} > 1.0"
                })

            two_theta = 2.0 * np.degrees(np.arcsin(sin_theta))

            return format_result({
                "tool": "xray_calculate",
                "calculation": "angle_from_d",
                "formula": "2θ = 2 * arcsin(λ / (2d))",
                "inputs": {"d_spacing": d_spacing, "wavelength_angstroms": wavelength_angstroms},
                "result": {"two_theta_degrees": round(two_theta, 4)}
            })

        # ENERGY TO WAVELENGTH
        elif calculation_type == "energy_to_wavelength":
            if energy_kev is None:
                return format_result({"error": "energy_kev required"})

            # Use xrayutilities energy-wavelength conversion
            wavelength = xu.en2lam(energy_kev * 1000)  # Convert keV to eV

            return format_result({
                "tool": "xray_calculate",
                "library": "xrayutilities.en2lam()",
                "calculation": "energy_to_wavelength",
                "inputs": {"energy_kev": energy_kev},
                "result": {
                    "wavelength_angstroms": round(wavelength, 6),
                    "energy_eV": energy_kev * 1000
                }
            })

        # WAVELENGTH TO ENERGY
        elif calculation_type == "wavelength_to_energy":
            if wavelength_angstroms is None:
                return format_result({"error": "wavelength_angstroms required"})

            # Use xrayutilities wavelength-energy conversion
            energy_eV = xu.lam2en(wavelength_angstroms)
            energy_kev = energy_eV / 1000

            return format_result({
                "tool": "xray_calculate",
                "library": "xrayutilities.lam2en()",
                "calculation": "wavelength_to_energy",
                "inputs": {"wavelength_angstroms": wavelength_angstroms},
                "result": {
                    "energy_kev": round(energy_kev, 6),
                    "energy_eV": round(energy_eV, 2)
                }
            })

        # STRAIN CALCULATION
        elif calculation_type == "strain":
            if measured_d is None or reference_d is None:
                return format_result({"error": "measured_d and reference_d required"})

            strain = (measured_d - reference_d) / reference_d
            microstrain = strain * 1e6

            return format_result({
                "tool": "xray_calculate",
                "calculation": "strain",
                "formula": "ε = (d_measured - d_ref) / d_ref",
                "inputs": {"measured_d": measured_d, "reference_d": reference_d},
                "result": {
                    "strain": round(strain, 9),
                    "microstrain": round(microstrain, 2),
                    "percent": round(strain * 100, 4),
                    "type": "tension" if strain > 0 else "compression" if strain < 0 else "no strain"
                }
            })

        # LIST AVAILABLE MATERIALS
        elif calculation_type == "list_materials":
            # Get all materials from xrayutilities
            materials = [name for name in dir(xumaterials)
                        if not name.startswith('_') and name[0].isupper()]

            return format_result({
                "tool": "xray_calculate",
                "library": "xrayutilities.materials",
                "calculation": "list_materials",
                "available_materials": sorted(materials[:50]),  # First 50
                "total_count": len(materials),
                "note": "Use material name in d_from_hkl calculation"
            })

        else:
            return format_result({
                "error": f"Unknown calculation_type: {calculation_type}",
                "valid_types": ["d_from_hkl", "d_from_angle", "angle_from_d",
                               "energy_to_wavelength", "wavelength_to_energy",
                               "strain", "list_materials"]
            })

    except Exception as e:
        return format_result({"error": f"X-ray calculation error: {str(e)}"})

@mcp.tool()
async def validate_beamline_parameters(
    energy_kev: float,
    detector_distance_mm: float,
    beam_center_x: float,
    beam_center_y: float,
    pixel_size_um: float = None
) -> str:
    """Validate that beamline experimental parameters are physically reasonable.

    Checks:
    - Energy is in typical synchrotron range
    - Detector distance is reasonable
    - Beam center is within typical detector bounds
    - Pixel size is reasonable if provided

    Args:
        energy_kev: X-ray energy in keV
        detector_distance_mm: Sample-to-detector distance in millimeters
        beam_center_x: Beam center X coordinate in pixels
        beam_center_y: Beam center Y coordinate in pixels
        pixel_size_um: Detector pixel size in micrometers (optional)

    Returns:
        JSON with validation results and any warnings
    """
    try:
        issues = []
        warnings = []

        # Energy validation (typical range: 5-150 keV for synchrotrons)
        if energy_kev < 5 or energy_kev > 150:
            issues.append(f"Energy {energy_kev} keV is outside typical synchrotron range (5-150 keV)")
        elif energy_kev < 10 or energy_kev > 120:
            warnings.append(f"Energy {energy_kev} keV is unusual (typical: 10-120 keV)")

        # Detector distance (typical: 50-2000 mm for HEDM)
        if detector_distance_mm < 50 or detector_distance_mm > 2000:
            issues.append(f"Detector distance {detector_distance_mm} mm seems unusual (typical: 50-2000 mm)")

        # Beam center (typical detector: 2048x2048 pixels, but can be 4096x4096)
        if beam_center_x < 0 or beam_center_x > 5000:
            issues.append(f"Beam center X={beam_center_x} outside typical range (0-5000 pixels)")
        if beam_center_y < 0 or beam_center_y > 5000:
            issues.append(f"Beam center Y={beam_center_y} outside typical range (0-5000 pixels)")

        # Pixel size validation (typical: 50-200 μm)
        if pixel_size_um is not None:
            if pixel_size_um < 10 or pixel_size_um > 500:
                warnings.append(f"Pixel size {pixel_size_um} μm is unusual (typical: 50-200 μm)")

        # Calculate wavelength for reference
        wavelength = 12.398 / energy_kev

        result = {
            "tool": "validate_beamline_parameters",
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "parameters": {
                "energy_kev": energy_kev,
                "wavelength_angstroms": round(wavelength, 6),
                "detector_distance_mm": detector_distance_mm,
                "beam_center": [beam_center_x, beam_center_y]
            }
        }

        if pixel_size_um:
            result["parameters"]["pixel_size_um"] = pixel_size_um

        if len(issues) == 0 and len(warnings) == 0:
            result["message"] = "All parameters are within reasonable ranges"

        return format_result(result)

    except Exception as e:
        return format_result({"error": f"Error validating parameters: {str(e)}"})

@mcp.tool()
async def list_common_calibrants() -> str:
    """List commonly used calibration materials with their properties.

    Returns:
        JSON with standard calibrants (CeO2, LaB6, Si, etc.) and their lattice parameters
    """
    calibrants = {
        "CeO2": {
            "name": "Cerium Dioxide (Ceria)",
            "formula": "CeO2",
            "crystal_system": "Cubic",
            "space_group": "Fm-3m (225)",
            "lattice_parameter_a": 5.411,
            "common_use": "Standard calibrant for HEDM, powder diffraction",
            "major_peaks_hkl": ["111", "200", "220", "311", "222", "400", "331", "420"]
        },
        "LaB6": {
            "name": "Lanthanum Hexaboride",
            "formula": "LaB6",
            "crystal_system": "Cubic",
            "space_group": "Pm-3m (221)",
            "lattice_parameter_a": 4.156,
            "common_use": "High-angle calibration, instrumental broadening",
            "major_peaks_hkl": ["100", "110", "111", "200", "210", "211", "220"]
        },
        "Si": {
            "name": "Silicon",
            "formula": "Si",
            "crystal_system": "Cubic",
            "space_group": "Fd-3m (227)",
            "lattice_parameter_a": 5.431,
            "common_use": "NIST standard reference material (SRM 640d)",
            "major_peaks_hkl": ["111", "220", "311", "400", "331", "422", "511", "440"]
        },
        "Al2O3": {
            "name": "Corundum (Alumina)",
            "formula": "Al2O3",
            "crystal_system": "Hexagonal",
            "space_group": "R-3c (167)",
            "lattice_parameter_a": 4.759,
            "lattice_parameter_c": 12.991,
            "common_use": "High-temperature studies",
            "major_peaks_hkl": ["012", "104", "110", "113", "024", "116", "214"]
        },
        "CaF2": {
            "name": "Calcium Fluoride (Fluorite)",
            "formula": "CaF2",
            "crystal_system": "Cubic",
            "space_group": "Fm-3m (225)",
            "lattice_parameter_a": 5.463,
            "common_use": "Calibration standard",
            "major_peaks_hkl": ["111", "220", "311", "400", "331", "422", "511", "440"]
        }
    }

    return format_result({
        "tool": "list_common_calibrants",
        "count": len(calibrants),
        "calibrants": calibrants,
        "note": "Lattice parameters are at room temperature (298K)"
    })

# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not XRAYUTILITIES_AVAILABLE:
        print("⚠ xrayutilities not available — X-ray calculations limited", file=sys.stderr)
    if not SCIPY_AVAILABLE:
        print("⚠ scipy/fabio not available — image analysis limited", file=sys.stderr)
    mcp.run()
