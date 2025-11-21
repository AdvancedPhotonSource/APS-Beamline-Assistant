#!/usr/bin/env python3
"""
Analysis Utilities MCP Server
Custom analysis tools for quick diagnostics and post-processing

⚠️ IMPORTANT: These tools are NOT official MIDAS tools
For official MIDAS workflows, use midas_comprehensive_server.py

Author: Beamline Assistant Team
Organization: Argonne National Laboratory
"""

from typing import Any, Optional, List, Dict
import json
import sys
import os
from pathlib import Path
import numpy as np
import logging
from mcp.server.fastmcp import FastMCP

# Suppress verbose MCP server logging
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# =============================================================================
# INITIALIZATION
# =============================================================================

mcp = FastMCP("Analysis Utilities")

# Try to import optional dependencies
try:
    import fabio
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some features limited", file=sys.stderr)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_result(result: dict) -> str:
    """Format analysis results into readable JSON string."""
    return json.dumps(result, indent=2)

def load_diffraction_image(image_path: str):
    """Load diffraction image using fabio."""
    try:
        if SCIPY_AVAILABLE:
            img = fabio.open(image_path)
            return img.data.astype(np.float64)
        else:
            return np.random.rand(2048, 2048) * 1000
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {e}")

# =============================================================================
# CUSTOM DIAGNOSTIC TOOLS (NOT MIDAS)
# =============================================================================

@mcp.tool()
async def detect_rings_quick(
    image_path: str,
    detector_distance: float = 1000.0,
    wavelength: float = 0.2066,
    beam_center_x: float = None,
    beam_center_y: float = None
) -> str:
    """⚠️ QUICK DIAGNOSTIC TOOL - NOT for calibration

    This is a custom NumPy-based tool for QUICK diffraction ring detection.
    This is NOT the official MIDAS calibration method.

    ❌ DO NOT USE FOR:
    - Detector calibration (use midas_auto_calibrate from midas_comprehensive_server instead)
    - Production FF-HEDM analysis (use MIDAS native tools)
    - Precise geometric refinement

    ✅ USE THIS FOR:
    - Quick sanity check of diffraction image quality
    - Fast ring counting before running full MIDAS pipeline
    - Educational/debugging purposes
    - Verify data quality before beamtime

    For detector calibration: Use midas_auto_calibrate from midas_comprehensive_server

    Args:
        image_path: Path to the 2D diffraction image file
        detector_distance: Sample-to-detector distance in millimeters (for 2theta calculation)
        wavelength: X-ray wavelength in Angstroms (for 2theta calculation)
        beam_center_x: Beam center X coordinate in pixels (default: image center)
        beam_center_y: Beam center Y coordinate in pixels (default: image center)

    Returns:
        JSON with detected ring positions and quality metrics

    Example:
        Quick check before calibration:
        detect_rings_quick("/data/CeO2.tif", detector_distance=650.0, wavelength=0.2021)
    """
    try:
        if not SCIPY_AVAILABLE:
            return format_result({
                "tool": "detect_rings_quick",
                "status": "error",
                "error": "scipy not available - install scipy for this feature"
            })

        if not Path(image_path).exists():
            return format_result({
                "tool": "detect_rings_quick",
                "status": "error",
                "error": f"Image file not found: {image_path}"
            })

        image_data = load_diffraction_image(image_path)

        if beam_center_x is None or beam_center_y is None:
            center = (image_data.shape[0] // 2, image_data.shape[1] // 2)
        else:
            center = (int(beam_center_y), int(beam_center_x))

        # Radial profile (simple NumPy implementation)
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r_int = r.astype(int)
        radial_prof = np.bincount(r_int.ravel(), image_data.ravel())
        radial_counts = np.bincount(r_int.ravel())

        valid_idx = radial_counts > 0
        radial_prof = radial_prof[valid_idx] / radial_counts[valid_idx]
        r_values = np.arange(len(radial_prof))[valid_idx]

        # Find peaks using scipy
        ring_radii = []
        ring_intensities = []
        if len(radial_prof) > 10:
            peaks, properties = find_peaks(radial_prof, height=np.mean(radial_prof) * 1.2, distance=10)
            ring_radii = r_values[peaks]
            ring_intensities = radial_prof[peaks]

        # Convert to 2theta (rough estimate)
        pixel_size = 172e-6  # Typical GE detector pixel size in meters
        ring_2theta = np.arctan(np.array(ring_radii) * pixel_size / (detector_distance * 1e-3)) * 180 / np.pi

        signal_to_noise = np.mean(ring_intensities) / np.std(image_data) if len(ring_intensities) > 0 else 0

        return format_result({
            "tool": "detect_rings_quick",
            "method": "Custom NumPy/SciPy (NOT MIDAS)",
            "image_file": image_path,
            "rings_detected": len(ring_radii),
            "ring_positions_2theta": ring_2theta.tolist() if len(ring_2theta) > 0 else [],
            "ring_radii_pixels": ring_radii.tolist() if len(ring_radii) > 0 else [],
            "quality_metrics": {
                "signal_to_noise": float(signal_to_noise),
                "mean_intensity": float(np.mean(ring_intensities)) if len(ring_intensities) > 0 else 0
            },
            "warning": "This is a diagnostic tool. For calibration, use midas_auto_calibrate.",
            "status": "success"
        })

    except Exception as e:
        return format_result({
            "tool": "detect_rings_quick",
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
async def identify_phases_basic(
    peak_positions: list,
    material_system: str = "unknown",
    temperature: float = 25.0,
    tolerance: float = 0.1
) -> str:
    """⚠️ BASIC PHASE IDENTIFICATION - NOT comprehensive

    Simple phase matching using a hardcoded database of common phases.
    This is NOT a comprehensive phase identification tool like GSAS-II or Match!

    ❌ DO NOT USE FOR:
    - Definitive phase identification (use GSAS-II, Match!, or PDF databases)
    - Publication-quality results
    - Complex multi-phase systems
    - Unknown materials

    ✅ USE THIS FOR:
    - Quick screening of common Fe-based phases
    - Educational purposes
    - Initial hypothesis generation
    - Sanity check after integration

    For comprehensive phase analysis: Use GSAS-II server (gsas2_server.py)

    Args:
        peak_positions: List of peak positions in degrees 2theta
        material_system: Expected material system (currently only "Fe" phases supported)
        temperature: Sample temperature in Celsius (not currently used)
        tolerance: Peak position tolerance in degrees 2theta

    Returns:
        JSON with matched phases from limited database

    Example:
        identify_phases_basic([31.5, 44.8, 65.1], material_system="Fe", tolerance=0.2)
    """
    try:
        # Hardcoded database (very limited!)
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
            },
            "martensite": {
                "formula": "α'-Fe",
                "space_group": "Im-3m",
                "peaks": [31.5, 44.8, 65.2, 82.5],
                "intensities": [100, 80, 60, 40],
                "hkl": ["(110)", "(200)", "(211)", "(220)"]
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
                            "hkl": phase_data["hkl"][i],
                            "delta": float(abs(obs_peak - ref_peak))
                        })
                        break

            # Require at least 3 matched peaks
            if len(matched_peaks) >= 3:
                confidence = len(matched_peaks) / len(phase_data["peaks"])
                identified_phases.append({
                    "phase_name": phase_name.title(),
                    "chemical_formula": phase_data["formula"],
                    "space_group": phase_data["space_group"],
                    "matched_peaks": matched_peaks,
                    "confidence": round(confidence, 2),
                    "total_expected_peaks": len(phase_data["peaks"])
                })

        return format_result({
            "tool": "identify_phases_basic",
            "method": "Hardcoded database (NOT comprehensive)",
            "identified_phases": identified_phases,
            "total_phases_found": len(identified_phases),
            "database_size": len(phase_database),
            "warning": "This is a basic screening tool. Use GSAS-II or Match! for definitive identification.",
            "recommendation": "For comprehensive phase analysis, use GSAS-II server or commercial software",
            "status": "success" if len(identified_phases) > 0 else "no_matches"
        })

    except Exception as e:
        return format_result({
            "tool": "identify_phases_basic",
            "status": "error",
            "error": str(e)
        })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting Analysis Utilities MCP Server...", file=sys.stderr)
    print("⚠️ These are custom diagnostic tools, NOT official MIDAS", file=sys.stderr)
    print("For MIDAS workflows, use midas_comprehensive_server.py", file=sys.stderr)
    mcp.run()
