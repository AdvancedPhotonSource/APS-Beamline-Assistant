#!/usr/bin/env python3
"""
MAUD MCP Server
Provides texture analysis and quantitative phase analysis tools

MAUD: Materials Analysis Using Diffraction
- Texture analysis (ODF, pole figures)
- Quantitative phase analysis
- Microstructure analysis
- Stress/strain analysis
- Combined analysis (Rietveld + texture)

Author: Beamline Assistant Team
Organization: Argonne National Laboratory
"""

from typing import Any, Optional, List, Dict
import json
import sys
import os
from pathlib import Path
import subprocess
import logging
from mcp.server.fastmcp import FastMCP

# Suppress verbose MCP server logging
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================

def find_maud_installation() -> Optional[Path]:
    """Find MAUD installation by checking common locations.

    Priority order:
    1. MAUD_PATH environment variable
    2. ~/MAUD (home directory)
    3. ~/.local/MAUD
    4. /opt/MAUD (system-wide)
    5. /Applications/Maud.app (macOS)
    """
    if "MAUD_PATH" in os.environ:
        maud_path = Path(os.environ["MAUD_PATH"]).expanduser().absolute()
        if maud_path.exists():
            return maud_path

    common_paths = [
        Path.home() / "MAUD",
        Path.home() / ".local" / "MAUD",
        Path("/opt/MAUD"),
        Path("/Applications/Maud.app/Contents/Resources")
    ]

    for path in common_paths:
        if path.exists() and path.is_dir():
            print(f"Found MAUD installation at: {path}", file=sys.stderr)
            return path

    print(f"WARNING: MAUD not found. Install from http://maud.radiographema.eu/", file=sys.stderr)
    return None

MAUD_ROOT = find_maud_installation()
MAUD_AVAILABLE = MAUD_ROOT is not None

# Initialize FastMCP server
mcp = FastMCP("maud-analysis")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_result(data: dict) -> str:
    """Format result as JSON string"""
    return json.dumps(data, indent=2, default=str)

# =============================================================================
# MAUD ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
async def check_maud_installation() -> str:
    """Check if MAUD is installed and available.

    Returns:
        Installation status and version information
    """
    try:
        if not MAUD_AVAILABLE:
            return format_result({
                "tool": "check_maud_installation",
                "status": "not_found",
                "available": False,
                "message": "MAUD not found. Please install from http://maud.radiographema.eu/",
                "installation_instructions": {
                    "website": "http://maud.radiographema.eu/",
                    "download": "Download latest version for your OS",
                    "environment_variable": "Set MAUD_PATH=/path/to/MAUD"
                }
            })

        return format_result({
            "tool": "check_maud_installation",
            "status": "installed",
            "available": True,
            "installation_path": str(MAUD_ROOT),
            "message": "MAUD is installed and ready to use"
        })

    except Exception as e:
        return format_result({
            "tool": "check_maud_installation",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def run_texture_analysis(
    par_file: str,
    data_files: List[str],
    analysis_type: str = "ODF",
    pole_figures: Optional[List[str]] = None
) -> str:
    """Run texture analysis on diffraction data.

    PLACEHOLDER: This is a template for future MAUD integration.

    Args:
        par_file: Path to MAUD parameter file (.par)
        data_files: List of diffraction data files
        analysis_type: Type of analysis (ODF, pole_figures, combined)
        pole_figures: List of pole figures to calculate (e.g., ["111", "200"])

    Returns:
        Texture analysis results
    """
    try:
        if not MAUD_AVAILABLE:
            return format_result({
                "tool": "run_texture_analysis",
                "status": "unavailable",
                "error": "MAUD not installed"
            })

        # TODO: Implement MAUD command-line interface
        # MAUD can be run in batch mode via command line

        return format_result({
            "tool": "run_texture_analysis",
            "status": "not_implemented",
            "message": "MAUD integration coming soon",
            "note": "This is a placeholder for future development"
        })

    except Exception as e:
        return format_result({
            "tool": "run_texture_analysis",
            "status": "error",
            "error": str(e)
        })

@mcp.tool()
async def run_quantitative_phase_analysis(
    par_file: str,
    data_files: List[str],
    phases: List[str]
) -> str:
    """Run quantitative phase analysis on powder diffraction data.

    PLACEHOLDER: This is a template for future MAUD integration.

    Args:
        par_file: Path to MAUD parameter file (.par)
        data_files: List of diffraction data files
        phases: List of phase names to quantify

    Returns:
        Phase fractions and analysis results
    """
    try:
        if not MAUD_AVAILABLE:
            return format_result({
                "tool": "run_quantitative_phase_analysis",
                "status": "unavailable",
                "error": "MAUD not installed"
            })

        # TODO: Implement MAUD quantitative phase analysis

        return format_result({
            "tool": "run_quantitative_phase_analysis",
            "status": "not_implemented",
            "message": "MAUD integration coming soon",
            "note": "This is a placeholder for future development"
        })

    except Exception as e:
        return format_result({
            "tool": "run_quantitative_phase_analysis",
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    print("=" * 70, file=sys.stderr)
    print("MAUD MCP Server", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"MAUD Available: {MAUD_AVAILABLE}", file=sys.stderr)
    if MAUD_ROOT:
        print(f"MAUD Path: {MAUD_ROOT}", file=sys.stderr)
    print("\nAvailable Tools:", file=sys.stderr)
    print("  - check_maud_installation", file=sys.stderr)
    print("  - run_texture_analysis (placeholder)", file=sys.stderr)
    print("  - run_quantitative_phase_analysis (placeholder)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    mcp.run(transport='stdio')
