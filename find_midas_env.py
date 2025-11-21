#!/usr/bin/env python3
"""
Diagnostic script to find MIDAS Python environment on beamline systems.
Run this on the beamline to identify where MIDAS dependencies are installed.
"""

import subprocess
import sys
from pathlib import Path

def check_python_has_deps(python_path):
    """Check if a Python interpreter has MIDAS dependencies."""
    deps = ["zarr", "diplib", "numba", "h5py", "skimage", "plotly"]
    try:
        cmd = [str(python_path), "-c", f"import {', '.join(deps)}"]
        result = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
        if result.returncode == 0:
            return True, deps
        else:
            # Try to find which deps are missing
            missing = []
            for dep in deps:
                result2 = subprocess.run(
                    [str(python_path), "-c", f"import {dep}"],
                    capture_output=True, timeout=5
                )
                if result2.returncode != 0:
                    missing.append(dep)
            return False, missing
    except Exception as e:
        return False, [f"Error: {e}"]

def main():
    print("="*70)
    print("MIDAS Python Environment Diagnostic Tool")
    print("="*70)
    print()

    # Current Python
    print(f"Current Python: {sys.executable}")
    has_deps, info = check_python_has_deps(sys.executable)
    if has_deps:
        print(f"  ✓ Has all MIDAS dependencies: {', '.join(info)}")
    else:
        print(f"  ✗ Missing dependencies: {', '.join(info)}")
    print()

    # Check conda environments
    print("Checking conda environments...")
    conda_locations = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / ".conda",
        Path("/opt/conda"),
        Path("/opt/miniconda3"),
        Path("/opt/anaconda3"),
    ]

    found_envs = []
    for conda_base in conda_locations:
        if not conda_base.exists():
            continue

        print(f"\n  Conda installation found: {conda_base}")

        # Check base environment
        base_python = conda_base / "bin" / "python"
        if base_python.exists():
            has_deps, info = check_python_has_deps(base_python)
            status = "✓" if has_deps else "✗"
            print(f"    {status} Base environment: {base_python}")
            if has_deps:
                found_envs.append(str(base_python))
            else:
                print(f"       Missing: {', '.join(info)}")

        # Check envs directory
        envs_dir = conda_base / "envs"
        if envs_dir.exists():
            for env_dir in sorted(envs_dir.iterdir()):
                if env_dir.is_dir():
                    env_python = env_dir / "bin" / "python"
                    if env_python.exists():
                        has_deps, info = check_python_has_deps(env_python)
                        status = "✓" if has_deps else "✗"
                        print(f"    {status} Environment '{env_dir.name}': {env_python}")
                        if has_deps:
                            found_envs.append(str(env_python))
                        elif isinstance(info, list) and info:
                            print(f"       Missing: {', '.join(info[:3])}...")  # Show first 3

    # Check system Python
    print("\n\nChecking system Python...")
    import shutil
    system_python = shutil.which("python3")
    if system_python:
        has_deps, info = check_python_has_deps(system_python)
        status = "✓" if has_deps else "✗"
        print(f"  {status} System python3: {system_python}")
        if has_deps:
            found_envs.append(system_python)
        else:
            print(f"     Missing: {', '.join(info)}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if found_envs:
        print(f"\n✓ Found {len(found_envs)} Python environment(s) with MIDAS dependencies:")
        for i, env in enumerate(found_envs, 1):
            print(f"  {i}. {env}")
        print("\nRecommendation: Use the first one found above.")
        print("\nTo configure APEXA, you can either:")
        print("  1. Do nothing - APEXA will auto-detect these environments")
        print("  2. Set MIDAS_PYTHON environment variable:")
        print(f"     export MIDAS_PYTHON={found_envs[0]}")
    else:
        print("\n✗ No Python environments found with complete MIDAS dependencies!")
        print("\nTo fix this, install MIDAS dependencies in one of these ways:")
        print("\nOption 1 - Create dedicated conda environment (RECOMMENDED):")
        print("  conda create -n midas_env python=3.10")
        print("  conda activate midas_env")
        print("  pip install zarr diplib numba h5py scikit-image plotly pandas pillow scipy")
        print("\nOption 2 - Install in current environment:")
        print("  pip install zarr diplib numba h5py scikit-image plotly pandas pillow scipy")
        print("\nOption 3 - Use uv in the beamline assistant venv:")
        print("  cd APS-Beamline-Assistant")
        print("  uv pip install zarr diplib numba h5py scikit-image plotly pandas pillow scipy")

    print()

if __name__ == "__main__":
    main()
