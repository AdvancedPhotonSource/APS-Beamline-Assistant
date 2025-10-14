#!/usr/bin/env python3
"""
Quick validation script to check MIDAS installation paths
"""

from pathlib import Path
import sys

MIDAS_ROOT = Path("/Users/b324240/opt/MIDAS")

print("=" * 70)
print("MIDAS Installation Validation")
print("=" * 70)

# Check root directory
print(f"\n1. MIDAS Root: {MIDAS_ROOT}")
print(f"   Exists: {'✓' if MIDAS_ROOT.exists() else '✗'}")

# Check key directories
dirs_to_check = {
    "Build bin": MIDAS_ROOT / "build" / "bin",
    "FF-HEDM bin": MIDAS_ROOT / "FF_HEDM" / "bin",
    "NF-HEDM bin": MIDAS_ROOT / "NF_HEDM" / "bin",
    "FF-HEDM v7": MIDAS_ROOT / "FF_HEDM" / "v7",
    "NF-HEDM v7": MIDAS_ROOT / "NF_HEDM" / "v7",
    "Utils": MIDAS_ROOT / "utils"
}

print("\n2. Directory Structure:")
for name, path in dirs_to_check.items():
    exists = path.exists()
    print(f"   {name:20} {'✓' if exists else '✗'} {path}")
    if exists and "bin" in name:
        # Count executables
        exes = list(path.glob("*"))
        exes = [e for e in exes if e.is_file() and not e.name.startswith('.')]
        print(f"   {'':20}   ({len(exes)} files)")

# Check for key executables
print("\n3. Key Executables:")
key_exes = [
    ("IndexerOMP", "FF-HEDM indexing"),
    ("FitPosOrStrainsOMP", "FF-HEDM refinement"),
    ("FitOrientationOMP", "NF-HEDM fitting"),
    ("GetHKLListNF", "NF-HEDM HKL generation"),
    ("CalibrantOMP", "Detector calibration")
]

for exe_name, description in key_exes:
    found = False
    location = None

    for bin_dir in [MIDAS_ROOT / "build" / "bin",
                    MIDAS_ROOT / "FF_HEDM" / "bin",
                    MIDAS_ROOT / "NF_HEDM" / "bin"]:
        exe_path = bin_dir / exe_name
        if exe_path.exists():
            found = True
            location = exe_path
            break

    status = "✓" if found else "✗"
    print(f"   {exe_name:30} {status}")
    if location:
        print(f"   {'':30}   → {location}")

# Check Python scripts
print("\n4. Python Workflow Scripts:")
python_scripts = [
    (MIDAS_ROOT / "FF_HEDM" / "v7" / "ff_MIDAS.py", "FF-HEDM workflow"),
    (MIDAS_ROOT / "NF_HEDM" / "v7" / "nf_MIDAS.py", "NF-HEDM workflow"),
    (MIDAS_ROOT / "FF_HEDM" / "v7" / "pf_MIDAS.py", "PF-HEDM workflow")
]

for script_path, description in python_scripts:
    exists = script_path.exists()
    print(f"   {script_path.name:30} {'✓' if exists else '✗'}")
    if exists:
        print(f"   {'':30}   {description}")

# Check utilities
print("\n5. Utility Scripts:")
utils_dir = MIDAS_ROOT / "utils"
if utils_dir.exists():
    util_scripts = list(utils_dir.glob("*.py"))
    print(f"   Found {len(util_scripts)} Python utility scripts")

    key_utils = ["GE2Tiff.py", "calcMiso.py", "SpotMatrixToSpotsHDF.py"]
    for util in key_utils:
        util_path = utils_dir / util
        print(f"   {util:30} {'✓' if util_path.exists() else '✗'}")

print("\n" + "=" * 70)

# Summary
all_good = all(path.exists() for path in dirs_to_check.values())
if all_good:
    print("✓ MIDAS installation looks good!")
    print("\nYou can now run:")
    print("  uv run argo_mcp_client.py midas:midas_comprehensive_server.py \\")
    print("    executor:command_executor_server.py filesystem:filesystem_server.py")
else:
    print("⚠ Some directories are missing. You may need to:")
    print("  1. Build MIDAS: cd /Users/b324240/opt/MIDAS && ./build.sh")
    print("  2. Check installation path is correct")

print("=" * 70)
