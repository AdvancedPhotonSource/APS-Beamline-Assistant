#!/usr/bin/env python3
"""
Direct MIDAS validation tool - bypasses Argo Gateway
"""
import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the validation tool
from midas_comprehensive_server import validate_midas_installation

async def main():
    print("=" * 70)
    print("MIDAS Installation Validation (Direct Tool Call)")
    print("=" * 70)
    print()

    # Call the tool directly
    result = await validate_midas_installation(
        midas_path="/Users/b324240/opt/MIDAS"
    )

    # Parse and display results
    import json
    try:
        data = json.loads(result)

        print(f"Tool: {data['tool']}")
        print(f"\nValidation Results:")
        print(f"  MIDAS Root: {data['validation']['midas_root']}")
        print(f"  Root Exists: {'✓' if data['validation']['root_exists'] else '✗'}")
        print(f"  Bin Directory: {'✓' if data['validation']['bin_exists'] else '✗'}")
        print()

        print(f"Executables Found:")
        exe_found = sum(data['validation']['executables'].values())
        exe_total = len(data['validation']['executables'])
        print(f"  {exe_found}/{exe_total} key executables")

        print(f"\nPython Modules:")
        scripts_found = sum(data['validation']['python_modules'].values())
        scripts_total = len(data['validation']['python_modules'])
        print(f"  {scripts_found}/{scripts_total} workflow scripts")

        print(f"\nDependencies:")
        deps_found = sum(data['validation']['dependencies'].values())
        deps_total = len(data['validation']['dependencies'])
        print(f"  {deps_found}/{deps_total} Python packages")

        print(f"\nOverall Status: {data['validation']['overall_status'].upper()}")

        if 'recommendations' in data['validation'] and data['validation']['recommendations']:
            print(f"\nRecommendations:")
            for rec in data['validation']['recommendations']:
                print(f"  • {rec}")

    except json.JSONDecodeError:
        # If JSON parsing fails, just print the raw result
        print(result)

    print()
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
