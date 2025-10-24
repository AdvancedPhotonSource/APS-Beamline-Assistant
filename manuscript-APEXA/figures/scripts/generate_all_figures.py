#!/usr/bin/env python3
"""
Master script to generate all manuscript figures for APEXA paper
Runs all individual figure generation scripts in sequence
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and report status"""
    script_path = Path(__file__).parent / script_name

    print(f"\n{'='*60}")
    print(f"Generating {script_name}...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with error:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Generate all figures"""
    print("\n" + "="*60)
    print("APEXA Manuscript Figure Generation")
    print("="*60)

    scripts = [
        'generate_fig1.py',  # System architecture
        'generate_fig2.py',  # Calibration performance
        'generate_fig3.py',  # Integration and phase ID
        'generate_fig4.py',  # User study results
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    # Summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)

    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{script:30s} {status}")

    # Overall status
    all_success = all(results.values())

    if all_success:
        print("\n" + "="*60)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("="*60)
        print("\nFigures saved to:")
        figures_dir = Path(__file__).parent.parent
        print(f"  {figures_dir}/")
        print("\nGenerated files:")
        for fig_num in range(1, 5):
            pdf_file = figures_dir / f"fig{fig_num}_*.pdf"
            print(f"  - fig{fig_num}_*.pdf (main figure)")
            print(f"  - fig{fig_num}_*.png (preview)")
        print("\nReady for LaTeX inclusion in manuscript!")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ SOME FIGURES FAILED TO GENERATE")
        print("="*60)
        print("\nPlease check error messages above and fix issues.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
