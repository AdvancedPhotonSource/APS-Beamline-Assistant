#!/usr/bin/env python3
"""
APEXA benchmark summary — scan a benchmark tree for per-run calibration outcome
manifests (APEXA_calibration.json) and emit a comparison table + CSV.

Generic: works for any dataset/att-sweep. Each `midas_auto_calibrate` run drops
an `APEXA_calibration.json`; this reads them all and tabulates the geometry so
you get a paper-ready cross-calibrant / cross-attenuation summary.

USAGE
-----
    python benchmark_summary.py <benchmark_dir> [--csv out.csv]

    # e.g. on the beamline:
    python benchmark_summary.py \
        /scratch/s1iduser/APEXA/startup_d_feb26/ai_tune/APEXA_benchmark

Prints a table (calibrant, att, engine, Lsd, BC, tilts, strain, converged,
integrated-output present?) and, with --csv, writes the same rows to CSV.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path


def _att_token(s: str):
    m = re.search(r"att(\d+)", s or "", re.IGNORECASE)
    return f"att{m.group(1)}" if m else ""


def _find_integrated(out_dir: Path, benchmark_root: Path) -> str:
    """Best-effort: does an integration output (.zarr.zip) exist for this run?
    Checks the calibration out_dir's sibling integration/<name>/ and the whole
    tree for a zarr matching the calibrant stem."""
    name = out_dir.name  # e.g. ceria_att3
    cand = benchmark_root / "integration" / name
    if cand.is_dir() and any(cand.glob("*.zarr.zip")):
        return str(next(cand.glob("*.zarr.zip")))
    return ""


def collect(benchmark_dir: Path):
    rows = []
    for mf in sorted(benchmark_dir.rglob("APEXA_calibration.json")):
        try:
            d = json.loads(mf.read_text())
        except Exception as e:
            print(f"  ⚠ skip {mf}: {e}", file=sys.stderr)
            continue
        cp = d.get("calibrated_parameters") or {}
        cm = d.get("convergence_metrics") or {}
        img = d.get("image_file", "")
        out_dir = Path(d.get("output_dir", mf.parent))
        lsd = cp.get("lsd")
        lsd_mm = (lsd / 1000.0) if (lsd and lsd > 1000) else lsd
        rows.append({
            "calibrant": d.get("calibrant", ""),
            "att": _att_token(Path(img).stem) or _att_token(out_dir.name),
            "status": d.get("status", ""),
            "engine": d.get("engine", ""),
            "Lsd_mm": round(lsd_mm, 3) if isinstance(lsd_mm, (int, float)) else "",
            "BC_X": cp.get("bc_x", ""),
            "BC_Y": cp.get("bc_y", ""),
            "tx": cp.get("tx", ""),
            "ty": cp.get("ty", ""),
            "tz": cp.get("tz", ""),
            "strain": cm.get("final_mean_strain", ""),
            "converged": cm.get("converged", ""),
            "integrated": "yes" if _find_integrated(out_dir, benchmark_dir) else "no",
            "manifest": str(mf),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="APEXA calibration benchmark summary")
    ap.add_argument("benchmark_dir", help="Path to an APEXA_benchmark/ tree")
    ap.add_argument("--csv", help="Also write rows to this CSV path")
    args = ap.parse_args()

    root = Path(args.benchmark_dir).expanduser()
    if not root.exists():
        sys.exit(f"benchmark_dir not found: {root}")

    rows = collect(root)
    if not rows:
        sys.exit(f"No APEXA_calibration.json manifests found under {root}")

    cols = ["calibrant", "att", "status", "engine", "Lsd_mm",
            "BC_X", "BC_Y", "ty", "tz", "strain", "converged", "integrated"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    # Cross-run geometry spread (a headline validation number)
    lsds = [r["Lsd_mm"] for r in rows if isinstance(r["Lsd_mm"], (int, float))]
    bxs = [r["BC_X"] for r in rows if isinstance(r["BC_X"], (int, float))]
    bys = [r["BC_Y"] for r in rows if isinstance(r["BC_Y"], (int, float))]
    if len(lsds) > 1:
        print(f"\nGeometry spread across {len(rows)} runs:")
        print(f"  Lsd:  {min(lsds):.3f}–{max(lsds):.3f} mm  (Δ {max(lsds)-min(lsds):.3f} mm)")
        if len(bxs) > 1:
            print(f"  BC_X: Δ {max(bxs)-min(bxs):.4f} px")
        if len(bys) > 1:
            print(f"  BC_Y: Δ {max(bys)-min(bys):.4f} px")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
