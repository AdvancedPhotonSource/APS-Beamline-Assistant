#!/usr/bin/env python3
"""Cross-engine scoring report for the detector zoo.

Reads each detector's two `summary.json` files (gsas2 + maud) and the
`refinement_crossvalidation.json` written by `run_gsas_refinement(engine="both")`,
then emits a Markdown table comparing the two engines against the NIST CeO2
reference (a = 5.41165 Å).

Layout expected per detector:
    benchmark/detector_zoo/<det>/refinement_v2/summary.json          # gsas2
    benchmark/detector_zoo/<det>/refinement_v2_maud/summary.json     # maud
    benchmark/detector_zoo/<det>/refinement_v2_both/refinement_crossvalidation.json

Detectors with one or both files missing are listed in the Markdown report
as "missing" rows so the gap is visible rather than silently dropped.

This is the artefact that backs NMI §4.6 — the load-bearing claim that the
0.07 mÅ NIST agreement is not an artefact of a single Rietveld engine.

Run: `uv run python benchmark/detector_zoo/score_crossengine.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

NIST_CEO2_A = 5.41165   # Å, SRM 674b
DEFAULT_DETECTORS = ["ge", "pilatus", "varex_aero", "varex_distortion"]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "—" if x is None else format(x, spec)


def collect(zoo_root: Path, detectors: list[str]) -> list[dict]:
    rows = []
    for det in detectors:
        det_dir = zoo_root / det
        gsas2 = _load_json(det_dir / "refinement_v2" / "summary.json")
        maud = _load_json(det_dir / "refinement_v2_maud" / "summary.json")
        cv = _load_json(det_dir / "refinement_v2_both" / "refinement_crossvalidation.json")

        row = {"detector": det, "gsas2": gsas2, "maud": maud, "crossval": cv}
        rows.append(row)
    return rows


def render_markdown(rows: list[dict], cell_tol_mA: float = 1.0) -> str:
    out: list[str] = []
    out.append("# Cross-engine validation — detector zoo")
    out.append("")
    out.append(f"Reference: NIST SRM 674b CeO2, a = {NIST_CEO2_A} Å.")
    out.append(f"Verdict threshold: |Δa(GSAS-II − MAUD)| < {cell_tol_mA:.1f} mÅ.")
    out.append("")
    out.append(
        "| Detector | n_succ (g/m) | a_GSAS-II (Å) | |Δa_NIST g| (mÅ) | "
        "a_MAUD (Å) | |Δa_NIST m| (mÅ) | |Δa(g−m)| (mÅ) | Rwp_g | Rwp_m | Verdict |"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )

    n_ok = 0
    n_disagree = 0
    n_missing = 0

    for row in rows:
        det = row["detector"]
        g, m, cv = row["gsas2"], row["maud"], row["crossval"]
        if g is None or m is None:
            n_missing += 1
            missing = []
            if g is None: missing.append("gsas2")
            if m is None: missing.append("maud")
            out.append(
                f"| {det} | — | — | — | — | — | — | — | — | "
                f"missing ({', '.join(missing)}) |"
            )
            continue

        delta = (cv or {}).get("delta_a_mA")
        if delta is None and g.get("mean_a") is not None and m.get("mean_a") is not None:
            delta = abs(g["mean_a"] - m["mean_a"]) * 1000.0

        verdict_raw = (cv or {}).get("verdict")
        if verdict_raw is None:
            verdict_raw = "ok" if (delta is not None and delta < cell_tol_mA) else "disagree"
        if verdict_raw == "ok":
            n_ok += 1
        elif verdict_raw == "disagree":
            n_disagree += 1

        out.append(
            f"| {det} "
            f"| {g.get('n_success', '—')}/{m.get('n_success', '—')} "
            f"| {_fmt(g.get('mean_a'), '.5f')} "
            f"| {_fmt(g.get('abs_da_mA'), '.3f')} "
            f"| {_fmt(m.get('mean_a'), '.5f')} "
            f"| {_fmt(m.get('abs_da_mA'), '.3f')} "
            f"| {_fmt(delta, '.3f')} "
            f"| {_fmt(g.get('median_Rwp'), '.2f')} "
            f"| {_fmt(m.get('median_Rwp'), '.2f')} "
            f"| {verdict_raw} |"
        )

    out.append("")
    out.append(
        f"**Summary:** ok={n_ok}, disagree={n_disagree}, missing={n_missing} "
        f"(of {len(rows)} detectors)."
    )
    return "\n".join(out) + "\n"


def main() -> int:
    zoo_root = Path(__file__).resolve().parent
    out_path = zoo_root / "crossvalidation_report.md"
    rows = collect(zoo_root, DEFAULT_DETECTORS)
    md = render_markdown(rows)
    out_path.write_text(md)
    print(md)
    print(f"Wrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
