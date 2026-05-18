"""APEXA MAUD/MILK Rietveld refinement driver — Phase 1 skeleton.

Mirrors the CLI surface of `apexa_gsas_robust.py` so the MCP dispatcher in
`midas_comprehensive_server.run_gsas_refinement(engine="maud"|"both")` can
shell out to either driver interchangeably.

Phase 1 scope (cell+microstructure parity with the GSAS-II driver):
    * NaN-safe per-η-slice extraction (reused verbatim from apexa_gsas_robust).
    * Calibrant-aware starting cell (reused verbatim from apexa_gsas_robust:
      CeO2/SRM 674b, LaB6/SRM 660c, Si/SRM 640f, Al2O3/SRM 676a).
    * Per-slice MAUD refinement via MILK, recipe Bkg+Scale → Cell → UVW → XY,SH/L.
    * Output schema matches `benchmark/detector_zoo/refine_v2.py`'s `summary.json`.

Phase 2 (separate PR) will add `--refine-texture` and `--refine-microstructure`
flags exposing the things only MAUD does well.

Phase 3 (separate PR) will add Spotlight as a global-optimization fallback.

This file is a SKELETON. The MILK API surface (project loading, parameter
toggling, batch refinement) is documented at
https://github.com/lanl/MILK and in J. Appl. Cryst. 56:1277 (2023). Filling
in the `_refine_one_milk` function requires MILK installed and a sample
MAUD `.par` template to reference. Until then:

    * `engine="maud"` returns a clean `engine_unavailable` from the MCP
      dispatcher when this file is on disk but MILK / MAUD are not.
    * `engine="both"` falls back to gsas2-only and tags the response with
      `fallback_used="gsas2"`.
    * Running this script directly with `--dry-run` writes a stub
      `summary.json` with the canonical schema so the dispatcher and
      benchmark scoring can be exercised end-to-end before the real
      refinement loop is wired up.

Run inside the apexa env (which carries `milk-rietveld` and a JDK on PATH).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Reuse engine-independent helpers from the GSAS-II driver. These functions
# (NaN-safe extraction, CIF prep, lineout-based cell estimation, calibrant
# detection) operate on the .zarr.zip and the CIF text — they do not touch
# GSAS-II — so they apply identically to MAUD.
from apexa_gsas_robust import (
    CALIBRANT_REFERENCES,
    _detect_calibrant,
    _read_cif_cell_a,
    estimate_lattice_from_lineout,
    extract_histograms_nan_safe,
    prepare_starting_cif,
)
from apexa_engines import EngineUnavailable, EngineResult, maud_install_hint, find_maud_installation


# ============================================================================
# SOFT IMPORT — MILK + JDK
# ============================================================================
def _import_milk():
    """Import MILK; raise EngineUnavailable cleanly if missing.

    MILK is the LANL Python wrapper around MAUD batch mode. It requires:
      * `pip install milk-rietveld` (PyPI), and
      * a working JDK on PATH (or JAVA_HOME pointing to one), and
      * MAUD itself, discoverable via apexa_engines.find_maud_installation().
    """
    if find_maud_installation() is None:
        raise EngineUnavailable("maud", maud_install_hint())
    try:
        import milk  # noqa: F401  (the actual symbols come later)
        return milk
    except ImportError as e:
        raise EngineUnavailable(
            "maud",
            f"MILK Python package not importable ({e}). " + maud_install_hint(),
        ) from e


# ============================================================================
# CORE — translate inputs to MAUD's parameter file form
# ============================================================================
def prepare_starting_par(
    cif_path: Path,
    out_dir: Path,
    *,
    experimental_a: Optional[float] = None,
    auto_calibrant: bool = True,
    estimated_a: Optional[float] = None,
) -> tuple[Path, dict]:
    """Build a MAUD `.par` parameter file with a calibrant-aware starting cell.

    Phase 1 strategy: reuse `prepare_starting_cif` from the GSAS-II driver to
    produce an experimentally-corrected CIF, then load that CIF into a MAUD
    `.par` template. (MAUD reads CIF directly via its CIF importer.) The
    cell-substitution policy and calibrant table thus stay in one place.

    Returns (par_path, info_dict) where info_dict has the same shape as the
    GSAS-II driver's `cif_preparation` entry so the summary schema is uniform.
    """
    corrected_cif, info = prepare_starting_cif(
        str(cif_path), out_dir,
        experimental_a=experimental_a,
        auto_calibrant=auto_calibrant,
        estimated_a=estimated_a,
    )
    # TODO(milk-api): instantiate a MAUD project from a .par template, swap
    # in the corrected CIF as the phase, and write the .par to out_dir.
    par_path = out_dir / (Path(corrected_cif).stem + ".par")
    info["par_file"] = str(par_path)
    info["status"] = "stub"  # remove when the .par writer lands
    return par_path, info


# ============================================================================
# PER-SLICE REFINEMENT (the core MILK work)
# ============================================================================
def _refine_one_milk(
    milk_module,
    slice_index: int,
    tth,
    intensity,
    sigma,
    par_path: Path,
    out_dir: Path,
    instprm_file: Optional[str],
    two_theta_limits: Optional[List[float]],
    bkg_terms: int,
    refine_atoms: bool,
) -> dict:
    """Refine a single η-slice through MILK → MAUD batch.

    PHASE 1 IMPLEMENTATION TARGET. The exact API call sequence depends on the
    installed MILK version; the high-level recipe should be:

        proj = milk.maud.MaudProject.from_par(par_path)
        proj.add_dataset(tth, intensity, sigma, instprm=instprm_file,
                         two_theta_limits=two_theta_limits)

        # Stage 1: Background + Scale
        proj.params.refine_only(["bkg", "scale"], bkg_terms=bkg_terms)
        proj.refine()

        # Stage 2: Cell
        proj.params.refine_add(["cell_a"])
        proj.refine()

        # Stage 3: Profile UVW
        proj.params.refine_add(["U", "V", "W"])
        proj.refine()

        # Stage 4: Profile XY + SH/L
        proj.params.refine_add(["X", "Y", "SH/L"])
        proj.refine()

        # Stage 5 (off by default): Atom positions
        if refine_atoms:
            proj.params.refine_add(["atoms.xu"])
            proj.refine()

        return {
            "slice_index": slice_index,
            "status": "success",
            "a":   proj.params.get("cell_a"),
            "Rwp": proj.last_rwp(),
            "engine": "maud",
        }

    Until the real call sequence is wired up, this raises NotImplementedError
    with a clear pointer; the dispatcher catches and returns
    `engine_unavailable` to the agent.
    """
    raise NotImplementedError(
        "MAUD/MILK per-slice refinement not yet implemented. "
        "Phase 1 wiring lands in a follow-up commit; until then the MCP "
        "dispatcher returns engine_unavailable for engine='maud' and falls "
        "back to gsas2 for engine='both'."
    )


# ============================================================================
# DRIVER
# ============================================================================
def run(
    data: str,
    cifs: List[str],
    out: str,
    *,
    instprm: Optional[str] = None,
    bkg_terms: int = 6,
    refine_atoms: bool = False,
    limits: Optional[List[float]] = None,
    wavelength_A: Optional[float] = None,
    experimental_a: Optional[float] = None,
    auto_calibrant: bool = True,
    reference_a_A: float = 5.41165,
    detector_tag: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """End-to-end MAUD refinement; mirrors apexa_gsas_robust.run signature.

    `dry_run=True` writes a stub `summary.json` with the canonical schema and
    no real refinement — used for end-to-end dispatcher testing before MILK
    is installed.
    """
    import numpy as np

    # 0. Fail fast on missing MAUD/MILK before doing any I/O so the caller
    # gets the structured engine_unavailable response, not a confusing
    # FileNotFoundError from zarr extraction.
    if not dry_run:
        _import_milk()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. NaN-safe extract — engine-independent, reused.
    hists, ext_stats = extract_histograms_nan_safe(data)
    if not hists:
        return {"status": "error", "engine": "maud",
                "error": "No usable histograms after NaN filter."}

    # 2. Estimate lattice from data (informational; informs CIF prep when CIF disagrees).
    estimated_a = None
    if wavelength_A is not None:
        estimated_a = estimate_lattice_from_lineout(hists, wavelength_A)

    # 3. Prepare CIFs / .par files — reuses calibrant table.
    prepared = []
    cif_infos = []
    for c in cifs:
        par, info = prepare_starting_par(
            Path(c), out_dir,
            experimental_a=experimental_a,
            auto_calibrant=auto_calibrant,
            estimated_a=estimated_a,
        )
        prepared.append(par)
        cif_infos.append(info)

    if dry_run:
        # Write a benchmark-schema-compatible stub. Useful for exercising
        # the MCP dispatcher and the cross-validation merge code without
        # MILK installed.
        stub = EngineResult(
            engine="maud",
            detector=detector_tag,
            n_total=len(hists),
            n_success=0,
            n_a_in_window=0,
            limits=limits,
            reference_a_A=reference_a_A,
            engine_diagnostics={
                "mode": "dry-run",
                "extraction_stats": ext_stats,
                "cif_preparation": cif_infos,
                "estimated_a_from_lineout": estimated_a,
                "note": "MILK not invoked; refinement loop is a Phase 1 TODO.",
            },
        ).to_dict()
        (out_dir / "summary.json").write_text(json.dumps(stub, indent=2))
        return stub

    # 4. Real MAUD refinement loop — requires MILK.
    milk = _import_milk()  # raises EngineUnavailable cleanly if not installed
    results = []
    for i, (tth, inten, sig) in enumerate(hists):
        r = _refine_one_milk(
            milk, i, tth, inten, sig,
            par_path=prepared[0],
            out_dir=out_dir,
            instprm_file=instprm,
            two_theta_limits=limits,
            bkg_terms=bkg_terms,
            refine_atoms=refine_atoms,
        )
        results.append(r)

    # 5. Aggregate to the canonical benchmark schema.
    a_vals = np.array([r["a"] for r in results
                       if r.get("status") == "success"
                       and r.get("a") is not None
                       and 4.5 < r["a"] < 6.5])
    rwps = np.array([r["Rwp"] for r in results
                     if r.get("status") == "success" and r.get("Rwp") is not None])

    summary = EngineResult(
        engine="maud",
        detector=detector_tag,
        n_total=len(hists),
        n_success=int(sum(1 for r in results if r.get("status") == "success")),
        n_a_in_window=int(len(a_vals)),
        median_Rwp=float(np.median(rwps)) if len(rwps) else None,
        mean_Rwp=float(np.mean(rwps)) if len(rwps) else None,
        mean_a=float(np.mean(a_vals)) if len(a_vals) else None,
        median_a=float(np.median(a_vals)) if len(a_vals) else None,
        std_a=float(np.std(a_vals)) if len(a_vals) else None,
        abs_da_mA=(float(abs(np.mean(a_vals) - reference_a_A) * 1000)
                   if len(a_vals) else None),
        reference_a_A=reference_a_A,
        limits=limits,
        engine_diagnostics={
            "extraction_stats": ext_stats,
            "cif_preparation": cif_infos,
            "estimated_a_from_lineout": estimated_a,
        },
    ).to_dict()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description="APEXA MAUD/MILK refinement driver (Phase 1 skeleton)")
    p.add_argument("--data", required=True, help="MIDAS .zarr.zip caked output")
    p.add_argument("--cif", nargs="+", required=True, help="CIF file(s)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--instprm", help="Instrument-parameter file (auto-converted to MAUD form)")
    p.add_argument("--bkg-terms", type=int, default=6)
    p.add_argument("--refine-atoms", action="store_true",
                   help="Enable atom-position stage (off by default — destabilizes low-stat data)")
    p.add_argument("--limits", nargs=2, type=float, metavar=("LO", "HI"),
                   help="2-theta limits in degrees")
    p.add_argument("--wavelength", type=float, dest="wavelength_A",
                   help="X-ray wavelength (Å); used to estimate starting cell from lineout")
    p.add_argument("--experimental-a", type=float,
                   help="Override CIF starting a (Å) — e.g., 5.41165 for NIST CeO2")
    p.add_argument("--no-auto-calibrant", action="store_true",
                   help="Disable automatic NIST-cell substitution for known calibrants")
    p.add_argument("--reference-a", type=float, default=5.41165, dest="reference_a_A",
                   help="Reference cell (Å) used to compute abs_da_mA (default: NIST CeO2)")
    p.add_argument("--detector-tag", help="Optional zoo detector tag for the summary")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip MILK invocation; write a stub summary.json with the canonical schema")
    args = p.parse_args()

    try:
        summary = run(
            data=args.data, cifs=args.cif, out=args.out,
            instprm=args.instprm, bkg_terms=args.bkg_terms,
            refine_atoms=args.refine_atoms, limits=args.limits,
            wavelength_A=args.wavelength_A,
            experimental_a=args.experimental_a,
            auto_calibrant=not args.no_auto_calibrant,
            reference_a_A=args.reference_a_A,
            detector_tag=args.detector_tag,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
    except EngineUnavailable as e:
        # Surface as a structured non-zero exit so the MCP dispatcher can
        # turn it into engine_unavailable rather than a generic crash.
        print(json.dumps({
            "status": "engine_unavailable",
            "engine": e.engine,
            "install_hint": e.install_hint,
        }, indent=2), file=sys.stderr)
        sys.exit(2)
    except NotImplementedError as e:
        print(json.dumps({
            "status": "engine_unavailable",
            "engine": "maud",
            "install_hint": str(e),
        }, indent=2), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
