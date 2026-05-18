"""Shared types and helpers for APEXA Rietveld-engine drivers.

Two engines are pluggable through the same MCP tool surface
(`run_gsas_refinement(..., engine="gsas2"|"maud"|"both")`):

  * `apexa_gsas_robust.py` — GSAS-II via GSASIIscriptable
  * `apexa_maud_milk.py`   — MAUD via MILK (LANL Python wrapper)

This module owns:
  - the canonical per-engine summary schema (`EngineResult`),
  - a soft-failure exception (`EngineUnavailable`) that lets the dispatcher
    degrade gracefully when a given engine is not installed,
  - MAUD/JDK installation discovery (`find_maud_installation`),
  - cross-engine validation for `engine="both"` (`cross_validate`).

The schema is the one the cross-detector benchmark scoring
(`benchmark/detector_zoo/refine_v2.py`) already consumes. New engines must
write `summary.json` with these keys; downstream tools stay unchanged.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# SOFT-FAIL EXCEPTION FOR MISSING ENGINES
# ============================================================================
class EngineUnavailable(RuntimeError):
    """Raised when a requested Rietveld engine is not installed.

    Carries a human-readable install hint that the MCP layer surfaces back to
    the user (and, for `engine="both"`, triggers a fallback to the available
    engine instead of failing the whole call).
    """

    def __init__(self, engine: str, install_hint: str):
        self.engine = engine
        self.install_hint = install_hint
        super().__init__(f"Rietveld engine '{engine}' is not available. {install_hint}")


# ============================================================================
# CANONICAL SCHEMA (matches benchmark/detector_zoo/refine_v2.py output)
# ============================================================================
@dataclass
class EngineResult:
    """Per-engine summary written as `summary.json` next to the per-slice fits.

    Field names match the benchmark scoring contract verbatim. Engine-specific
    diagnostics (extraction stats, recipe trace, MILK workflow id) go in
    `engine_diagnostics` so the schema stays stable across engines.
    """
    engine: str                       # "gsas2" | "maud"
    detector: Optional[str] = None    # zoo detector tag, if known
    n_total: int = 0
    n_success: int = 0
    n_a_in_window: int = 0
    median_Rwp: Optional[float] = None
    mean_Rwp: Optional[float] = None
    mean_a: Optional[float] = None
    median_a: Optional[float] = None
    std_a: Optional[float] = None
    abs_da_mA: Optional[float] = None         # |mean_a - reference_a| * 1000
    reference_a_A: Optional[float] = None     # NIST cell used to compute abs_da_mA
    limits: Optional[List[float]] = None      # 2θ window (deg)
    engine_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MAUD INSTALLATION DISCOVERY
# Migrated from servers/maud_server.py — single source of truth.
# ============================================================================
def find_maud_installation() -> Optional[Path]:
    """Resolve a MAUD install directory from the standard locations.

    Priority:
      1. $MAUD_PATH
      2. ~/MAUD
      3. ~/.local/MAUD
      4. /opt/MAUD
      5. /Applications/Maud.app/Contents/Resources  (macOS)

    Returns None when no install is found; callers should raise
    EngineUnavailable("maud", ...) so the dispatcher can degrade gracefully.
    """
    env_path = os.environ.get("MAUD_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser().absolute()
        if p.exists() and p.is_dir():
            return p

    candidates = [
        Path.home() / "MAUD",
        Path.home() / ".local" / "MAUD",
        Path("/opt/MAUD"),
        Path("/Applications/Maud.app/Contents/Resources"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def maud_install_hint() -> str:
    """The install hint surfaced through EngineUnavailable when MAUD is missing."""
    return (
        "Install MAUD from http://maud.radiographema.eu/ and either set "
        "$MAUD_PATH or place it at ~/MAUD, ~/.local/MAUD, /opt/MAUD, or "
        "/Applications/Maud.app. MILK (the Python wrapper) installs via "
        "`uv pip install milk-rietveld` and requires a working JDK on PATH."
    )


# ============================================================================
# CROSS-ENGINE VALIDATION (engine="both" mode)
# ============================================================================
def cross_validate(
    gsas2: EngineResult,
    maud: EngineResult,
    *,
    cell_tol_mA: float = 1.0,
    slice_tol_mA: float = 0.5,
    slice_agreement_frac: float = 0.95,
) -> Dict[str, Any]:
    """Compare two engine summaries on the same dataset.

    `cell_tol_mA` is the maximum tolerable disagreement between the two engines'
    aggregate `mean_a` (default 1 mÅ — the same threshold the v2 benchmark
    applies against NIST). `slice_tol_mA` and `slice_agreement_frac` are
    placeholders for per-slice agreement once both engines emit per-slice CSVs;
    until then the verdict is driven by the aggregate gap.

    Returns a dict suitable for `refinement_crossvalidation.json`.
    """
    if gsas2.mean_a is None or maud.mean_a is None:
        return {
            "verdict": "incomplete",
            "reason": "one or both engines returned no successful slices",
            "gsas2_mean_a_A": gsas2.mean_a,
            "maud_mean_a_A": maud.mean_a,
        }

    delta_mA = abs(gsas2.mean_a - maud.mean_a) * 1000.0
    rwp_ratio = (
        (gsas2.median_Rwp / maud.median_Rwp)
        if (gsas2.median_Rwp and maud.median_Rwp) else None
    )

    verdict = "ok" if delta_mA < cell_tol_mA else "disagree"

    return {
        "verdict": verdict,
        "delta_a_mA": delta_mA,
        "cell_tol_mA": cell_tol_mA,
        "gsas2": {
            "mean_a_A": gsas2.mean_a,
            "median_Rwp": gsas2.median_Rwp,
            "n_success": gsas2.n_success,
            "abs_da_mA": gsas2.abs_da_mA,
        },
        "maud": {
            "mean_a_A": maud.mean_a,
            "median_Rwp": maud.median_Rwp,
            "n_success": maud.n_success,
            "abs_da_mA": maud.abs_da_mA,
        },
        "rwp_ratio_gsas2_over_maud": rwp_ratio,
        "notes": (
            "Per-slice agreement (slice_tol_mA, slice_agreement_frac) will be "
            "computed once both engines emit per-slice CSVs in a shared format."
        ),
    }


# ============================================================================
# CONVENIENCE: load an EngineResult from a written summary.json
# ============================================================================
def load_summary(path: Path, engine: str) -> EngineResult:
    """Read a summary.json (benchmark-schema) from disk and tag it with the engine."""
    import json
    data = json.loads(Path(path).read_text())
    return EngineResult(
        engine=engine,
        detector=data.get("detector"),
        n_total=int(data.get("n_total", 0)),
        n_success=int(data.get("n_success", 0)),
        n_a_in_window=int(data.get("n_a_in_window", 0)),
        median_Rwp=data.get("median_Rwp"),
        mean_Rwp=data.get("mean_Rwp"),
        mean_a=data.get("mean_a"),
        median_a=data.get("median_a"),
        std_a=data.get("std_a"),
        abs_da_mA=data.get("abs_da_mA"),
        reference_a_A=data.get("reference_a_A"),
        limits=data.get("limits"),
        engine_diagnostics=data.get("engine_diagnostics", {}),
    )


if __name__ == "__main__":
    # Self-check: print where MAUD would be found.
    maud = find_maud_installation()
    print(f"MAUD install: {maud}", file=sys.stderr)
    print(f"Hint: {maud_install_hint()}", file=sys.stderr)
