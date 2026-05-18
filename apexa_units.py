"""Centralized unit conversions for APEXA.

Single source of truth for X-ray energy↔wavelength used by every MCP tool
and helper that needs it. Replaces scattered `12.398 / E` literals.

Backend priority:
  1. xrayutilities.en2lam / lam2en  (CODATA-based; ~6 μÅ more accurate
     than 12.398, and matches what xrayutilities downstream reports)
  2. CODATA 2018 fallback (h·c = 12.39841984332 keV·Å) — kicks in when
     this module is imported by a subprocess running in midas_env (which
     does not have xrayutilities). Behavior is identical to the
     xrayutilities path within float64 precision.

The fallback exists ONLY so subprocess invocations of MIDAS/GSAS-II tools
don't have to pull in xrayutilities; in the APEXA process itself,
xrayutilities is always the active backend.
"""
from __future__ import annotations

# CODATA 2018 product of Planck's constant and speed of light, in keV·Å.
# Source: https://physics.nist.gov/cgi-bin/cuu/Value?hcev (4.135667696e-15 eV·s × 2.99792458e8 m/s)
_HC_KEV_ANGSTROM = 12.39841984332

try:
    import xrayutilities as _xu  # type: ignore
    _BACKEND = "xrayutilities"
except ImportError:
    _xu = None
    _BACKEND = "codata-fallback"


def kev_to_angstrom(energy_kev: float) -> float:
    """Convert X-ray photon energy (keV) to wavelength (Å)."""
    e = float(energy_kev)
    if e <= 0:
        raise ValueError(f"energy must be positive, got {energy_kev}")
    if _xu is not None:
        return float(_xu.en2lam(e * 1000.0))  # xu wants eV
    return _HC_KEV_ANGSTROM / e


def angstrom_to_kev(wavelength_a: float) -> float:
    """Convert X-ray wavelength (Å) to photon energy (keV)."""
    w = float(wavelength_a)
    if w <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength_a}")
    if _xu is not None:
        return float(_xu.lam2en(w) / 1000.0)  # xu returns eV
    return _HC_KEV_ANGSTROM / w


def backend() -> str:
    """Name of the active backend: 'xrayutilities' or 'codata-fallback'."""
    return _BACKEND
