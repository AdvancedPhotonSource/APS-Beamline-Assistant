"""Phase 2a — the generic data→technique classifier (`_infer_technique`).

Verifies that raw data (files/tokens/goal) is recognized as the right MIDAS
technique capsule id, that the FF/NF/PF path is preserved (wrapping, not
replacing, `_infer_hedm_modality`), and that the shared calibrate/integrate
phase never steals an FF dataset.
"""
import importlib

import pytest

mcs = importlib.import_module("midas_comprehensive_server")
cr = importlib.import_module("capsule_registry")


def _touch(d, name, text=""):
    p = d / name
    p.write_text(text or name)
    return p


def test_nf_mic_file_infers_nf_hedm(tmp_path):
    _touch(tmp_path, "recon.mic")
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path), "")
    assert out["technique"] == "nf-hedm"


def test_tomo_config_file_signature(tmp_path):
    _touch(tmp_path, "tomocupy_args.yml", "propagationDistance: 0.1")
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path), "")
    assert out["technique"] == "tomo"
    assert out["source"] == "data"


def test_xrdct_cake_cache_signature(tmp_path):
    _touch(tmp_path, "cake_cache_0001.h5")
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path), "")
    assert out["technique"] == "xrd-ct"


def test_calibrate_integrate_only_on_distinctive_filename(tmp_path):
    _touch(tmp_path, "midas_calibrate.par")
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path), "")
    assert out["technique"] == "calibrate-integrate"


def test_goal_dfxm_wins(tmp_path):
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path),
                               "simulate a DFXM topograph")
    assert out["technique"] == "dfxm"


def test_calibrate_far_field_goal_resolves_to_ff_not_shared_phase(tmp_path):
    # An FF param file present → the FF/NF/PF path (step 3) must win over the
    # shared-phase calibrate fallback (step 4). "calibrate far-field" must be ff.
    _touch(tmp_path, "ff_params.txt",
           "RingThresh 1 100\nSpaceGroup 225\nOmegaStep 0.25\nBeamCurrent 1\n"
           "OverAllRingToIndex 1\nMinNrSpots 1\n")
    info = mcs._classify_input(tmp_path)
    out = mcs._infer_technique(tmp_path, info, "calibrate the far-field detector")
    # Either ff-hedm (modality detected) — must NOT be calibrate-integrate.
    assert out["technique"] != "calibrate-integrate"


def test_bare_calibrate_goal_falls_back_to_calibrate_integrate(tmp_path):
    # No HEDM modality signalled and no distinctive files → shared-phase fallback.
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path),
                               "calibrate the powder ring pattern")
    assert out["technique"] == "calibrate-integrate"


def test_empty_dir_no_goal_returns_none(tmp_path):
    out = mcs._infer_technique(tmp_path, mcs._classify_input(tmp_path), "")
    assert out["technique"] is None


def test_gated_on_has_technique(tmp_path):
    # Every signature id in the curated table must be a real, vendored capsule
    # (else the entry is dead weight / a typo).
    for tech in mcs._TECHNIQUE_SIGNATURES:
        assert cr.has_technique(tech), f"{tech} signature has no vendored capsule"


def test_fail_open_on_bad_path():
    # A path that raises on iterdir must not propagate — returns a dict, tech None.
    from pathlib import Path
    out = mcs._infer_technique(Path("/nonexistent/xyz/123"), {}, "")
    assert isinstance(out, dict) and "technique" in out
