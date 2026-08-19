"""Tests for the reviewable analysis-plan schema + acquisition watcher.

Covers the pieces that make the plan/watcher layer *in sync* with the FF-HEDM
graph + idempotency work:

  1. APEXAPlan round-trips through YAML and validate() is correct (energy is
     optional when it comes from a parameter_file).
  2. to_integrate_series_kwargs() drops unset fields (so tool defaults apply).
  3. The watcher groups frames by stem and excludes darks.
  4. is_complete() honours an explicit expected_count and the quiet-period rule.
  5. THE COHERENCE GUARD: the watcher refuses to auto-fire midas_integrate_series
     for HEDM/calibration techniques (those need the gated graph, not a blind
     integrate call) and only auto-executes the integration family.

Run: python -m pytest tests/test_plan_and_watcher.py  (or run the file directly)
"""
import os
import sys
import tempfile
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apexa_plan import (APEXAPlan, Beam, Calibration, DataSpec, DarkSpec, Grid,
                        Sample)
import apexa_acquisition_watcher as watcher


def test_plan_roundtrip_and_kwargs():
    plan = APEXAPlan(
        technique="waxs",
        sample=Sample(name="JL_0Nb", expected_phases=["FCC"]),
        beam=Beam(energy_keV=61.332),
        calibration=Calibration(calibrant="CeO2", parameter_file="ceria.txt"),
        data=DataSpec(image_dir="/scratch/JL_0Nb", pattern="JL_0Nb_*.h5",
                      stem="JL_0Nb", data_location="exchange/data"),
        dark=DarkSpec(source="file", kind="after", location="exchange/data"),
        grid=Grid(q_min=0.5, q_max=12.0, n_channels=2500),
    )
    plan.assume("dark.kind", "after", "beamline default trailing dark")
    text = plan.to_yaml()
    back = APEXAPlan.from_dict(yaml.safe_load(text))
    assert back.technique == "waxs"
    assert back.sample.name == "JL_0Nb"
    assert back.grid.n_channels == 2500
    assert back.assumptions and back.assumptions[0].field == "dark.kind"
    # kwargs projection drops None; keeps set grid + dark fields
    kw = plan.to_integrate_series_kwargs()
    assert kw["parameter_file"] == "ceria.txt"
    assert kw["q_min"] == 0.5 and kw["n_channels"] == 2500
    assert "r_min" not in kw  # unset -> dropped so the tool default applies
    print("test_plan_roundtrip_and_kwargs OK")


def test_validate_energy_optional_with_param_file():
    # integration with a param file: energy lives in the params -> no energy needed
    p = APEXAPlan(technique="integration",
                  calibration=Calibration(parameter_file="p.txt"),
                  data=DataSpec(stem="s"))
    assert p.validate() == [], p.validate()
    # bare calibration with no param file + no energy -> flagged
    q = APEXAPlan(technique="calibration", calibration=Calibration(calibrant="CeO2"))
    assert any("energy" in i for i in q.validate())
    # unknown technique flagged
    r = APEXAPlan(technique="bogus")
    assert any("unknown technique" in i for i in r.validate())
    print("test_validate_energy_optional_with_param_file OK")


def test_stem_grouping_and_dark_exclusion():
    files = [
        "/d/JL_0Nb_00001.vrx.h5", "/d/JL_0Nb_00002.vrx.h5", "/d/JL_0Nb_00003.vrx.h5",
        "/d/JL_5Nb_00001.vrx.h5",
        "/d/dark_00001.vrx.h5",          # excluded
    ]
    groups = watcher.group_by_stem(files, exclude_substring="dark")
    assert set(groups) == {"JL_0Nb", "JL_5Nb"}, groups
    assert len(groups["JL_0Nb"]) == 3
    assert all("dark" not in f for g in groups.values() for f in g)
    print("test_stem_grouping_and_dark_exclusion OK")


def test_is_complete_count_and_quiet():
    paths = ["a", "b", "c"]
    done, why = watcher.is_complete(paths, expected_count=3)
    assert done and "expected_count" in why
    wait, why = watcher.is_complete(paths, expected_count=5)
    assert not wait and "3/5" in why
    # quiet-period path needs real files with mtimes
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "f.h5")
        open(fp, "w").close()
        old = time.time() - 120
        os.utime(fp, (old, old))
        done, why = watcher.is_complete([fp], expected_count=None,
                                        quiet_seconds=30.0)
        assert done and "no new frames" in why
    print("test_is_complete_count_and_quiet OK")


def test_watcher_refuses_hedm_autoexecute():
    """The coherence guard: HEDM/calibration must NOT auto-fire integrate."""
    for tech in ("ff-hedm", "nf-hedm", "pf-hedm", "calibration"):
        plan = APEXAPlan(technique=tech,
                         calibration=Calibration(parameter_file="p.txt"),
                         data=DataSpec(stem="s"))
        try:
            watcher.execute_plan(plan)
            assert False, f"execute_plan should refuse technique {tech}"
        except ValueError as e:
            assert "gated" in str(e) or "auto-executable" in str(e)
    # integration family is allowed through the guard (import happens lazily)
    for tech in ("waxs", "saxs", "integration"):
        assert tech in watcher._AUTO_EXECUTABLE
    print("test_watcher_refuses_hedm_autoexecute OK")


if __name__ == "__main__":
    test_plan_roundtrip_and_kwargs()
    test_validate_energy_optional_with_param_file()
    test_stem_grouping_and_dark_exclusion()
    test_is_complete_count_and_quiet()
    test_watcher_refuses_hedm_autoexecute()
    print("\nall plan/watcher tests passed")
