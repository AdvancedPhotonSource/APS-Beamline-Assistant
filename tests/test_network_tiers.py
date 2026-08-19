"""Network-tier tests — the copland launch failure must not recur.

APEXA hung at startup on a beamline host with data + ANL-internal access but no
public internet: the knowledge-base pre-warm blocked on a HuggingFace HEAD
request. The existing ``try/except`` did not help, because a network hang is not
an exception — so these tests assert the *avoidance*, not the recovery.

No network is touched: everything here is env-var driven policy.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import apexa_network as N  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("APEXA_NETWORK", "APEXA_OFFLINE", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "APEXA_KB_PREWARM"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(N)
    yield


def test_default_tier_is_internal_not_web():
    """The default must assume the SMALLER network.

    Defaulting to web-optimistic is what caused the copland hang: a host that
    cannot reach the internet discovers it by blocking. Beamline hosts are the
    primary deployment, so they must be the safe default and web-capable hosts
    opt up.
    """
    assert N.tier() == N.INTERNAL
    assert not N.has_web()
    assert N.has_internal()


@pytest.mark.parametrize("value,expected", [
    ("web", N.WEB), ("internal", N.INTERNAL), ("data", N.DATA),
    ("WEB", N.WEB), ("  internal  ", N.INTERNAL), ("nonsense", N.INTERNAL),
])
def test_tier_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("APEXA_NETWORK", value)
    assert N.tier() == expected


def test_legacy_offline_flag_caps_the_tier(monkeypatch):
    """APEXA_OFFLINE=1 predates this module and must keep working."""
    monkeypatch.setenv("APEXA_NETWORK", "web")
    monkeypatch.setenv("APEXA_OFFLINE", "1")
    assert N.tier() == N.INTERNAL
    assert not N.has_web()


def test_hf_hub_offline_also_caps_the_tier(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "web")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert N.tier() == N.INTERNAL


# ── the actual bug: startup must never block ─────────────────────────────────

def test_prewarm_blocked_below_web_tier(monkeypatch):
    """THE regression test. Pre-warm loads a ~700MB HF model; without the web and
    without a staged cache that call blocks and the MCP server never starts."""
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    assert N.prewarm_allowed() is False
    monkeypatch.setenv("APEXA_NETWORK", "data")
    assert N.prewarm_allowed() is False


def test_prewarm_allowed_on_web_tier(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "web")
    assert N.prewarm_allowed() is True


def test_operator_can_opt_into_prewarm_with_staged_cache(monkeypatch):
    """Correct once the embedder cache really is staged on an offline host."""
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    monkeypatch.setenv("APEXA_KB_PREWARM", "1")
    assert N.prewarm_allowed() is True


def test_operator_can_force_prewarm_off_even_on_web(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "web")
    monkeypatch.setenv("APEXA_KB_PREWARM", "0")
    assert N.prewarm_allowed() is False


def test_offline_env_is_applied_below_web_and_not_above(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    assert N.apply_offline_env() is True
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    monkeypatch.setenv("APEXA_NETWORK", "web")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert N.apply_offline_env() is False
    assert "HF_HUB_OFFLINE" not in os.environ


# ── web-only tools ───────────────────────────────────────────────────────────

def test_web_only_tools_refused_below_web_tier(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    for name in N.WEB_ONLY_TOOLS:
        reason = N.tool_unavailable_reason(name)
        assert reason and "public internet" in reason
        # Must tell the model not to fabricate — a blocked tool is a common
        # trigger for inventing the result it would have returned.
        assert "invent" in reason.lower()
    assert set(N.disabled_tools()) == set(N.WEB_ONLY_TOOLS)


def test_web_only_tools_allowed_on_web_tier(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "web")
    for name in N.WEB_ONLY_TOOLS:
        assert N.tool_unavailable_reason(name) is None
    assert tuple(N.disabled_tools()) == ()


def test_data_reduction_tools_are_never_network_gated(monkeypatch):
    """Everything that actually reduces data must work at `internal`."""
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    for name in ("run_ff_hedm_full_workflow", "midas_integrate_series",
                 "midas_auto_calibrate", "list_directory", "run_command",
                 "run_remote_command", "read_grains_summary",
                 "build_detector_mask", "query_hedm_knowledge"):
        assert N.tool_unavailable_reason(name) is None, f"{name} must not be gated"


def test_rag_is_degraded_not_disabled(monkeypatch):
    """RAG needs the web only for the FIRST model download; with a staged cache
    it works offline, so it must never be hard-disabled."""
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    assert "query_hedm_knowledge" not in N.disabled_tools()
    assert "query_hedm_knowledge" in N.WEB_DEGRADED


def test_describe_is_operator_readable(monkeypatch):
    monkeypatch.setenv("APEXA_NETWORK", "internal")
    d = N.describe()
    assert "internal" in d and "disabled" in d
    monkeypatch.setenv("APEXA_NETWORK", "data")
    assert "will not work" in N.describe()


# ── tool surface integration ─────────────────────────────────────────────────

def test_tool_surface_hides_web_tools_below_web_tier(monkeypatch):
    """Defence in depth: the chokepoint refuses them, the surface hides them so
    the model never tries in the first place."""
    import apexa_toolsurface as TS
    importlib.reload(TS)

    def schema(n):
        return {"type": "function",
                "function": {"name": n, "description": f"{n} tool",
                             "parameters": {"type": "object", "properties": {}}}}

    cat = [schema(n) for n in ("list_directory", "read_file", "write_file",
                               "get_file_info", "run_command", "run_remote_command",
                               "recommend_workflow", "query_hedm_knowledge",
                               "inspect_dataset_file", "fetch_cif_from_mp",
                               "get_bibtex")]

    monkeypatch.setenv("APEXA_NETWORK", "internal")
    names = {TS.tool_name(t) for t in TS.initial_surface(cat)}
    assert "fetch_cif_from_mp" not in names and "get_bibtex" not in names
    assert "list_directory" in names
    assert not any(h["name"] in N.WEB_ONLY_TOOLS
                   for h in TS.search(cat, "fetch a cif structure file"))

    monkeypatch.setenv("APEXA_NETWORK", "web")
    hits = {h["name"] for h in TS.search(cat, "fetch cif from materials project")}
    assert "fetch_cif_from_mp" in hits


def test_probe_returns_a_valid_tier_and_checks():
    """Bounded TCP probe; used only by setup_user.sh, never in the hot path."""
    detected, checks = N.probe(timeout=0.001)     # near-instant, likely all False
    assert detected in (N.WEB, N.INTERNAL, N.DATA)
    assert set(checks) == {"anl_internal", "alcf", "web"}
