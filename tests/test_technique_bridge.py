"""Phase 2c — the bridge: `_maybe_capsule_msg` reads `capsule_technique` from a
tool RESULT payload and injects that technique's methodology spine.

This is the headline handoff — "data looks like NF" (emitted by recommend_workflow
as capsule_technique) → "load the NF methodology" (spine injected before the next
workflow tool fires). Also covers dedup, fall-through to tool-name resolution, and
fail-open on malformed input.
"""
import importlib
import json

import pytest

agents = importlib.import_module("apexa_agents")
cr = importlib.import_module("capsule_registry")

_inject = agents.AgentRunner._maybe_capsule_msg


def test_bridge_result_technique_injects_spine():
    seen = set()
    result = json.dumps({"status": "success", "capsule_technique": "nf-hedm"})
    msg = _inject("recommend_workflow", seen, result=result)
    assert msg is not None
    assert msg["role"] == "system"
    assert "nf-hedm" in seen
    # The injected body should be the technique handbook, not empty boilerplate.
    assert "TECHNIQUE HANDBOOK" in msg["content"]


def test_bridge_dedup_second_call_returns_none():
    seen = set()
    result = json.dumps({"capsule_technique": "ff-hedm"})
    first = _inject("recommend_workflow", seen, result=result)
    assert first is not None
    second = _inject("recommend_workflow", seen, result=result)
    assert second is None  # already injected this turn


def test_bridge_scans_recommendations_list():
    seen = set()
    result = json.dumps({
        "status": "success",
        "recommendations": [
            {"tool": "foo", "capsule_technique": ""},
            {"tool": "bar", "capsule_technique": "pf-hedm"},
        ],
    })
    msg = _inject("recommend_workflow", seen, result=result)
    assert msg is not None
    assert "pf-hedm" in seen


def test_bridge_result_wins_over_tool_name():
    # A tool whose NAME carries ff, but whose RESULT says nf → result wins.
    seen = set()
    result = json.dumps({"capsule_technique": "nf-hedm"})
    _inject("run_ff_hedm_full_workflow", seen, result=result)
    assert "nf-hedm" in seen and "ff-hedm" not in seen


def test_bridge_falls_back_to_tool_name_without_result():
    seen = set()
    msg = _inject("run_ff_hedm_full_workflow", seen, result=None)
    # ff-hedm resolvable from the tool name.
    assert msg is not None
    assert "ff-hedm" in seen


def test_bridge_malformed_json_is_fail_open():
    seen = set()
    # Malformed result + a tool name that resolves to nothing → None, no raise.
    msg = _inject("list_directory", seen, result="{not json")
    assert msg is None


def test_bridge_unknown_technique_ignored():
    seen = set()
    result = json.dumps({"capsule_technique": "not-a-real-technique"})
    msg = _inject("list_directory", seen, result=result)
    assert msg is None
    assert "not-a-real-technique" not in seen
