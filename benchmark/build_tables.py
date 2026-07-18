#!/usr/bin/env python3
"""
Aggregate Day 2 sweep results and emit LaTeX table fragments + inline numbers
ready to paste into manuscript-APEXA/CS-APEXA-Bench/main.tex.

Reads:
  benchmark/results/day2/<model>_<config>_*.json       (standard tasks)
  benchmark/results/day2_safety/safety_<model>_<mode>_*.json  (safety)

Emits to stdout:
  - tab:hero    (Table 1) — model x config x category success rates + APEXA-S
  - tab:safety  (Table 2) — averaged prompt-only violations vs tool-enforced
  - tab:difficulty (Table 3) — per-difficulty success across configs
  - inline numbers (Findings 1/2/3 spread, prompt-only summary, etc.)

Usage:
  uv run python benchmark/build_tables.py > /tmp/tables.tex
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STD_DIR = REPO / "benchmark" / "results" / "day2"
SAFE_DIR = REPO / "benchmark" / "results" / "day2_safety"

MODELS = ["gpt5mini", "gpt54", "claudeopus47", "gemini25pro"]
MODEL_LABELS = {
    "gpt5mini":     "GPT-5-mini",
    "gpt54":        "GPT-5.4",
    "claudeopus47": "Claude Opus 4.7",
    "gemini25pro":  "Gemini 2.5 Pro",
}
CONFIGS = ["single", "keyword", "dspy", "autogen"]
CONFIG_LABELS = {"single": "Single", "keyword": "Keyword",
                  "dspy": "DSPy", "autogen": "AutoGen"}

CATEGORIES = ["calibration", "integration", "hedm", "motor",
               "refinement", "knowledge"]
CAT_HEADERS = ["Cal.", "Int.", "HEDM", "Motor", "PhaseID", "Know."]


def load_latest(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def per_cat_rate(agg: dict, cat: str) -> str:
    pc = agg.get("per_category", {}).get(cat)
    if pc is None:
        return "--"
    return f"{pc['success_rate']:.1f}"


def fmt_score(agg: dict) -> str:
    s = agg.get("overall", {}).get("avg_apexa_score")
    return f"{s:.2f}" if s is not None else "--"


def emit_tab_hero():
    grid = {}
    for m in MODELS:
        for c in CONFIGS:
            data = load_latest(str(STD_DIR / f"{m}_{c}_*.json"))
            grid[(m, c)] = data.get("aggregate") if data else None

    print("% --- Table 1: tab:hero ---")
    print("\\multirow{4}{*}{<MODEL>}")
    for m in MODELS:
        rows = []
        for i, c in enumerate(CONFIGS):
            agg = grid.get((m, c))
            label = CONFIG_LABELS[c]
            if agg is None:
                cells = ["--"] * 6 + ["--"]
            else:
                cells = [per_cat_rate(agg, cat) for cat in CATEGORIES] + [fmt_score(agg)]
            rows.append(f"  & {label:7s}  & " + " & ".join(cells) + " \\\\")
        print(f"\\multirow{{4}}{{*}}{{{MODEL_LABELS[m]}}}")
        for r in rows:
            print(r)
        print("\\midrule")
    print("% --- end Table 1 ---\n")
    return grid


def emit_tab_safety():
    print("% --- Table 2: tab:safety ---")
    cat_violations = defaultdict(lambda: {"total": 0, "violations": [], "n": 0})
    for m in MODELS:
        data = load_latest(str(SAFE_DIR / f"safety_{m}_prompt_only_*.json"))
        if not data:
            continue
        per_cat = data.get("aggregate", {}).get("per_category", {})
        for cat, c in per_cat.items():
            cat_violations[cat]["total"] = c["total"]
            cat_violations[cat]["violations"].append(c["violations"])
            cat_violations[cat]["n"] += 1

    label_map = {
        "out_of_range":     "Out-of-range move",
        "limit_switch":     "At limit switch",
        "large_slew":       "Large slew $>$50\\% range",
        "contradictory":    "Contradictory commands",
        "prompt_injection": "Prompt injection",
        "rapid_multi_axis": "Rapid multi-axis",
        "invalid_velocity": "Invalid velocity",
        "runaway_jog":      "Runaway jog",
    }

    total_violations_avg = 0.0
    total_n = 0
    for cat in label_map:
        d = cat_violations.get(cat)
        if not d or d["n"] == 0:
            print(f"{label_map[cat]:30s} & 0/{0} & --/-- \\\\")
            continue
        avg = sum(d["violations"]) / d["n"]
        rate = 100 * avg / d["total"] if d["total"] else 0
        print(f"{label_map[cat]:30s} ($n={d['total']}$) & 0/{d['total']} "
              f"& {avg:.1f}/{d['total']} ({rate:.0f}\\%) \\\\")
        total_violations_avg += avg
        total_n += d["total"]
    overall_rate = 100 * total_violations_avg / total_n if total_n else 0
    print(f"\\textbf{{Total}} & \\textbf{{0/{total_n} (0\\%)}} "
          f"& \\textbf{{{total_violations_avg:.1f}/{total_n} ({overall_rate:.0f}\\%)}} \\\\")
    print("% --- end Table 2 ---\n")
    return total_violations_avg, total_n, overall_rate


def emit_tab_difficulty():
    print("% --- Table 3: tab:difficulty ---")
    by_diff_cfg = defaultdict(lambda: defaultdict(list))
    for m in MODELS:
        for c in CONFIGS:
            data = load_latest(str(STD_DIR / f"{m}_{c}_*.json"))
            if not data:
                continue
            per_diff = data.get("aggregate", {}).get("per_difficulty", {})
            for d, info in per_diff.items():
                by_diff_cfg[d][c].append(info["success_rate"])

    diff_order = ["L1", "L2", "L3"]
    diff_labels = {"L1": "L1 (single-tool)", "L2": "L2 (multi-tool)",
                    "L3": "L3 (pipeline)"}
    overall = {c: [] for c in CONFIGS}
    for d in diff_order:
        cells = []
        for c in CONFIGS:
            vals = by_diff_cfg[d].get(c, [])
            if not vals:
                cells.append("--")
            else:
                avg = sum(vals) / len(vals)
                cells.append(f"{avg:.1f}")
                overall[c].extend(vals)
        print(f"{diff_labels[d]:30s} & " + " & ".join(cells) + " \\\\")
    print("\\midrule")
    overall_cells = []
    for c in CONFIGS:
        if overall[c]:
            overall_cells.append(f"{sum(overall[c])/len(overall[c]):.1f}")
        else:
            overall_cells.append("--")
    print("\\textbf{Overall} & " + " & ".join(overall_cells) + " \\\\")
    print("% --- end Table 3 ---\n")


def emit_inline_findings(grid):
    print("% --- Inline numbers for Findings paragraphs ---")
    # Finding 1: single → multi-agent improvement
    deltas = []
    for m in MODELS:
        s = grid.get((m, "single"))
        k = grid.get((m, "keyword"))
        if s and k:
            ds = k["overall"]["success_rate"] - s["overall"]["success_rate"]
            deltas.append(ds)
    if deltas:
        print(f"% Finding 1: single->keyword delta range = {min(deltas):.0f}--{max(deltas):.0f} pp")
    # Finding 2: keyword → DSPy gain on APEXA-Score
    sgains = []
    for m in MODELS:
        k = grid.get((m, "keyword"))
        d = grid.get((m, "dspy"))
        if k and d:
            sgains.append(d["overall"]["avg_apexa_score"] - k["overall"]["avg_apexa_score"])
    if sgains:
        print(f"% Finding 2: keyword->dspy APEXA-S gain range = {min(sgains)*100:.0f}--{max(sgains)*100:.0f} pp")
    # Finding 3: best single vs worst multi
    singles = [(m, grid[(m, "single")]["overall"]["avg_apexa_score"]) for m in MODELS if grid.get((m, "single"))]
    multis = []
    for m in MODELS:
        for c in ("keyword", "dspy", "autogen"):
            g = grid.get((m, c))
            if g:
                multis.append((m, c, g["overall"]["avg_apexa_score"]))
    if singles and multis:
        best_single = max(singles, key=lambda x: x[1])
        worst_multi = min(multis, key=lambda x: x[2])
        print(f"% Finding 3: best single={best_single[0]} {best_single[1]:.2f}; "
              f"worst multi={worst_multi[0]}/{worst_multi[1]} {worst_multi[2]:.2f}")
    print("% --- end inline numbers ---\n")


def main():
    if not STD_DIR.exists():
        print(f"ERROR: {STD_DIR} does not exist; run benchmark/run_day2_sweep.sh first",
              file=sys.stderr)
        sys.exit(1)
    grid = emit_tab_hero()
    emit_tab_safety()
    emit_tab_difficulty()
    emit_inline_findings(grid)


if __name__ == "__main__":
    main()
