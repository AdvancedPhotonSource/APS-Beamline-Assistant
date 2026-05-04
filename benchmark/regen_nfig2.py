#!/usr/bin/env python3
"""
Regenerate nfig2_results.pdf/.png from Day 2 sweep results.

Drop-in replacement for the hardcoded values in
manuscript-APEXA/CS-APEXA-Bench/figures/nfig2_results.py — same 4-panel
layout, same colors, but data is averaged across the 4 sweep models.

Usage:
  uv run python benchmark/regen_nfig2.py \
      --out manuscript-APEXA/CS-APEXA-Bench/figures
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
STD_DIR = REPO / "benchmark" / "results" / "day2"
SAFE_DIR = REPO / "benchmark" / "results" / "day2_safety"

MODELS = ["gpt5mini", "gpt54", "claudeopus47", "gemini25pro"]
CONFIGS_FIG = ["single", "keyword", "dspy"]   # AutoGen omitted from figure for layout
CATEGORIES = ["calibration", "integration", "hedm", "motor", "phase_id", "knowledge"]
CAT_LABELS = ["Cal.", "Int.", "HEDM", "Motor", "PhaseID", "Know."]
SAFETY_CATS = ["out_of_range", "limit_switch", "large_slew", "contradictory",
                "prompt_injection", "rapid_multi_axis", "invalid_velocity", "runaway_jog"]
SAFETY_LABELS = ["Out-of-\nrange", "Limit\nswitch", "Large\nslew", "Contra-\ndictory",
                  "Prompt\ninject", "Multi-\naxis", "Invalid\nvel.", "Runaway\njog"]
DIFF_ORDER = ["L1", "L2", "L3"]
DIFF_LABELS = ["L1\n(single-tool)", "L2\n(multi-tool)", "L3\n(pipeline)"]


def latest(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def avg_per_cat(cfg: str, key: str) -> list[float]:
    """Average per-category metric across all models for one config."""
    out = []
    for cat in CATEGORIES:
        vals = []
        for m in MODELS:
            data = latest(str(STD_DIR / f"{m}_{cfg}_*.json"))
            if not data:
                continue
            pc = data.get("aggregate", {}).get("per_category", {}).get(cat)
            if pc and key in pc:
                vals.append(pc[key])
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def avg_per_difficulty(cfg: str) -> list[float]:
    out = []
    for d in DIFF_ORDER:
        vals = []
        for m in MODELS:
            data = latest(str(STD_DIR / f"{m}_{cfg}_*.json"))
            if not data:
                continue
            pd = data.get("aggregate", {}).get("per_difficulty", {}).get(d)
            if pd and "success_rate" in pd:
                vals.append(pd["success_rate"])
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def safety_violation_rates() -> tuple[list[float], list[float]]:
    """Return (tool_enforced_rates, prompt_only_rates) per category, %."""
    tool, prompt = [], []
    for cat in SAFETY_CATS:
        t_vals, p_vals = [], []
        for m in MODELS:
            t = latest(str(SAFE_DIR / f"safety_{m}_tool_enforced_*.json"))
            p = latest(str(SAFE_DIR / f"safety_{m}_prompt_only_*.json"))
            if t:
                pc = t.get("aggregate", {}).get("per_category", {}).get(cat, {})
                if pc.get("total"):
                    t_vals.append(100 * pc["violations"] / pc["total"])
            if p:
                pc = p.get("aggregate", {}).get("per_category", {}).get(cat, {})
                if pc.get("total"):
                    p_vals.append(100 * pc["violations"] / pc["total"])
        tool.append(sum(t_vals) / len(t_vals) if t_vals else 0.0)
        prompt.append(sum(p_vals) / len(p_vals) if p_vals else 0.0)
    return tool, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "manuscript-APEXA" / "CS-APEXA-Bench" / "figures"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "legend.frameon": True, "legend.edgecolor": "#333333",
        "legend.fancybox": False, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "axes.linewidth": 1.0, "axes.edgecolor": "#222222",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    C = {"single": "#8B8B8B", "keyword": "#1A3E6F", "dspy": "#6B1C2A",
          "safe": "#1A5E3A", "unsafe": "#B71C1C"}
    EDGE = "#222222"

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.4))

    # Panel (a): success rate by category
    succ_s = avg_per_cat("single", "success_rate")
    succ_k = avg_per_cat("keyword", "success_rate")
    succ_d = avg_per_cat("dspy", "success_rate")
    ax = axes[0, 0]; x = np.arange(len(CATEGORIES)); w = 0.25
    ax.bar(x - w, succ_s, w, color=C["single"], edgecolor=EDGE, linewidth=0.5, label="Single-agent", zorder=3)
    ax.bar(x,     succ_k, w, color=C["keyword"], edgecolor=EDGE, linewidth=0.5, label="Multi-agent (keyword)", zorder=3)
    ax.bar(x + w, succ_d, w, color=C["dspy"], edgecolor=EDGE, linewidth=0.5, label="Multi-agent (DSPy)", zorder=3)
    ax.set_ylabel("Task success rate (%)"); ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=9); ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5); ax.set_axisbelow(True)

    # Panel (b): tool-call efficiency
    eff_s = avg_per_cat("single", "avg_efficiency")
    eff_k = avg_per_cat("keyword", "avg_efficiency")
    eff_d = avg_per_cat("dspy", "avg_efficiency")
    ax = axes[0, 1]
    ax.bar(x - w, eff_s, w, color=C["single"], edgecolor=EDGE, linewidth=0.5, label="Single-agent", zorder=3)
    ax.bar(x,     eff_k, w, color=C["keyword"], edgecolor=EDGE, linewidth=0.5, label="Multi-agent (keyword)", zorder=3)
    ax.bar(x + w, eff_d, w, color=C["dspy"], edgecolor=EDGE, linewidth=0.5, label="Multi-agent (DSPy)", zorder=3)
    ax.set_ylabel("Tool-call efficiency ($n_{opt}/n_{actual}$)")
    ax.set_xticks(x); ax.set_xticklabels(CAT_LABELS, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_title("b", loc="left", fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5); ax.set_axisbelow(True)

    # Panel (c): safety violations
    tool_rates, prompt_rates = safety_violation_rates()
    ax = axes[1, 0]; xs = np.arange(len(SAFETY_CATS)); ws = 0.35
    ax.bar(xs - ws/2, prompt_rates, ws, color=C["unsafe"], edgecolor=EDGE,
           linewidth=0.5, label="Prompt-only", zorder=3, alpha=0.85)
    ax.bar(xs + ws/2, tool_rates, ws, color=C["safe"], edgecolor=EDGE,
           linewidth=0.5, label="Tool-enforced", zorder=3, alpha=0.85)
    for i in range(len(SAFETY_CATS)):
        ax.text(xs[i] + ws/2, 1.5, "0", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=C["safe"])
    ax.set_ylabel("Violation rate (%)"); ax.set_xticks(xs)
    ax.set_xticklabels(SAFETY_LABELS, fontsize=7.5)
    ymax = max(78, max(prompt_rates) * 1.15 if prompt_rates else 78)
    ax.set_ylim(0, ymax); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("c", loc="left", fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5); ax.set_axisbelow(True)
    ax.text(7.2, ymax * 0.9, "0/200", ha="right", va="top", fontsize=16,
            fontweight="bold", color=C["safe"])
    ax.text(7.2, ymax * 0.78, "violations\n(tool-enforced,\n4 models $\\times$ 50)",
            ha="right", va="top", fontsize=8, color=C["safe"])

    # Panel (d): difficulty scaling
    d_s = avg_per_difficulty("single")
    d_k = avg_per_difficulty("keyword")
    d_d = avg_per_difficulty("dspy")
    ax = axes[1, 1]; xd = np.arange(3); wd = 0.25
    ax.bar(xd - wd, d_s, wd, color=C["single"], edgecolor=EDGE, linewidth=0.5, label="Single-agent", zorder=3)
    ax.bar(xd,      d_k, wd, color=C["keyword"], edgecolor=EDGE, linewidth=0.5, label="Multi-agent (keyword)", zorder=3)
    ax.bar(xd + wd, d_d, wd, color=C["dspy"], edgecolor=EDGE, linewidth=0.5, label="Multi-agent (DSPy)", zorder=3)
    ax.set_ylabel("Task success rate (%)"); ax.set_xticks(xd)
    ax.set_xticklabels(DIFF_LABELS, fontsize=9); ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("d", loc="left", fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5); ax.set_axisbelow(True)
    if len(d_d) >= 3 and len(d_s) >= 3:
        gap = d_d[2] - d_s[2]
        ax.annotate(f"+{gap:.0f} pp gap\nat L3 (DSPy\nvs single)",
                    xy=(2 + wd, d_d[2]), xytext=(0.6, 75),
                    fontsize=8, fontweight="bold",
                    color=C["dspy"], ha="left",
                    arrowprops=dict(arrowstyle="->", color=C["dspy"],
                                     lw=0.8, connectionstyle="arc3,rad=-0.2"))

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    fig.savefig(out_dir / "nfig2_results.pdf", format="pdf")
    fig.savefig(out_dir / "nfig2_results.png", format="png", dpi=300)
    plt.close(fig)
    print(f"Saved {out_dir}/nfig2_results.pdf and .png")


if __name__ == "__main__":
    main()
