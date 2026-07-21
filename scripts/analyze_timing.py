#!/usr/bin/env python3
"""Summarise APEXA timing JSONL into the tables you'd bring to a benchmark
discussion (e.g. mapping APEXA against NVIDIA's DGX Spark agentic numbers).

Reads the JSONL emitted by apexa_timing.log_llm_call() — three row types:

    endpoint = argo-chat       one row per blocking /chat/ HTTP call        (Tier 1)
    endpoint = argo-messages   one row per streaming /messages/ call        (Tier 3)
                               — adds ttft_s and tpot_ms
    endpoint = query           one row per user query wrapping the above    (Tier 2)

Usage
-----
    ./scripts/analyze_timing.py                          # ~/.apexa/timing.jsonl
    ./scripts/analyze_timing.py path/to/log.jsonl
    ./scripts/analyze_timing.py --json                   # machine-readable
    ./scripts/analyze_timing.py --since 2026-07-15       # filter by date
    ./scripts/analyze_timing.py --model gpt55            # single model

Depends only on the stdlib. Prints three tables:
  1. Per-model LLM call summary   (elapsed, tokens, gen_tps, ttft, tpot, retry rate)
  2. Per-agent query summary       (round-trips, tool_s, llm_s, wall_s)
  3. Percentiles for the workhorse metric (user-perceived wall_clock_s)
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_LOG = Path.home() / ".apexa" / "timing.jsonl"


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load(path: Path, since: datetime | None, model: str | None) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"error: log not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and d.get("iso_ts", "") < since.isoformat():
                continue
            if model and d.get("model") not in (model, None):
                continue
            rows.append(d)
    return rows


def _fmt(v: Any, w: int = 8, prec: int = 2) -> str:
    if v is None:
        return f"{'—':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.{prec}f}"
    return f"{str(v):>{w}}"


# ─── Table 1: per-model LLM call summary ──────────────────────────────────

def summarise_llm(rows: list[dict]) -> dict:
    # Group by (endpoint, model); include only completed calls (http 200, no error).
    groups: dict[tuple, list[dict]] = defaultdict(list)
    retries: dict[tuple, int]       = defaultdict(int)
    for r in rows:
        ep = r.get("endpoint")
        if ep not in ("argo-chat", "argo-messages"):
            continue
        key = (ep, r.get("model", "?"))
        err = r.get("error", "")
        if err.startswith("retry_"):
            retries[key] += 1
            continue
        if r.get("http_status") != 200:
            continue
        groups[key].append(r)
    out: dict[tuple, dict] = {}
    for key, rs in groups.items():
        elapsed = [r["elapsed_s"] for r in rs if r.get("elapsed_s")]
        p_tok   = [r["prompt_tok"] for r in rs if r.get("prompt_tok")]
        r_tok   = [r["response_tok"] for r in rs if r.get("response_tok")]
        gen_tps = [r["gen_tps"] for r in rs if r.get("gen_tps")]
        ttft    = [r["ttft_s"] for r in rs if r.get("ttft_s") is not None]
        tpot    = [r["tpot_ms"] for r in rs if r.get("tpot_ms") is not None]
        n_call  = len(rs)
        n_retry = retries.get(key, 0)
        out[key] = {
            "n_calls":       n_call,
            "n_retries":     n_retry,
            "retry_rate":    n_retry / (n_call + n_retry) if (n_call + n_retry) else 0,
            "elapsed_p50":   median(elapsed) if elapsed else 0,
            "elapsed_p95":   _pct(elapsed, 0.95) if elapsed else 0,
            "elapsed_mean":  mean(elapsed) if elapsed else 0,
            "prompt_mean":   mean(p_tok) if p_tok else 0,
            "response_mean": mean(r_tok) if r_tok else 0,
            "gen_tps_mean":  mean(gen_tps) if gen_tps else 0,
            "ttft_p50":      median(ttft) if ttft else None,
            "tpot_p50":      median(tpot) if tpot else None,
        }
    return out


def print_llm_table(summary: dict) -> None:
    print("\n=== Per-model LLM call summary ==========================================\n")
    header = (f"{'endpoint':16s} {'model':16s} "
              f"{'n':>4s} {'ret%':>5s} "
              f"{'p50_s':>7s} {'p95_s':>7s} "
              f"{'in_tok':>8s} {'out_tok':>8s} "
              f"{'tps':>7s} "
              f"{'ttft_s':>7s} {'tpot_ms':>8s}")
    print(header)
    print("-" * len(header))
    for (ep, model), s in sorted(summary.items()):
        print(f"{ep:16s} {model:16s} "
              f"{s['n_calls']:>4d} "
              f"{s['retry_rate']*100:>4.1f}% "
              f"{s['elapsed_p50']:>7.2f} {s['elapsed_p95']:>7.2f} "
              f"{s['prompt_mean']:>8.0f} {s['response_mean']:>8.0f} "
              f"{s['gen_tps_mean']:>7.1f} "
              f"{_fmt(s['ttft_p50'], 7)} {_fmt(s['tpot_p50'], 8)}")


# ─── Table 2: per-agent query summary ─────────────────────────────────────

def summarise_queries(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("endpoint") != "query":
            continue
        agent = r.get("agent") or "(unrouted)"
        groups[agent].append(r)
    out: dict[str, dict] = {}
    for agent, rs in groups.items():
        wall     = [r["wall_clock_s"] for r in rs if r.get("wall_clock_s") is not None]
        llm      = [r["sum_llm_elapsed_s"] for r in rs if r.get("sum_llm_elapsed_s") is not None]
        tool     = [r["tool_elapsed_s"] for r in rs if r.get("tool_elapsed_s") is not None]
        n_calls  = [r["n_llm_calls"] for r in rs if r.get("n_llm_calls") is not None]
        n_tools  = [r["n_tools"] for r in rs if r.get("n_tools") is not None]
        p_tok    = [r["llm_prompt_tok"] for r in rs if r.get("llm_prompt_tok")]
        r_tok    = [r["llm_response_tok"] for r in rs if r.get("llm_response_tok")]
        out[agent] = {
            "n":              len(rs),
            "wall_p50":       median(wall) if wall else 0,
            "wall_p95":       _pct(wall, 0.95) if wall else 0,
            "llm_p50":        median(llm) if llm else 0,
            "tool_p50":       median(tool) if tool else 0,
            "calls_median":   median(n_calls) if n_calls else 0,
            "tools_median":   median(n_tools) if n_tools else 0,
            "prompt_mean":    mean(p_tok) if p_tok else 0,
            "response_mean":  mean(r_tok) if r_tok else 0,
        }
    return out


def print_query_table(summary: dict) -> None:
    print("\n=== Per-agent user-query summary ========================================\n")
    header = (f"{'agent':16s} "
              f"{'n':>4s} "
              f"{'wall_p50':>9s} {'wall_p95':>9s} "
              f"{'llm_p50':>8s} {'tool_p50':>9s} "
              f"{'calls':>6s} {'tools':>6s} "
              f"{'in_tok':>8s} {'out_tok':>8s}")
    print(header)
    print("-" * len(header))
    for agent, s in sorted(summary.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{agent:16s} "
              f"{s['n']:>4d} "
              f"{s['wall_p50']:>9.2f} {s['wall_p95']:>9.2f} "
              f"{s['llm_p50']:>8.2f} {s['tool_p50']:>9.2f} "
              f"{s['calls_median']:>6.1f} {s['tools_median']:>6.1f} "
              f"{s['prompt_mean']:>8.0f} {s['response_mean']:>8.0f}")


# ─── Table 3: user-perceived wall-clock percentiles ───────────────────────

def print_percentiles(rows: list[dict]) -> None:
    wall = [r["wall_clock_s"] for r in rows
            if r.get("endpoint") == "query" and r.get("wall_clock_s") is not None]
    if not wall:
        return
    print("\n=== User-perceived wall-clock (all queries) =============================\n")
    for p, label in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95"),
                     (0.99, "p99"), (1.0, "max")]:
        v = _pct(wall, p) if p < 1.0 else max(wall)
        print(f"  {label:>4s}   {v:>7.2f} s")
    print(f"  {'mean':>4s}   {mean(wall):>7.2f} s   ({len(wall)} queries)")


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG,
                    help=f"JSONL log path (default: {DEFAULT_LOG})")
    ap.add_argument("--since", type=str, default=None,
                    help="only rows with iso_ts >= this ISO date (e.g. 2026-07-15)")
    ap.add_argument("--model", type=str, default=None,
                    help="filter to a single model (still shows unrelated query rows)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of tables")
    args = ap.parse_args()

    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"error: --since must be ISO format, got {args.since!r}",
                  file=sys.stderr)
            sys.exit(2)

    rows      = load(args.log, since_dt, args.model)
    llm_sum   = summarise_llm(rows)
    query_sum = summarise_queries(rows)

    if args.json:
        # Keys need to be str for JSON, so flatten (endpoint, model) tuples
        json.dump({
            "log":        str(args.log),
            "n_rows":     len(rows),
            "per_model":  {f"{ep}::{m}": v for (ep, m), v in llm_sum.items()},
            "per_agent":  query_sum,
        }, sys.stdout, indent=2, default=str)
        print()
        return

    print(f"Log: {args.log}  ({len(rows)} rows)")
    print_llm_table(llm_sum)
    print_query_table(query_sum)
    print_percentiles(rows)


if __name__ == "__main__":
    main()
