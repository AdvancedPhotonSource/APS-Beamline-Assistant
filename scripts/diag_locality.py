#!/usr/bin/env python3
"""Print exactly how APEXA decides run-locality for a set of paths, ON THIS HOST.

Run this on the machine where APEXA is deployed (e.g. copland) to see whether the
remote-exec registry + hostname identity route a run local or remote, and whether
the run_ff_hedm_full_workflow co-location guard would fire.

Usage:
  uv run python scripts/diag_locality.py [PARAM DATA RESULT_FOLDER]

With no args it uses the pokharel_jul26 Au FF paths from the live session.
Reads local config/env only; the registry probe may SSH if a path misses locally.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import apexa_remote_exec as r  # noqa: E402

_DEF = [
    "/gdata/dm/1ID/2026/pokharel_jul26/analysis/APEXA-analysis/Parameters_Au_fresh_fffix.txt",
    "/gdata/dm/1ID/2026/pokharel_jul26/analysis/APEXA-analysis/recon_fresh_fffix_clean/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip",
    "/gdata/dm/1ID/2026/pokharel_jul26/analysis/APEXA-analysis/harness-comparison/APEXA-FF",
]


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 3:
        param, data, rf = args
    else:
        param, data, rf = _DEF

    print("── host identity ──────────────────────────────────────────────")
    print(f"socket.gethostname() = {socket.gethostname()!r}")
    print(f"socket.getfqdn()     = {socket.getfqdn()!r}")
    print(f"local_hostnames()    = {sorted(r.local_hostnames())}")
    print(f"APEXA_LOCAL_HOSTNAMES= {os.environ.get('APEXA_LOCAL_HOSTNAMES','')!r}")

    print("\n── remote-exec registry ───────────────────────────────────────")
    reg = r.load_remote_registry()
    print(f"default_host = {reg.get('default_host')!r}")
    for name, rec in reg["hosts"].items():
        print(f"  host key {name!r}: host={rec.get('host')!r} "
              f"data_roots={rec.get('data_roots')} "
              f"activate={rec.get('activate','')!r}")
    for h in {reg.get("default_host"), *(rec.get("host") or k
              for k, rec in reg["hosts"].items())}:
        if h:
            print(f"  is_local_host({h!r}) = {r.is_local_host(h)}")

    print("\n── per-path registry resolution ────────────────────────────────")
    for label, p in (("param", param), ("data", data), ("result_folder", rf)):
        rh = r.resolve_host_for_path(p)
        rr = bool(rh) and not r.is_local_host(rh)
        print(f"  {label:<14} resolve_host={rh!r:<12} "
              f"is_remote_host={rr}  exists_local={os.path.exists(os.path.expanduser(p))}")
        print(f"                 {p}")

    print("\n── decide_exec_host(param, data) ───────────────────────────────")
    dec = r.decide_exec_host(param, data)
    for k in ("is_remote", "host", "reason", "unreachable", "error"):
        if k in dec:
            print(f"  {k} = {dec.get(k)!r}")

    print("\n── run_ff_hedm_full_workflow co-location guard ─────────────────")
    is_remote = dec.get("is_remote", False)
    rf_host = r.resolve_host_for_path(rf)
    rf_remote = bool(rf_host) and not r.is_local_host(rf_host)
    old_block = (not is_remote and bool(rf_host))
    new_block = (not is_remote and rf_remote)
    print(f"  is_remote={is_remote}  rf_host={rf_host!r}  rf_remote={rf_remote}")
    print(f"  OLD guard would block = {old_block}")
    print(f"  NEW guard would block = {new_block}")
    if old_block and not new_block:
        print("  => this is the fixed bug: local run was being false-blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
