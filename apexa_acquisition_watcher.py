"""APEXA acquisition watcher (prototype) — turn APEXA from post-hoc to acquisition-driven.

This closes the gap called out when comparing APEXA to the reflectometry
``nr-analyzer``/AuRE stack: those tools (and APEXA today) only act on data that
already exists. The reflectometry *planner* nonetheless contains the key trick —
it groups files by sample "stem", knows the EXPECTED set for the technique, and
only fires when the set is complete:

    "Sequence/stem 'Ni58_stitched' is complete because both required PNR spin
     states (_u, _d) are present ... this arriving file is the last missing
     spin state for the complete stem."

This module generalises that to synchrotron series (FF/PF-HEDM, WAXS, SAXS):

  1. **Group** files in a directory by stem (strip frame index / spin / segment
     suffixes and extension).
  2. **Judge completeness** — either by an explicit expected frame count, or by a
     "quiet period" (no new frames for N seconds → the scan has finished).
  3. **Plan + fire** — build an :class:`apexa_plan.APEXAPlan`, record every
     inferred default as an assumption, write ``APEXA_plan.yaml`` next to the
     data, and (in --execute mode) call ``midas_integrate_series`` once for the
     whole stem. Each stem fires at most once.

It is a PROTOTYPE: polling (no watchdog dependency), heuristic completeness, and
integration-only execution. It is intentionally standalone — stdlib + PyYAML +
apexa_plan — so it runs on a beamline data mover without the agent stack, and
only lazy-imports the MCP server when actually executing.

Usage
-----
    # dry-run: watch a live scan dir, print the plan it WOULD run
    python apexa_acquisition_watcher.py /scratch/beam/mytscan \\
        --technique waxs --param-file ceria_params.txt --expected-count 180

    # one pass over an already-finished directory, then exit
    python apexa_acquisition_watcher.py /data/done --once --param-file p.txt

    # actually integrate each completed stem
    python apexa_acquisition_watcher.py /scratch/beam/live --execute --param-file p.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from glob import glob

from apexa_plan import (APEXAPlan, Beam, Calibration, DarkSpec, DataSpec,
                        Sample)

# ── stem grouping ────────────────────────────────────────────────────────────
# Suffixes that denote a member of a set rather than a distinct sample.
_SPIN_SEG = re.compile(
    r"(?:"
    r"_(?:uu|ud|du|dd|u|d)"        # polarised-neutron spin states (refl carry-over)
    r"|_partial"                    # co-refined angle segments
    r"|_combined(?:_data)?(?:_auto)?"
    r")$",
    re.IGNORECASE,
)
# A trailing frame/scan index like _00001, .00042, -12 (optionally before ext).
_FRAME_IDX = re.compile(r"[._-]\d{1,6}$")
# Compound extensions to strip whole (order matters: longest first).
_EXTS = (".vrx.h5", ".ge5", ".ge3", ".ge2", ".tif", ".tiff", ".h5",
         ".hdf5", ".dat", ".txt", ".edf", ".cbf", ".raw")


def stem_of(filename: str) -> str:
    """Reduce a data filename to the sample/scan stem that groups its set.

    JL_0Nb_00042.vrx.h5       -> JL_0Nb          (trailing _index stripped)
    Ni58_stitched_u.dat       -> Ni58_stitched   (spin state stripped)
    REFL_226642_3_partial.txt -> REFL_226642     (segment + index stripped;
                                                  co-refined partials group as one)
    scan007.ge5               -> scan007         (no separator before the index,
                                                  so it is left intact)

    Heuristic, not infallible: a trailing number is treated as a frame index only
    when a separator (``_ . -``) precedes it. For unusual naming, pass a stricter
    ``--pattern`` or an ``--expected-count`` rather than relying on grouping.
    """
    name = os.path.basename(filename)
    low = name.lower()
    for ext in _EXTS:
        if low.endswith(ext):
            name = name[: -len(ext)]
            break
    else:
        name = os.path.splitext(name)[0]
    # peel one spin/segment suffix, then one frame index (loop to catch _3_partial)
    changed = True
    while changed:
        changed = False
        new = _SPIN_SEG.sub("", name)
        if new != name:
            name, changed = new, True
    name = _FRAME_IDX.sub("", name)
    return name or os.path.basename(filename)


def is_dark(filename: str, exclude_substring: str = "dark") -> bool:
    return exclude_substring and exclude_substring.lower() in os.path.basename(filename).lower()


# ── grouping + completeness ──────────────────────────────────────────────────
def group_by_stem(files, exclude_substring="dark"):
    """{stem: [paths...]} for non-dark files."""
    groups: dict[str, list] = {}
    for f in files:
        if is_dark(f, exclude_substring):
            continue
        groups.setdefault(stem_of(f), []).append(f)
    for v in groups.values():
        v.sort()
    return groups


def is_complete(paths, *, expected_count=None, quiet_seconds=30.0, now=None):
    """Decide whether a stem's set looks finished.

    Returns (complete: bool, reason: str) — reason feeds the plan's assumptions.
    """
    n = len(paths)
    if expected_count is not None:
        if n >= expected_count:
            return True, f"{n} frames present >= expected_count {expected_count}"
        return False, f"{n}/{expected_count} frames so far"
    # No count known: fall back to a quiet period on the newest file's mtime.
    now = time.time() if now is None else now
    try:
        newest = max(os.path.getmtime(p) for p in paths)
    except OSError:
        return False, "files vanished during scan"
    idle = now - newest
    if idle >= quiet_seconds:
        return True, f"no new frames for {idle:.0f}s (>= quiet {quiet_seconds:.0f}s); {n} frames"
    return False, f"still arriving ({idle:.0f}s idle < {quiet_seconds:.0f}s); {n} frames"


# ── plan construction ────────────────────────────────────────────────────────
def build_plan(stem, paths, args) -> APEXAPlan:
    """Assemble a reviewable plan for one completed stem, logging assumptions."""
    image_dir = os.path.dirname(paths[0]) if paths else args.directory
    plan = APEXAPlan(
        technique=args.technique,
        instrument=args.instrument or "",
        describe=f"{args.technique} of stem '{stem}' ({len(paths)} frames), "
                 f"triggered on acquisition completion.",
        sample=Sample(name=stem),
        beam=Beam(energy_keV=args.energy),
        calibration=Calibration(calibrant=args.calibrant or "",
                                parameter_file=args.param_file or ""),
        data=DataSpec(
            image_dir=image_dir,
            pattern=f"{stem}*",
            stem=stem,
            data_location=args.data_location or "",
            expected_count=args.expected_count,
        ),
        dark=DarkSpec(source=args.dark_source, kind=args.dark_kind,
                      location=args.dark_location or ""),
        result_folder=args.result_folder or os.path.join(image_dir, f"APEXA_{stem}"),
        perform_execution=bool(args.execute),
    )
    # provenance for every value we inferred rather than were told
    if args.expected_count is None:
        plan.assume("data.expected_count", None,
                    "no count given; completeness judged by acquisition quiet-period")
    if not args.data_location:
        plan.assume("data.data_location", "",
                    "HDF5 dataset path not specified; integrator default assumed")
    if args.energy is None:
        plan.assume("beam.energy_keV", None,
                    "energy not supplied on CLI; must come from parameter_file")
    plan.assume("dark.kind", args.dark_kind,
                f"dark timing not specified per-stem; using CLI default '{args.dark_kind}'")
    return plan


# ── execution seam ───────────────────────────────────────────────────────────
# Only the azimuthal-integration family is safe to auto-fire from a watcher: it is
# a single, idempotency-guarded tool call (midas_integrate_series carries the
# @idempotent guard, so a stem fired twice replays the prior result rather than
# re-integrating). HEDM setup (calibrate → in-plane tx → reconstruct) is a *gated*
# procedure — it needs the human-in-the-loop decisions the FF-HEDM LangGraph
# workflow enforces (APEXA_WORKFLOW_MODE=graph). A watcher cannot answer those
# gates, so for HEDM techniques we emit the reviewable plan and hand off, never
# fire a blind integrate call (which would be the wrong tool entirely).
_AUTO_EXECUTABLE = ("waxs", "saxs", "integration")


def execute_plan(plan: APEXAPlan):
    """Run the plan via midas_integrate_series (lazy import; async).

    Returns the tool result. Raises ValueError for a technique the watcher must
    not auto-execute (HEDM/calibration) — the caller handles the handoff.
    """
    if plan.technique not in _AUTO_EXECUTABLE:
        raise ValueError(
            f"technique '{plan.technique}' is not watcher-auto-executable; "
            f"HEDM/calibration need the gated FF-HEDM graph (APEXA_WORKFLOW_MODE=graph)")
    import asyncio
    from midas_comprehensive_server import midas_integrate_series  # heavy; lazy
    kwargs = plan.to_integrate_series_kwargs()
    return asyncio.run(midas_integrate_series(**kwargs))


def _is_cached_result(out) -> bool:
    """True if a tool result reports the idempotency guard replayed a prior run."""
    if isinstance(out, dict):
        return bool(out.get("cached"))
    if isinstance(out, str):
        # tool results are JSON strings; cheap substring check avoids a full parse
        return '"cached": true' in out.lower()
    return False


# ── watch loop ───────────────────────────────────────────────────────────────
def scan_once(args, fired: set):
    files = glob(os.path.join(args.directory, "**", args.pattern), recursive=True)
    groups = group_by_stem(files, args.exclude_substring)
    for stem, paths in sorted(groups.items()):
        if stem in fired:
            continue
        complete, reason = is_complete(
            paths, expected_count=args.expected_count,
            quiet_seconds=args.quiet_seconds)
        status = "READY" if complete else "wait "
        print(f"  [{status}] {stem:<24} {len(paths):>4} frames — {reason}")
        if not complete:
            continue
        plan = build_plan(stem, paths, args)
        plan.assume("_trigger", reason, "completeness rule that fired this stem")
        plan_path = os.path.join(plan.data.image_dir, f"APEXA_plan_{stem}.yaml")
        plan.to_yaml(plan_path)
        issues = plan.validate()
        print(f"     -> wrote {plan_path}")
        if issues:
            print("     -> NOT executing; plan has issues:")
            for i in issues:
                print(f"          - {i}")
            fired.add(stem)
            continue
        if args.technique not in _AUTO_EXECUTABLE:
            # HEDM/calibration: the plan is the deliverable; a human runs the
            # gated FF-HEDM graph. Never auto-fire integrate for these.
            print(f"     -> technique '{args.technique}' needs the gated FF-HEDM graph "
                  "(human-in-the-loop). Plan written for review; run it with "
                  "APEXA_WORKFLOW_MODE=graph and ask APEXA to reconstruct this stem.")
        elif args.execute:
            print(f"     -> executing midas_integrate_series for '{stem}' ...")
            try:
                out = execute_plan(plan)
                if _is_cached_result(out):
                    print(f"     -> idempotent skip: '{stem}' already integrated "
                          "(prior result replayed; pass force=True to redo).")
                else:
                    print(f"     -> done: {str(out)[:200]}")
            except Exception as exc:
                print(f"     -> execution FAILED: {exc}")
        else:
            print("     -> dry-run (pass --execute to integrate). Would call:")
            for k, v in plan.to_integrate_series_kwargs().items():
                print(f"          {k}={v!r}")
        fired.add(stem)
    return fired


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="acquisition directory to watch")
    ap.add_argument("--pattern", default="*.h5", help="glob for data files (default *.h5)")
    ap.add_argument("--exclude-substring", default="dark", help="skip files containing this")
    ap.add_argument("--technique", default="waxs", help="ff-hedm|pf-hedm|waxs|saxs|integration")
    ap.add_argument("--instrument", default="")
    ap.add_argument("--param-file", default="", help="MIDAS parameter file (calibration)")
    ap.add_argument("--calibrant", default="")
    ap.add_argument("--energy", type=float, default=None, help="beam energy (keV)")
    ap.add_argument("--data-location", default="", help="HDF5 dataset path, e.g. exchange/data")
    ap.add_argument("--dark-source", default="file", choices=["file", "embedded", "none"])
    ap.add_argument("--dark-kind", default="after", choices=["after", "before", "any"])
    ap.add_argument("--dark-location", default="")
    ap.add_argument("--result-folder", default="")
    ap.add_argument("--expected-count", type=int, default=None,
                    help="frames for a COMPLETE set; omit to use quiet-period rule")
    ap.add_argument("--quiet-seconds", type=float, default=30.0,
                    help="idle time that marks a scan finished when no count given")
    ap.add_argument("--interval", type=float, default=10.0, help="poll interval (s)")
    ap.add_argument("--once", action="store_true", help="one pass then exit")
    ap.add_argument("--execute", action="store_true",
                    help="actually run integration (default: dry-run)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.directory):
        ap.error(f"not a directory: {args.directory}")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"APEXA acquisition watcher [{mode}] — {args.directory} "
          f"(pattern {args.pattern}, technique {args.technique})")
    fired: set = set()
    try:
        while True:
            print(f"-- scan @ {time.strftime('%H:%M:%S')} --")
            fired = scan_once(args, fired)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
