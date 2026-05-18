#!/usr/bin/env python3
"""Wrapper script for midas_stress operations.

Called via subprocess from APEXA MCP tools.
Outputs JSON to stdout, diagnostics to stderr.
"""

import argparse
import json
import sys

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size <= 36:
                return obj.tolist()
            return {
                "shape": list(obj.shape),
                "mean": float(np.nanmean(obj)),
                "std": float(np.nanstd(obj)),
                "min": float(np.nanmin(obj)),
                "max": float(np.nanmax(obj)),
            }
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _output(data):
    json.dump(data, sys.stdout, cls=NumpyEncoder, indent=2)
    sys.stdout.write("\n")


def _error(msg):
    _output({"status": "error", "error": msg})
    sys.exit(1)


def _parse_vector(s, n=None):
    vals = [float(x) for x in s.split(",")]
    if n is not None and len(vals) != n:
        _error(f"Expected {n} comma-separated values, got {len(vals)}: {s}")
    return np.array(vals)


def _array_stats(arr, name=""):
    if arr is None or arr.size == 0:
        return None
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "median": float(np.nanmedian(arr)),
    }


def _percentiles(arr):
    if arr is None or arr.size == 0:
        return None
    return {
        "p5": float(np.nanpercentile(arr, 5)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p95": float(np.nanpercentile(arr, 95)),
    }


# ── Subcommands ─────────────────────────────────────────────────────────


def cmd_read_grains(args):
    import midas_stress as ms

    grains = ms.read_grains(args.grains)
    n_grains = grains["orientations"].shape[0] if "orientations" in grains else 0

    result = {
        "status": "success",
        "grains_file": args.grains,
        "n_grains": n_grains,
    }

    if "positions" in grains:
        pos = grains["positions"]
        result["positions"] = {
            "X": _array_stats(pos[:, 0]),
            "Y": _array_stats(pos[:, 1]),
            "Z": _array_stats(pos[:, 2]),
        }

    if "strain" in grains:
        diag = np.array([grains["strain"][:, i, i] for i in range(3)])
        result["strain_diagonal"] = {
            f"e{i+1}{i+1}": _array_stats(diag[i]) for i in range(3)
        }
        result["strain_magnitude"] = _array_stats(
            np.linalg.norm(grains["strain"].reshape(n_grains, -1), axis=1)
        )

    if "lattice_params" in grains:
        lp = grains["lattice_params"]
        names = ["a", "b", "c", "alpha", "beta", "gamma"]
        result["lattice_params"] = {
            names[i]: _array_stats(lp[:, i]) for i in range(min(6, lp.shape[1]))
        }

    if "confidences" in grains:
        result["confidences"] = _array_stats(grains["confidences"])
        result["confidences"]["percentiles"] = _percentiles(grains["confidences"])

    if "radii" in grains:
        result["radii"] = _array_stats(grains["radii"])

    if "euler_angles" in grains:
        ea = np.degrees(grains["euler_angles"])
        result["euler_angles_deg"] = {
            "phi1": _array_stats(ea[:, 0]),
            "Phi": _array_stats(ea[:, 1]),
            "phi2": _array_stats(ea[:, 2]),
        }

    _output(result)


def cmd_compute_stress(args):
    import midas_stress as ms

    grains = ms.read_grains(args.grains)
    n_grains = grains["orientations"].shape[0]

    stiffness = ms.get_stiffness(args.material)

    applied = None
    if args.applied_stress:
        v = _parse_vector(args.applied_stress, 6)
        applied = ms.voigt_to_tensor(v)

    result_data = ms.compute_stress(
        strain=grains["strain"],
        stiffness=stiffness,
        orient=grains["orientations"],
        volumes=grains.get("radii", np.ones(n_grains)) ** 3,
        confidences=grains.get("confidences"),
        applied_stress=applied,
        min_confidence=args.min_confidence,
        frame="lab",
    )

    vm = result_data["von_mises"]
    hydro = result_data["hydrostatic_corrected"]

    result = {
        "status": "success",
        "grains_file": args.grains,
        "material": args.material,
        "n_grains": n_grains,
        "von_mises": _array_stats(vm),
        "von_mises_percentiles": _percentiles(vm),
        "hydrostatic_corrected": _array_stats(hydro),
        "hydrostatic_raw": _array_stats(result_data["hydrostatic_raw"]),
        "hydrostatic_shift": float(result_data["hydrostatic_shift"]),
    }

    unc = result_data.get("uncertainty", {})
    if unc:
        result["uncertainty"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in unc.items()
            if not isinstance(v, np.ndarray) or v.size <= 6
        }

    _output(result)


def cmd_material_info(args):
    import midas_stress as ms

    stiffness = ms.get_stiffness(args.material)
    lib_entry = ms.STIFFNESS_LIBRARY[args.material]
    sens = ms.d0_sensitivity(material=args.material)

    result = {
        "status": "success",
        "material": args.material,
        "symmetry": lib_entry["symmetry"],
        "elastic_constants": {
            k: v for k, v in lib_entry.items() if k != "symmetry"
        },
        "stiffness_matrix_GPa": stiffness.tolist(),
        "d0_sensitivity": {
            "bulk_modulus_GPa": sens["bulk_modulus_GPa"],
            "MPa_per_ppm": sens["sensitivity_MPa_per_ppm"],
            "MPa_per_100ppm": sens["sensitivity_MPa_per_100ppm"],
            "MPa_per_1000ppm": sens["sensitivity_MPa_per_1000ppm"],
            "is_pure_hydrostatic": sens["is_pure_hydrostatic"],
            "hydrostatic_fraction": sens["hydrostatic_fraction"],
        },
        "available_materials": ms.list_materials(),
    }

    _output(result)


def cmd_correct_d0(args):
    import midas_stress as ms

    grains = ms.read_grains(args.grains)
    n_grains = grains["orientations"].shape[0]
    stiffness = ms.get_stiffness(args.material)

    applied = None
    if args.applied_stress:
        v = _parse_vector(args.applied_stress, 6)
        applied = ms.voigt_to_tensor(v)

    volumes = grains.get("radii", np.ones(n_grains)) ** 3
    confidences = grains.get("confidences")

    correction = ms.correct_d0(
        strains=grains["strain"],
        stiffness=stiffness,
        orientations=grains["orientations"],
        volumes=volumes,
        confidences=confidences,
        applied_stress=applied,
        min_confidence=args.min_confidence,
    )

    result = {
        "status": "success",
        "grains_file": args.grains,
        "material": args.material,
        "n_grains": n_grains,
        "eps_iso_ppm": float(correction["eps_iso"]) * 1e6,
        "residual_norm_before": float(correction["residual_norm_before"]),
        "residual_norm_after": float(correction["residual_norm_after"]),
        "improvement_factor": (
            float(correction["residual_norm_before"] / correction["residual_norm_after"])
            if correction["residual_norm_after"] > 0 else float("inf")
        ),
    }

    unc = correction.get("uncertainty", {})
    if unc:
        result["uncertainty"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in unc.items()
            if not isinstance(v, np.ndarray) or v.size <= 6
        }

    vm_corrected = ms.von_mises(correction["stresses_corrected"])
    hydro_corrected = ms.hydrostatic(correction["stresses_corrected"])
    result["stress_after_correction"] = {
        "von_mises": _array_stats(vm_corrected),
        "hydrostatic": _array_stats(hydro_corrected),
    }

    _output(result)


def cmd_plasticity(args):
    import midas_stress as ms

    grains = ms.read_grains(args.grains)
    n_grains = grains["orientations"].shape[0]
    stiffness = ms.get_stiffness(args.material)
    volumes = grains.get("radii", np.ones(n_grains)) ** 3

    stress_result = ms.compute_stress(
        strain=grains["strain"],
        stiffness=stiffness,
        orient=grains["orientations"],
        volumes=volumes,
        confidences=grains.get("confidences"),
        frame="lab",
    )

    stress = stress_result["stress_corrected"]
    orient = grains["orientations"]
    load_dir = _parse_vector(args.load_direction, 3)

    n_crystal, b_crystal = ms.get_slip_systems_for_material(args.material)
    n_systems = n_crystal.shape[0]

    sf = ms.schmid_factor(orient, load_dir, n_crystal, b_crystal)
    dom = ms.dominant_slip_system(
        orient, n_crystal, b_crystal, stress=stress, top_k=3,
    )
    tf = ms.taylor_factor(orient, load_dir, n_crystal, b_crystal, volumes=volumes)

    result = {
        "status": "success",
        "grains_file": args.grains,
        "material": args.material,
        "n_grains": n_grains,
        "n_slip_systems": n_systems,
        "load_direction": load_dir.tolist(),
        "schmid_factor": {
            "max_per_grain": _array_stats(sf.max(axis=1)),
            "mean_per_grain": _array_stats(sf.mean(axis=1)),
            "overall_max": float(sf.max()),
        },
        "taylor_factor": {
            "M_poly": tf["M_poly"],
            "M_uniform": tf["M_uniform"],
            "M_per_grain": _array_stats(tf["M_per_grain"]),
        },
        "dominant_system": {
            "best_score": _array_stats(dom["best_score"]),
            "best_score_percentiles": _percentiles(dom["best_score"]),
        },
    }

    if args.crss > 0:
        yp = ms.yield_proximity(stress, orient, n_crystal, b_crystal, args.crss)
        result["yield_proximity"] = {
            "crss_MPa": args.crss,
            "proximity": _array_stats(yp["proximity"]),
            "proximity_percentiles": _percentiles(yp["proximity"]),
            "n_yielded": int(yp["yielded"].sum()),
            "fraction_yielded": float(yp["yielded"].mean()),
        }

    _output(result)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="APEXA stress-analysis wrapper")
    sub = parser.add_subparsers(dest="command")

    p_read = sub.add_parser("read_grains")
    p_read.add_argument("--grains", required=True)

    p_stress = sub.add_parser("compute_stress")
    p_stress.add_argument("--grains", required=True)
    p_stress.add_argument("--material", required=True)
    p_stress.add_argument("--applied-stress", default="")
    p_stress.add_argument("--min-confidence", type=float, default=0.0)

    p_mat = sub.add_parser("material_info")
    p_mat.add_argument("--material", required=True)

    p_d0 = sub.add_parser("correct_d0")
    p_d0.add_argument("--grains", required=True)
    p_d0.add_argument("--material", required=True)
    p_d0.add_argument("--applied-stress", default="")
    p_d0.add_argument("--min-confidence", type=float, default=0.0)

    p_plast = sub.add_parser("plasticity")
    p_plast.add_argument("--grains", required=True)
    p_plast.add_argument("--material", required=True)
    p_plast.add_argument("--load-direction", default="0,0,1")
    p_plast.add_argument("--crss", type=float, default=0.0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        {
            "read_grains": cmd_read_grains,
            "compute_stress": cmd_compute_stress,
            "material_info": cmd_material_info,
            "correct_d0": cmd_correct_d0,
            "plasticity": cmd_plasticity,
        }[args.command](args)
    except FileNotFoundError as e:
        _error(str(e))
    except ValueError as e:
        _error(str(e))
    except ImportError:
        _error(
            "midas_stress not installed. Run: "
            "pip install -e $MIDAS_PATH/packages/midas_stress"
        )
    except Exception as e:
        _error(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
