#!/usr/bin/env python3
"""Wrapper for the 8 new MIDAS v0.1.0 capability packages (pdf, defect, dfxm,
xaf, 2d/ultrafast, grain-odf, pf-odf, pink).

Called via subprocess from APEXA MCP tools (midas_comprehensive_server.py) under
the APEXA .venv interpreter with a CLEAN env — the pip torch stack these libs
depend on breaks under the C++ DYLD/LD injection get_midas_env() applies. Each
subcommand imports its package lazily, so one broken/absent lib fails only its
own path. Outputs a single JSON object to stdout; diagnostics to stderr.

Maturity tiers (honesty — surfaced in every payload as mode/real_data_supported):
  real       — genuine data-driven analysis (pdf: I(Q)->G(r); defect/grain-odf CLIs)
  synthetic  — forward model / demo on generated inputs (2d, xaf, dfxm)
  deferred   — capability present but real-data ingestion / synthetic self-test
               pending upstream (pf-odf, pink) — guarded attempt, honest fallback
"""

import argparse
import json
import os
import subprocess
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


# Reserve the real stdout for our single JSON payload. During dispatch we point
# sys.stdout at stderr so any chatter a package prints (tutorials do `print(...)`)
# can't corrupt the JSON the MCP server parses off stdout.
_REAL_STDOUT = sys.stdout


def _output(data):
    json.dump(data, _REAL_STDOUT, cls=NumpyEncoder, indent=2)
    _REAL_STDOUT.write("\n")
    _REAL_STDOUT.flush()


def _error(msg, **extra):
    _output({"status": "error", "error": msg, **extra})
    sys.exit(1)


def _array_stats(arr, name=""):
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "median": float(np.nanmedian(arr)),
    }


def _load_1d(path):
    """Load a 1D pattern (2 columns: x, intensity) from .xy/.xye/.dat/.csv/.txt.

    Returns (x, y). Ignores comment/header lines and trailing error columns.
    """
    x, y = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;/%" or s.lower().startswith(("x", "q", "2", "tth", "angle")):
                # allow a numeric first token even if line begins with '2theta' header
                try:
                    float(s.split()[0])
                except (ValueError, IndexError):
                    continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
            except ValueError:
                continue
    if not x:
        raise ValueError(f"no 2-column numeric data found in {path}")
    return np.asarray(x), np.asarray(y)


# ── pdf: I(Q) -> G(r) pair distribution (REAL data) ───────────────────────

def cmd_pdf(args):
    import midas_pdf as mp

    x, intensity = _load_1d(args.pattern)
    # Convert 2θ (deg) to Q if requested; else assume column is already Q (Å⁻¹).
    if args.x_is_two_theta:
        if not args.wavelength:
            _error("wavelength required to convert 2theta -> Q")
        tth = np.radians(x)
        q = 4.0 * np.pi * np.sin(tth / 2.0) / float(args.wavelength)
    else:
        q = x
    fractions = {}
    for tok in args.composition.split(","):
        tok = tok.strip()
        if not tok:
            continue
        el, _, frac = tok.partition(":")
        fractions[el.strip()] = float(frac) if frac else 1.0
    if not fractions:
        _error("composition required, e.g. 'Ni:1' or 'Ce:1,O:2'")
    comp = mp.Composition(fractions)
    r_grid = np.linspace(float(args.r_min), float(args.r_max), int(args.n_r))
    # i_of_q_to_Gr returns (G, sigma_G, S) — G(r), its 1σ band, and S(Q).
    out = mp.i_of_q_to_Gr(
        q, intensity, comp, r_grid, wavelength_A=float(args.wavelength or 0.1),
        q_max=(float(args.q_max) if args.q_max else None),
        window=args.window, return_S=True)
    G_r = np.asarray(out[0])
    sigma_G = np.asarray(out[1]) if out[1] is not None else None
    S_q = np.asarray(out[2]) if len(out) > 2 and out[2] is not None else None
    result = {
        "status": "success", "mode": "real", "real_data_supported": True,
        "package": "midas_pdf", "version": getattr(mp, "__version__", "?"),
        "pattern": args.pattern, "composition": fractions,
        "n_q": int(len(q)), "q_range": [float(np.min(q)), float(np.max(q))],
        "window": args.window,
        "r_grid": [float(args.r_min), float(args.r_max), int(args.n_r)],
        "G_r_stats": _array_stats(G_r),
        "sigma_G_stats": _array_stats(sigma_G) if sigma_G is not None else None,
        "S_q_stats": _array_stats(S_q) if S_q is not None else None,
    }
    if args.out:
        arr = np.column_stack([r_grid, G_r])
        np.savetxt(args.out, arr, header="r(A)  G(r)", fmt="%.6g")
        result["output_file"] = args.out
    # first peak (nearest-neighbour distance): argmax of G(r) for r>0.5 Å
    mask = r_grid > 0.5
    if mask.any():
        idx = int(np.nanargmax(G_r[mask]))
        result["first_peak_r_A"] = float(r_grid[mask][idx])
    _output(result)


# ── defect: dislocation density / diffuse-scattering rods (REAL via CLI) ───

def cmd_defect(args):
    import midas_defect  # noqa: F401  (import-guard only; work is via CLI)

    # midas_defect is genuinely real-data-only: every console script (rods /
    # asterism / polytype / inventory) needs a q-space voxel NPZ from FF-HEDM
    # diffuse scattering (inventory additionally needs the indexed Grains.csv).
    # There is no zero-input demo — so fail honestly when inputs are absent.
    if not args.voxels:
        _error("midas_defect needs a q-space voxel NPZ (FF-HEDM diffuse "
               "scattering); none given.",
               mode="real", real_data_supported=True, package="midas_defect",
               required=["voxels(NPZ)"] + (["grains(Grains.csv)"]
                                           if args.mode == "inventory" else []),
               modes=["rods", "asterism", "polytype", "inventory"])
    script = {
        "rods": "midas-defect-rods",
        "asterism": "midas-defect-asterism",
        "polytype": "midas-defect-polytype",
        "inventory": "midas-defect-inventory",
    }.get(args.mode, "midas-defect-rods")
    if args.mode == "inventory":
        if not args.grains:
            _error("inventory mode also needs the indexed Grains.csv (--grains).",
                   mode="real", real_data_supported=True, package="midas_defect")
        cmd = [script, "--voxels", args.voxels, "--grains", args.grains]
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        cmd = [script, "--voxels", args.voxels, "--out-dir", args.out_dir]
        if args.no_html:
            cmd.append("--no-html")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    _output({
        "status": "success" if p.returncode == 0 else "error",
        "mode": "real", "real_data_supported": True,
        "package": "midas_defect", "cli": script, "voxels": args.voxels,
        "out_dir": args.out_dir, "returncode": p.returncode,
        "stdout_tail": (p.stdout or "").strip().splitlines()[-15:],
        "stderr_tail": (p.stderr or "").strip().splitlines()[-15:],
    })


# ── grain-odf: per-grain ODF inversion from FF-HEDM (REAL via CLI) ─────────

def cmd_grain_odf(args):
    import midas_grain_odf  # noqa: F401
    missing = [f for f in ("geometry", "grains", "spots", "frames")
               if not getattr(args, f)]
    if missing:
        _error(f"grain-odf fit needs real FF-HEDM inputs; missing: {missing}",
               mode="real", real_data_supported=True, package="midas_grain_odf",
               required=["geometry(JSON)", "grains(csv)", "spots(csv)", "frames(npy/h5)"])
    cmd = ["midas-grain-odf", "fit", "--geometry", args.geometry,
           "--grains", args.grains, "--spots", args.spots,
           "--frames", args.frames, "--output", args.out,
           "--odf-type", args.odf_type]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    _output({
        "status": "success" if p.returncode == 0 else "error",
        "mode": "real", "real_data_supported": True, "package": "midas_grain_odf",
        "odf_type": args.odf_type, "output": args.out, "returncode": p.returncode,
        "stdout_tail": (p.stdout or "").strip().splitlines()[-15:],
        "stderr_tail": (p.stderr or "").strip().splitlines()[-15:],
    })


# ── 2d/ultrafast: coherent-diffraction forward model (SYNTHETIC demo) ──────

def cmd_twod(args):
    import midas_2d  # noqa: F401
    from importlib import import_module
    tut = args.tutorial
    mod = import_module(f"midas_2d.examples.{tut}")
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        mod.main(out_dir=args.out_dir, seed=args.seed)
    except TypeError:
        mod.main(out_dir=args.out_dir)   # some tutorials take no seed
    produced = sorted(os.listdir(args.out_dir))
    _output({
        "status": "success", "mode": "synthetic", "real_data_supported": False,
        "package": "midas_2d", "version": getattr(midas_2d, "__version__", "?"),
        "tutorial": tut, "out_dir": args.out_dir,
        "artifacts": produced[:40],
        "note": "forward/synthetic demonstration (coherent-diffraction / ultrafast "
                "2D). Real-data ingestion deferred upstream (midas-2d 0.1.0).",
    })


# ── xaf: Cross-Axis Faceted HEDM experiment design (SYNTHETIC/design) ──────

def cmd_xaf(args):
    import midas_xaf as mx
    cfg = mx.XAFConfig(
        energy_keV=float(args.energy_keV),
        opening_full_deg=float(args.opening_deg),
        n_grains=int(args.n_grains),
        n_mountings=int(args.n_mountings),
        material=args.material,
        sample_radius_um=float(args.sample_radius_um),
        seed=int(args.seed),
    )
    grains = mx.make_sample(cfg)
    result = {
        "status": "success", "mode": "synthetic", "real_data_supported": False,
        "package": "midas_xaf", "version": getattr(mx, "__version__", "?"),
        "config": {"energy_keV": cfg.energy_keV, "opening_full_deg": cfg.opening_full_deg,
                   "n_grains": cfg.n_grains, "n_mountings": cfg.n_mountings,
                   "material": cfg.material},
        "note": "anvil-cell / Cross-Axis Faceted HEDM design + simulation. "
                "Synthetic sample; real-data ingestion deferred upstream (midas-xaf 0.1.0).",
    }
    try:
        from midas_xaf.pipeline import run_pipeline
        res = run_pipeline(cfg, grains, seed=int(args.seed))
        # PipelineResult is a dataclass; surface a few scalar fields if present.
        for k in ("n_indexed", "n_grains", "completeness", "coverage_fraction",
                  "mean_completeness", "success_fraction"):
            v = getattr(res, k, None)
            if isinstance(v, (int, float)):
                result.setdefault("pipeline", {})[k] = v
        result.setdefault("pipeline", {})["result_type"] = type(res).__name__
    except Exception as e:
        result["pipeline_note"] = f"run_pipeline skipped: {type(e).__name__}: {e}"
    try:
        from midas_xaf import coverage as _cov
        cf = _cov.coverage_fraction(cfg)
        if isinstance(cf, (int, float)):
            result["coverage_fraction"] = float(cf)
    except Exception:
        pass
    _output(result)


# ── dfxm: Dark-Field X-ray Microscopy forward image (SYNTHETIC) ────────────

def cmd_dfxm(args):
    import torch
    import midas_dfxm as md

    # Synthetic uniform-strain deformation field on a small grid.
    n = int(args.grid)
    lin = torch.linspace(-5.0, 5.0, n, dtype=torch.float64)
    Y, Z = torch.meshgrid(lin, lin, indexing="ij")
    positions = torch.stack([Y.reshape(-1), Z.reshape(-1),
                             torch.zeros(n * n, dtype=torch.float64)], dim=1)
    strain = torch.zeros(positions.shape[0], 3, 3, dtype=torch.float64)
    strain[:, 0, 0] = float(args.strain)      # uniform εxx
    field = md.generators.field_from_strain(strain, positions)
    hkl = torch.tensor([float(v) for v in args.hkl.split(",")], dtype=torch.float64)
    gonio = md.GoniometerSetting()
    two_theta = float(args.two_theta_deg)
    q_mag = 2.0 * np.pi / max(float(args.d_spacing_A), 1e-6)
    q_nom = torch.tensor([0.0, 0.0, q_mag], dtype=torch.float64)
    resolution = md.ResolutionFunction(q_nom)
    optics = md.ObjectiveOptics(two_theta_deg=two_theta,
                                detector_shape=(n, n))
    img = md.forward.dfxm_image(field, hkl, gonio, resolution, optics)
    img_np = np.asarray(img.detach().cpu().numpy(), dtype=float)
    result = {
        "status": "success", "mode": "synthetic", "real_data_supported": False,
        "package": "midas_dfxm", "version": getattr(md, "__version__", "?"),
        "hkl": hkl.tolist(), "two_theta_deg": two_theta, "grid": n,
        "uniform_strain_xx": float(args.strain),
        "image_shape": list(img_np.shape), "image_stats": _array_stats(img_np),
        "note": "DFXM forward image from a synthetic strain field. Real strain/"
                "orientation-field ingestion deferred upstream (midas-dfxm 0.1.0).",
    }
    if args.out:
        np.save(args.out, img_np)
        result["output_file"] = args.out
    _output(result)


# ── pf-odf: joint per-grain peak-shape ODF inversion (DEFERRED) ────────────

def cmd_pf_odf(args):
    import midas_pf_odf as mpf
    # Real-data path requires a pf-HEDM layer dir + a built HEDMForwardModel
    # (geometry from paramstest/zarr). A turnkey synthetic self-test is not
    # shipped upstream, so we confirm the capability + attempt a guarded build.
    result = {
        "status": "success", "mode": "deferred", "real_data_supported": False,
        "package": "midas_pf_odf", "version": getattr(mpf, "__version__", "?"),
        "capability": "joint per-grain peak-shape ODF inversion for pf-HEDM",
        "real_data_entry": "load_pf_grain(layer_dir, grain_id, ...) -> "
                           "fit_grain_peakshape(...) / fit_multi_grain(...)",
        "required_for_real_run": ["pf-HEDM layer dir (Grains.csv + frames)",
                                  "geometry (paramstest or zarr)"],
        "note": "capability wired; real-data ingestion / synthetic self-test "
                "deferred upstream (midas-pf-odf 0.1.0). Provide a layer dir to "
                "drive load_pf_grain once the reader is finalized.",
    }
    if args.layer_dir and args.grain_id is not None:
        try:
            ds = mpf.load_pf_grain(args.layer_dir, int(args.grain_id),
                                   n_pixels_y=int(args.n_pixels_y),
                                   n_pixels_z=int(args.n_pixels_z))
            result["mode"] = "real"
            result["real_data_supported"] = True
            result["loaded"] = {"layer_dir": args.layer_dir,
                                "grain_id": int(args.grain_id),
                                "dataset_type": type(ds).__name__}
        except Exception as e:
            result["load_attempt"] = f"{type(e).__name__}: {e}"
    _output(result)


# ── pink: pink-beam spectrum-aware differentiable inversion (DEFERRED) ─────

def cmd_pink(args):
    import midas_pink as mpk
    spec = None
    try:
        spec = mpk.ParameterisedSpectrum(E0_keV=float(args.energy_keV),
                                         half_bw=float(args.half_bw),
                                         init_kind="boxcar",
                                         init_rel_bw=float(args.half_bw))
        spec_ok = True
    except Exception as e:
        spec_ok = False
        spec_err = f"{type(e).__name__}: {e}"
    result = {
        "status": "success", "mode": "deferred", "real_data_supported": False,
        "package": "midas_pink", "version": getattr(mpk, "__version__", "?"),
        "capability": "pink-beam spectrum-aware differentiable grain-state inversion",
        "spectrum_built": spec_ok,
        "energy_keV": float(args.energy_keV), "half_bw": float(args.half_bw),
        "real_data_entry": "build_pink_bank(spectrum, geom_factory=...) -> "
                           "recover_grain_state / recover_two_stage(observed_rois, ...)",
        "required_for_real_run": ["ParameterisedSpectrum", "HEDMGeometry factory "
                                  "(midas_diffract)", "observed ROI stack + init state"],
        "note": "capability wired; a turnkey synthetic self-test is not shipped "
                "upstream (midas-pink 0.1.0) — real pink-beam ROI ingestion deferred. "
                "Spectrum construction is exercised here as a smoke test.",
    }
    if not spec_ok:
        result["spectrum_error"] = spec_err
    _output(result)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="APEXA MIDAS capability wrapper")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("pdf")
    p.add_argument("--pattern", required=True)
    p.add_argument("--composition", required=True)
    p.add_argument("--wavelength", default="")
    p.add_argument("--x-is-two-theta", action="store_true")
    p.add_argument("--q-max", default="")
    p.add_argument("--window", default="lorch")
    p.add_argument("--r-min", default="0.0")
    p.add_argument("--r-max", default="20.0")
    p.add_argument("--n-r", default="500")
    p.add_argument("--out", default="")

    p = sub.add_parser("defect")
    p.add_argument("--voxels", default="")
    p.add_argument("--grains", default="")   # inventory mode only
    p.add_argument("--mode", default="rods",
                   choices=["rods", "asterism", "polytype", "inventory"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)

    p = sub.add_parser("grain_odf")
    p.add_argument("--geometry", default="")
    p.add_argument("--grains", default="")
    p.add_argument("--spots", default="")
    p.add_argument("--frames", default="")
    p.add_argument("--out", default="grain_odf.h5")
    p.add_argument("--odf-type", default="bingham", choices=["particle", "bingham", "voxel"])
    p.add_argument("--timeout", type=int, default=3600)

    p = sub.add_parser("twod")
    p.add_argument("--tutorial", default="tutorial_coherent_rsm")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("xaf")
    p.add_argument("--energy-keV", default="80.0")
    p.add_argument("--opening-deg", default="15.0")
    p.add_argument("--n-grains", default="50")
    p.add_argument("--n-mountings", default="2")
    p.add_argument("--material", default="zirconia_monoclinic")
    p.add_argument("--sample-radius-um", default="50.0")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("dfxm")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--strain", default="0.001")
    p.add_argument("--hkl", default="1,1,1")
    p.add_argument("--two-theta-deg", default="10.0")
    p.add_argument("--d-spacing-A", default="2.0")
    p.add_argument("--out", default="")

    p = sub.add_parser("pf_odf")
    p.add_argument("--layer-dir", default="")
    p.add_argument("--grain-id", default=None)
    p.add_argument("--n-pixels-y", default="2048")
    p.add_argument("--n-pixels-z", default="2048")

    p = sub.add_parser("pink")
    p.add_argument("--energy-keV", default="55.0")
    p.add_argument("--half-bw", default="0.02")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Keep stdout clean: package prints go to stderr; only _output hits stdout.
    sys.stdout = sys.stderr

    dispatch = {
        "pdf": cmd_pdf, "defect": cmd_defect, "grain_odf": cmd_grain_odf,
        "twod": cmd_twod, "xaf": cmd_xaf, "dfxm": cmd_dfxm,
        "pf_odf": cmd_pf_odf, "pink": cmd_pink,
    }
    try:
        dispatch[args.command](args)
    except ImportError as e:
        _error(f"package not importable: {e}. Install via the midas-suite extras "
               f"(pyproject: midas-suite[pdf,defect,dfxm,xaf,ultrafast,grain-odf,"
               f"pf-odf,pink]).", package_import_error=True)
    except FileNotFoundError as e:
        _error(str(e))
    except ValueError as e:
        _error(str(e))
    except Exception as e:
        _error(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
