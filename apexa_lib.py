"""apexa_lib — reusable primitives for APEXA debug / test / analysis scripts.

WHY THIS EXISTS
Agent-written and human scripts kept re-deriving fragile building blocks and
getting them wrong — e.g. HDF5 loaders that grabbed ``/WM/ADCoreVersion``
(1-D metadata) instead of ``/exchange/data`` (the detector image), or ad-hoc
parsers for MIDAS param files and lineouts. Import these vetted helpers in ANY
new script instead of reinventing them, across ALL stages — calibration,
integration, refinement, and general QC/debugging:

    import sys; sys.path.insert(0, "<APEXA repo path>")
    import apexa_lib as ax

    # images (calibration / integration / pixel diffs / QC)
    img  = ax.load_image("scan_002603.vrx.h5", dark="dark_1p0s.h5")
    st   = ax.image_stats("att5.h5")                     # min/max/mean/std

    # MIDAS parameter files (calibration / seeding / geometry diffs)
    p    = ax.read_params("refined_MIDAS_params_att4.txt")   # dict; +bc_x/bc_y/lsd_um
    d    = ax.compare_geometry("att4_params.txt", "att5_params.txt")

    # 1-D lineouts (integration / refinement inputs)
    tth, I = ax.read_lineout("scan.h5.analysis.MIDAS_lineout.xy")

    # outcome manifests (any stage)
    m    = ax.read_manifest(".../ceria_att5")            # APEXA_*.json

Dependency-light: numpy always; h5py for HDF5; fabio/PIL only for TIFF/GE.
"""
from pathlib import Path
import json
import numpy as np

# ── Detector images ─────────────────────────────────────────────────────────
_PREFERRED_KEYS = (
    "exchange/data", "entry/data/data", "entry/instrument/detector/data",
    "measurement/data", "data", "exchange/bright",
)


def find_image_dataset(h5) -> str:
    """HDF5 path of the detector image: a conventional key if present and ≥2-D,
    else the LARGEST ≥2-D numeric dataset — never a 1-D metadata array.
    `h5` is an open ``h5py.File``. Raises KeyError if none exists."""
    for k in _PREFERRED_KEYS:
        obj = h5.get(k)
        if obj is not None and getattr(obj, "shape", None) is not None \
                and getattr(obj, "ndim", 0) >= 2:
            return k
    best_n, best_key = 0, None

    def _visit(name, obj):
        nonlocal best_n, best_key
        try:
            if getattr(obj, "shape", None) is not None and obj.ndim >= 2 \
                    and np.issubdtype(obj.dtype, np.number):
                n = int(obj.shape[-1]) * int(obj.shape[-2])
                if n > best_n:
                    best_n, best_key = n, name
        except Exception:
            pass

    h5.visititems(_visit)
    if best_key is None:
        raise KeyError("no ≥2-D numeric dataset found in HDF5 file")
    return best_key


def _collapse(arr, frame):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        if frame == "mean":
            return arr.mean(axis=0).astype(np.float64)
        if frame == "sum":
            return arr.sum(axis=0).astype(np.float64)
        return arr[int(frame)].astype(np.float64)
    raise ValueError(f"unexpected image ndim={arr.ndim} (shape {arr.shape})")


def _load_raw(path, frame, dataset):
    path = Path(path).expanduser()
    if path.suffix.lower() in (".h5", ".hdf5", ".nxs", ".hdf"):
        import h5py
        with h5py.File(str(path), "r") as h:
            key = dataset or find_image_dataset(h)
            ds = h[key]
            if ds.ndim == 3 and isinstance(frame, int):
                return ds[frame].astype(np.float64)   # read one frame, not the stack
            return _collapse(ds[()], frame)
    try:
        import fabio
        return _collapse(fabio.open(str(path)).data, frame)
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(str(path))).astype(np.float64)


def load_image(path, frame="mean", dataset=None, dark=None, dark_frame="mean"):
    """Detector image as a 2-D float64 array. frame: 'mean'|'sum'|int for a
    stack. dark: optional path to subtract (shape-checked)."""
    img = _load_raw(path, frame, dataset)
    if dark is not None:
        d = _load_raw(dark, dark_frame, None)
        if d.shape != img.shape:
            raise ValueError(f"dark shape {d.shape} != image shape {img.shape}")
        img = img - d
    return img


def load_dark(path, frame="mean", dataset=None):
    """Dark frame (mean of the stack by default) as 2-D float64."""
    return _load_raw(path, frame, dataset)


def image_stats(path, dark=None, frame="mean"):
    img = load_image(path, frame=frame, dark=dark)
    return {"path": str(path), "shape": list(img.shape),
            "min": float(img.min()), "max": float(img.max()),
            "mean": float(img.mean()), "std": float(img.std())}


# ── MIDAS parameter files ────────────────────────────────────────────────────
def _num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def read_params(path) -> dict:
    """Parse a MIDAS parameter file (``refined_MIDAS_params*.txt``) into a dict.
    Single-value keys → int/float/str; multi-value keys (BC, LatticeConstant) →
    list. Adds convenience keys ``bc_y``/``bc_x`` (MIDAS BC order is Y X) and
    ``lsd_um`` when present."""
    out = {}
    for ln in Path(path).expanduser().read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        vals = [_num(v) for v in parts[1:]]
        out[parts[0]] = vals[0] if len(vals) == 1 else vals
    bc = out.get("BC")
    if isinstance(bc, list) and len(bc) >= 2:
        out["bc_y"], out["bc_x"] = float(bc[0]), float(bc[1])
    if "Lsd" in out:
        try:
            out["lsd_um"] = float(out["Lsd"])
        except Exception:
            pass
    return out


def write_params(params: dict, path) -> str:
    """Write a params dict back to a MIDAS parameter file (skips the convenience
    keys bc_x/bc_y/lsd_um)."""
    skip = {"bc_x", "bc_y", "lsd_um"}
    lines = []
    for k, v in params.items():
        if k in skip:
            continue
        if isinstance(v, (list, tuple)):
            lines.append(k + " " + " ".join(str(x) for x in v))
        else:
            lines.append(f"{k} {v}")
    p = Path(path).expanduser()
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def compare_geometry(a, b) -> dict:
    """Δ(b − a) of the geometry between two param files (or dicts): Lsd, tilts, BC."""
    pa = a if isinstance(a, dict) else read_params(a)
    pb = b if isinstance(b, dict) else read_params(b)
    d = {}
    for k in ("Lsd", "tx", "ty", "tz"):
        if k in pa and k in pb:
            try:
                d[k] = float(pb[k]) - float(pa[k])
            except Exception:
                pass
    for c in ("bc_x", "bc_y"):
        if c in pa and c in pb:
            d[c] = pb[c] - pa[c]
    return d


# ── Lineouts & manifests ─────────────────────────────────────────────────────
def read_lineout(path):
    """Read a MIDAS ``*_lineout.xy`` (2θ°, intensity) → (tth, intensity) arrays."""
    arr = np.loadtxt(str(Path(path).expanduser()))
    return arr[:, 0], arr[:, 1]


def read_manifest(path) -> dict:
    """Read an ``APEXA_*.json`` outcome manifest. Accepts the JSON file OR the
    directory containing it (calibration / integration / series)."""
    p = Path(path).expanduser()
    if p.is_dir():
        for name in ("APEXA_calibration.json", "APEXA_integration.json",
                     "APEXA_integration_series.json"):
            if (p / name).exists():
                p = p / name
                break
    return json.loads(p.read_text())


if __name__ == "__main__":   # CLI: python apexa_lib.py <image> [dark]
    import sys
    a = sys.argv[1:]
    if not a:
        print("usage: python apexa_lib.py <image.h5> [dark.h5]")
        sys.exit(1)
    print(json.dumps(image_stats(a[0], dark=a[1] if len(a) > 1 else None), indent=2))
