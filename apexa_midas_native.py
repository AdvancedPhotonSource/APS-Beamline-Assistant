"""Native-Python MIDAS engine wrappers (pure-Python pip packages).

MIDAS v11.x ships its analysis pipeline as ~18 pip-installable, pure-Python
(PyTorch-based) packages under `MIDAS/packages/*`. This module wraps the
subset that APEXA's MCP tools currently shell out to via subprocess. Each
wrapper:

  * probes the import lazily and raises `MidasEngineUnavailable` with an
    install hint if the package is missing — caller can fall back to the
    legacy subprocess path
  * accepts the same arguments APEXA's MCP tools already accept (so the
    surface stays stable)
  * returns a JSON-serializable dict matching the subprocess-path schema,
    so downstream consumers (Gradio, web UI, RAG context) don't notice

The aim: kill ~400 ms conda-env interpreter startup per call, eliminate
flag-name fragility (the recent `-Wavelength` → `--wavelength` bug class),
and remove subprocess stderr scraping — without losing the legacy path
during the transition.

This module mirrors the soft-fail pattern in apexa_engines.py
(`EngineUnavailable`) so the architecture stays consistent across the two
engine families (Rietveld engines + MIDAS analysis engines).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ============================================================================
# SOFT-FAIL EXCEPTION
# ============================================================================
class MidasEngineUnavailable(RuntimeError):
    """Raised when a native MIDAS pip package is not importable.

    Caller is expected to catch this and fall back to the legacy subprocess
    path that shells out to the corresponding script in `MIDAS/utils/` or
    `MIDAS/FF_HEDM/workflows/`.
    """

    def __init__(self, package: str, install_hint: str):
        self.package = package
        self.install_hint = install_hint
        super().__init__(
            f"Native MIDAS package '{package}' not importable. {install_hint}"
        )


# ============================================================================
# IMPORT PROBES
# ============================================================================
def _require(package: str) -> Any:
    """Import `package` or raise MidasEngineUnavailable with install hint."""
    try:
        return __import__(package)
    except ImportError as e:
        hint = (
            f"Install with `pip install -e $MIDAS_PATH/packages/"
            f"{package.replace('_', '-')}` inside the APEXA uv venv "
            f"(or the midas_env conda env). Underlying error: {e}"
        )
        raise MidasEngineUnavailable(package, hint) from e


def native_engine_status() -> Dict[str, Any]:
    """Report which native MIDAS packages are importable + their versions.

    Used by `validate_midas_installation` and the startup banner. Never
    raises — missing packages are reported as `installed=False`.
    """
    packages = [
        "midas_calibrate", "midas_integrate", "midas_diffract",
        "midas_transforms", "midas_pipeline", "midas_ff_pipeline",
        "midas_nf_pipeline", "midas_index", "midas_peakfit",
        "midas_fit_grain", "midas_hkls", "midas_stress",
        "midas_params", "midas_process_grains", "midas_nf_preprocess",
        "midas_nf_fitorientation", "midas_parsl_configs", "midas_suite",
    ]
    out: Dict[str, Any] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            out[pkg] = {
                "installed": True,
                "version": getattr(mod, "__version__", "unknown"),
            }
        except ImportError:
            out[pkg] = {"installed": False, "version": None}
    return out


# ============================================================================
# IMAGE LOADER (mirrors midas_calibrate.cli._load_image)
# ============================================================================
def _load_image(path: Path) -> np.ndarray:
    """Load a 2D detector image from disk into a numpy array.

    Supports the same formats midas_calibrate's CLI accepts plus .ge*/.zip
    so APEXA users don't have to convert before calling.
    """
    p = Path(path)
    sfx = p.suffix.lower()
    if sfx in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(p)
    if sfx in (".h5", ".hdf5", ".nxs"):
        import h5py
        with h5py.File(p, "r") as f:
            # Pick the largest 2D dataset — typical MIDAS HDF5 layout
            # has the image at /entry/data/data or similar
            best_key = None
            best_size = 0
            def _walk(name, obj):
                nonlocal best_key, best_size
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    sz = int(np.prod(obj.shape))
                    if sz > best_size:
                        best_key = name
                        best_size = sz
            f.visititems(_walk)
            if best_key is None:
                raise ValueError(f"no 2D dataset found in {p}")
            arr = np.asarray(f[best_key])
            if arr.ndim > 2:
                arr = arr[0]  # first frame for a stack
            return arr
    if sfx == ".npy":
        return np.load(p)
    if sfx in (".ge", ".ge1", ".ge2", ".ge3", ".ge4", ".ge5"):
        # GE binary: 2048×2048 uint16 with 8192-byte header
        raw = p.read_bytes()
        header = 8192
        n_pix = 2048 * 2048
        if len(raw) < header + n_pix * 2:
            raise ValueError(f"GE file {p} too small for 2048² uint16")
        return np.frombuffer(raw[header:header + n_pix * 2],
                             dtype=np.uint16).reshape(2048, 2048)
    if sfx == ".zip":
        # Zarr archive — first array
        import zarr
        store = zarr.ZipStore(str(p), mode="r")
        try:
            root = zarr.open(store, mode="r")
            for k in root.keys():
                arr = np.asarray(root[k])
                if arr.ndim >= 2:
                    return arr[0] if arr.ndim > 2 else arr
        finally:
            store.close()
        raise ValueError(f"no 2D array in zarr {p}")
    raise ValueError(f"unsupported image extension: {sfx}")


# ============================================================================
# NATIVE AUTOCALIBRATE
# ============================================================================
def _has_torch_accelerator() -> tuple[bool, str]:
    """Return (available, name). MPS or CUDA → True; CPU-only → False."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, "cuda"
        if (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()):
            return True, "mps"
    except ImportError:
        return False, "no-torch"
    return False, "cpu"


def native_autocalibrate(
    image_file: str,
    parameters_file: str = "",
    dark_file: str = "",
    *,
    n_iterations: Optional[int] = None,
    output_file: Optional[str] = None,
    verbose: bool = True,
    force_native: bool = False,
) -> Dict[str, Any]:
    """Run midas_calibrate.autocalibrate on the given image.

    Returns a dict matching the subprocess-path schema in
    `midas_auto_calibrate`:

        {
          "tool": "midas_auto_calibrate",
          "engine": "native",
          "status": "success",
          "calibrated_parameters": {bc_x, bc_y, lsd, tx, ty, tz, p0..p3,
                                    wavelength, px},
          "convergence": {num_iterations, final_mean_strain, converged,
                          history: [...]},
          "output_file": "<path to refined params .txt>",
        }

    Raises:
        MidasEngineUnavailable: if midas_calibrate is not importable.
        ValueError, FileNotFoundError: for bad inputs.
    """
    mc = _require("midas_calibrate")

    # Hardware gate: native PyTorch path is currently CPU-prohibitive on
    # large area detectors (build_csr scales with NrPixelsY × NrPixelsZ;
    # 2880² takes >7 min/iter on CPU vs ~30s for the whole subprocess
    # path). Refuse to run on CPU-only hosts unless the caller explicitly
    # asks (e.g. for a benchmark or when profiling).
    accel_ok, accel_name = _has_torch_accelerator()
    if not accel_ok and not force_native:
        raise MidasEngineUnavailable(
            "midas_calibrate",
            f"Hardware accelerator unavailable (got '{accel_name}'); native "
            "PyTorch calibration is too slow on CPU-only hosts. Pass "
            "force_native=True to override (e.g. for an overnight benchmark)."
        )
    if verbose:
        print(f"[native] torch accelerator: {accel_name} "
              f"(force_native={force_native})", file=sys.stderr)

    if not parameters_file:
        raise ValueError(
            "native_autocalibrate requires a parameters_file. The native API "
            "doesn't yet support the auto-detect-from-filename path that "
            "AutoCalibrateZarr.py provides — fall back to subprocess for that."
        )

    img_path = Path(image_file).expanduser().resolve()
    par_path = Path(parameters_file).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    if not par_path.exists():
        raise FileNotFoundError(f"parameters file not found: {par_path}")

    if verbose:
        print(f"[native] midas_calibrate v{mc.__version__}", file=sys.stderr)
        print(f"[native]   image:  {img_path}", file=sys.stderr)
        print(f"[native]   params: {par_path}", file=sys.stderr)

    params = mc.CalibrationParams.from_file(par_path)
    if n_iterations is not None:
        params.nIterations = int(n_iterations)

    # Bridge legacy AutoCalibrateZarr.py ps.txt → native CalibrationParams.
    # The native API has a richer schema than the legacy file format. Fill
    # safe defaults for fields the legacy parser doesn't populate. Logged
    # verbosely so anyone debugging a divergence vs. subprocess can see the
    # exact substitutions.
    bridge_log = []
    if params.MaxRingRad <= 0:
        diag = (params.NrPixelsY ** 2 + params.NrPixelsZ ** 2) ** 0.5
        params.MaxRingRad = float(diag / 2.0)
        bridge_log.append(f"MaxRingRad → {params.MaxRingRad:.1f} px (half-diagonal)")
    if params.RBinSize <= 0:
        # Legacy RBinDivisions/RBinWidth aren't parsed; use dataclass default.
        params.RBinSize = 0.25
        bridge_log.append("RBinSize → 0.25 px (native default; legacy used RBinDivisions/RBinWidth)")
    if params.EtaBinSize <= 0:
        params.EtaBinSize = 5.0
        bridge_log.append("EtaBinSize → 5.0 deg (native default)")
    if verbose and bridge_log:
        for line in bridge_log:
            print(f"[native][bridge] {line}", file=sys.stderr)

    image = _load_image(img_path)
    dark = _load_image(Path(dark_file)) if dark_file else None

    result = mc.autocalibrate(params, image, dark=dark, verbose=verbose)

    # Write refined params alongside the input (matches CLI default).
    if output_file:
        out = Path(output_file).expanduser().resolve()
    else:
        # Mirror legacy convention: refined_MIDAS_params_<material>.txt
        # Material is stored on params.SpaceGroup → name (best effort).
        # Safe default: refined_<paramfile_stem>.txt next to the image.
        out = img_path.parent / f"refined_{par_path.stem}.txt"
    result.params.write(out)

    # Extract the same fields the subprocess path parses from the .txt file.
    p = result.params
    calibrated = {
        "bc_x": float(p.BC_z) if hasattr(p, "BC_z") else None,
        "bc_y": float(p.BC_y) if hasattr(p, "BC_y") else None,
        "lsd": float(p.Lsd),
        "tx": float(p.tx),
        "ty": float(p.ty),
        "tz": float(p.tz),
        "p0": float(getattr(p, "p0", 0.0)),
        "p1": float(getattr(p, "p1", 0.0)),
        "p2": float(getattr(p, "p2", 0.0)),
        "p3": float(getattr(p, "p3", 0.0)),
        "wavelength": float(getattr(p, "Wavelength", 0.0)),
        "px": float(0.5 * (p.pxY + p.pxZ)) if p.pxZ > 0 else float(p.pxY),
    }

    history = [
        {
            "iter": rec.iteration,
            "n_fits": rec.n_fitted,
            "mean_strain_uE": rec.mean_strain_uE,
            "Lsd": rec.Lsd,
            "BC_y": rec.BC_y,
            "BC_z": rec.BC_z,
        }
        for rec in result.history
    ]
    final_mean_strain = (result.history[-1].mean_strain_uE
                         if result.history else None)

    return {
        "tool": "midas_auto_calibrate",
        "engine": "native",
        "engine_version": mc.__version__,
        "status": "success",
        "calibrated_parameters": calibrated,
        "convergence": {
            "num_iterations": len(result.history),
            "final_mean_strain_uE": final_mean_strain,
            "converged": (final_mean_strain is not None
                          and final_mean_strain < 50.0),
            "history": history,
        },
        "output_file": str(out),
        "image_file": str(img_path),
        "parameters_file": str(par_path),
    }


# ============================================================================
# NATIVE INTEGRATE 2D → 1D  (Strategy C: PyTorch compute + dual-format write)
# ============================================================================
def _build_legacy_retamap(
    *, n_r: int, n_eta: int,
    RMin: float, RBinSize: float,
    EtaMin: float, EtaBinSize: float,
    Lsd: float, px: float, Wavelength: float,
    area_per_bin: np.ndarray,
) -> np.ndarray:
    """Construct the (5, n_r, n_eta) REtaMap that IntegratorZarrOMP writes.

    Layout (matches midas_integrate.exporters docstring):
        REtaMap[0] = R in pixels
        REtaMap[1] = 2θ in degrees
        REtaMap[2] = η in degrees
        REtaMap[3] = bin area (sum of pixel weights per bin)
        REtaMap[4] = Q in Å⁻¹

    refine_v2.py uses [1] (tth) and [3] (area > 0 mask) only, so analytical
    R/2θ/η suffice — the C path produces per-bin centroids that vary slightly
    across η due to detector tilts, but for refinement the uniform centers are
    indistinguishable at the precision of refine_v2's centroid extraction.
    """
    R = RMin + RBinSize * (np.arange(n_r, dtype=np.float64) + 0.5)
    eta = EtaMin + EtaBinSize * (np.arange(n_eta, dtype=np.float64) + 0.5)
    R_grid = np.broadcast_to(R[:, None], (n_r, n_eta)).copy()
    eta_grid = np.broadcast_to(eta[None, :], (n_r, n_eta)).copy()
    two_theta_deg = np.degrees(np.arctan2(R * px, Lsd))
    tth_grid = np.broadcast_to(two_theta_deg[:, None], (n_r, n_eta)).copy()
    if Wavelength > 0:
        theta = 0.5 * np.radians(two_theta_deg)
        Q = (4.0 * np.pi / Wavelength) * np.sin(theta)
    else:
        Q = np.full(n_r, np.nan)
    Q_grid = np.broadcast_to(Q[:, None], (n_r, n_eta)).copy()
    area_grid = area_per_bin.reshape(n_r, n_eta).astype(np.float64)
    return np.stack([R_grid, tth_grid, eta_grid, area_grid, Q_grid], axis=0)


def _write_legacy_zarr_zip(
    out_path: Path, *,
    int2d: np.ndarray, retamap: np.ndarray,
    Lsd: float, Wavelength: float, BC_y: float, BC_z: float,
    px: float,
) -> None:
    """Write the IntegratorZarrOMP-style zarr.zip (refine_v2.py compatible).

    Layout:
        REtaMap                      shape (5, n_r, n_eta)
        IntegrationResult/FrameNr_0  shape (n_r, n_eta)
        OmegaSumFrame/LastFrameNumber_0  shape (n_r, n_eta)
        InstrumentParameters/{Distance, Lam, ...}  scalars (1,)
        Omegas                       shape (1,)
    """
    import zarr
    if out_path.exists():
        out_path.unlink()
    store = zarr.ZipStore(str(out_path), mode="w")
    try:
        root = zarr.open(store, mode="w")
        root.create_dataset("REtaMap", data=retamap, dtype=np.float64)
        ir = root.create_group("IntegrationResult")
        ir.create_dataset("FrameNr_0", data=int2d.astype(np.float64))
        osf = root.create_group("OmegaSumFrame")
        osf.create_dataset("LastFrameNumber_0", data=int2d.astype(np.float64))
        root.create_dataset("Omegas", data=np.array([0.0], dtype=np.float64))
        ip = root.create_group("InstrumentParameters")
        # Match the GSAS-II caked.hdf names used elsewhere in the zoo so
        # downstream readers (plot_caked_peaks.py, refine_v2.py side helpers)
        # see the expected keys.
        ip.create_dataset("Distance", data=np.array([Lsd / 1000.0], dtype=np.float64))  # mm
        ip.create_dataset("Lam",      data=np.array([Wavelength],   dtype=np.float64))  # Å
        ip.create_dataset("Polariz",  data=np.array([0.99],         dtype=np.float64))
        ip.create_dataset("BC_y",     data=np.array([BC_y],         dtype=np.float64))
        ip.create_dataset("BC_z",     data=np.array([BC_z],         dtype=np.float64))
        ip.create_dataset("px",       data=np.array([px],           dtype=np.float64))
        # GSAS-II-style profile placeholders so plotters that expect them
        # don't crash; values are nominal.
        for k, v in (("U", 2.0), ("V", -2.0), ("W", 5.0),
                     ("X", 0.0), ("Y", 0.0), ("Z", 0.0), ("SH_L", 0.002)):
            ip.create_dataset(k, data=np.array([v], dtype=np.float64))
    finally:
        store.close()


def native_integrate_2d_to_1d(
    image_file: str,
    calibration_file: str,
    *,
    dark_file: str = "",
    result_folder: Optional[str] = None,
    out_name: Optional[str] = None,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    r_bin_size: Optional[float] = None,
    eta_min: Optional[float] = None,
    eta_max: Optional[float] = None,
    eta_bin_size: Optional[float] = None,
    integration_mode: str = "floor",
    device: Optional[str] = None,
    verbose: bool = True,
    force_native: bool = False,
) -> Dict[str, Any]:
    """Native PyTorch 2D→1D azimuthal integration with dual-format output.

    Strategy C (cleanest demonstration): native PyTorch computes the (R, η)
    cake; this writer emits BOTH the legacy IntegratorZarrOMP zarr.zip
    layout (so refine_v2.py and the rest of the zoo plumbing work
    unchanged) AND the native lineout.bin/lineout.xy outputs.

    Args:
        image_file: 2D detector image (formats per `_load_image`).
        calibration_file: refined_MIDAS_params*.txt from native_autocalibrate
            or the subprocess AutoCalibrateZarr.py path. Standard MIDAS
            parameter file syntax — parsed by midas_integrate.parse_params.
        dark_file: optional multi-frame dark; subtracted from image.
        result_folder: where outputs land (default: <image>.parent/integration).
        out_name: stem for the zarr.zip (default: <image_stem>.caked.hdf).
        r_min/r_max/r_bin_size: override RMin/RMax/RBinSize from params.
        eta_min/eta_max/eta_bin_size: override the eta grid.
        integration_mode: 'floor' | 'bilinear' | 'gradient' (kernels.integrate).
        device: 'cpu' | 'cuda' | 'mps'; auto-pick best available if None.
        force_native: bypass the CPU-only refusal gate.

    Returns:
        Dict matching the subprocess `midas_integrate_2d_to_1d` schema, with
        an extra `engine='native'` key and `engine_details` describing what
        was actually written.

    Raises:
        MidasEngineUnavailable: if midas_integrate is not importable, or if
            no hardware accelerator is available and force_native=False.
        FileNotFoundError, ValueError: for bad inputs.
    """
    mi = _require("midas_integrate")
    from midas_integrate.params import parse_params
    from midas_integrate.detector_mapper import build_map
    from midas_integrate.bin_io import PixelMap
    from midas_integrate.kernels import build_csr, integrate, profile_1d, r_axis
    import torch

    # Device selection. PyTorch MPS does not support sparse CSR tensors as of
    # 2026.05 (`Could not run 'new_compressed_tensor' from the 'mps:0' device`),
    # so MPS hosts run integration on CPU; CUDA hosts use the GPU.
    accel_ok, accel_name = _has_torch_accelerator()
    if device is None:
        device = "cuda" if (accel_ok and accel_name == "cuda") else "cpu"
    if not accel_ok and not force_native and device == "cpu":
        # No CUDA, no MPS: refuse on big detectors. Pilatus 1475×1679 is fine
        # on CPU; varex 2880² is not. Use n_pixels as a rough gate.
        # (Cannot read params here yet — defer the check until after parse.)
        pass
    if verbose:
        print(f"[native][integrate] torch device: {device} "
              f"(force_native={force_native})", file=sys.stderr)

    img_path = Path(image_file).expanduser().resolve()
    par_path = Path(calibration_file).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    if not par_path.exists():
        raise FileNotFoundError(f"calibration file not found: {par_path}")

    out_dir = (Path(result_folder).expanduser().resolve()
               if result_folder
               else img_path.parent / "integration")
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[native][integrate] midas_integrate v{mi.__version__}",
              file=sys.stderr)
        print(f"[native][integrate]   image:  {img_path}", file=sys.stderr)
        print(f"[native][integrate]   params: {par_path}", file=sys.stderr)
        print(f"[native][integrate]   out:    {out_dir}", file=sys.stderr)

    params = parse_params(par_path)
    # Apply overrides before validate() so n_r_bins / n_eta_bins reflect them.
    bridge_log = []
    if r_min is not None:        params.RMin = float(r_min);        bridge_log.append(f"RMin={r_min}")
    if r_max is not None:        params.RMax = float(r_max);        bridge_log.append(f"RMax={r_max}")
    if r_bin_size is not None:   params.RBinSize = float(r_bin_size); bridge_log.append(f"RBinSize={r_bin_size}")
    if eta_min is not None:      params.EtaMin = float(eta_min);    bridge_log.append(f"EtaMin={eta_min}")
    if eta_max is not None:      params.EtaMax = float(eta_max);    bridge_log.append(f"EtaMax={eta_max}")
    if eta_bin_size is not None: params.EtaBinSize = float(eta_bin_size); bridge_log.append(f"EtaBinSize={eta_bin_size}")
    # Defaults for missing R range (legacy ps.txt files often omit RMin/RMax).
    if params.RMin <= 0 and params.RMax <= 0:
        params.RMin = 10.0
        diag = (params.NrPixelsY ** 2 + params.NrPixelsZ ** 2) ** 0.5
        params.RMax = float(diag / 2.0)
        bridge_log.append(f"RMin/RMax → 10..{params.RMax:.0f} px (half-diagonal default)")
    if params.RBinSize <= 0:
        params.RBinSize = 0.25
        bridge_log.append("RBinSize → 0.25 px")
    if params.EtaBinSize <= 0:
        params.EtaBinSize = 1.0
        bridge_log.append("EtaBinSize → 1.0 deg")
    if verbose and bridge_log:
        for line in bridge_log:
            print(f"[native][integrate][bridge] {line}", file=sys.stderr)
    params.validate()

    # Now apply the deferred CPU-size check (see device selection block above).
    n_pixels = params.NrPixelsY * params.NrPixelsZ
    PIXEL_BUDGET_CPU = 4_000_000  # ~pilatus 2.5M ok; varex 8.3M not
    if device == "cpu" and not force_native and n_pixels > PIXEL_BUDGET_CPU:
        raise MidasEngineUnavailable(
            "midas_integrate",
            f"Detector has {n_pixels:,} pixels (>{PIXEL_BUDGET_CPU:,} CPU budget); "
            "native PyTorch integration would be too slow without a CUDA GPU. "
            "Pass force_native=True to override."
        )

    # Build the pixel→bin map (numba-accelerated; one-shot per (params, geometry)).
    # Defensively neutralise sidecar-file fields that legacy refined-params files
    # often reference but that don't exist on disk (PanelShiftsFile,
    # MaskFile, ResidualCorrectionMap, etc.) — auto_load would FileNotFoundError.
    for sidecar in ("PanelShiftsFile", "MaskFile", "FlatFile",
                    "DistortionFile", "ResidualCorrectionMap"):
        path_str = getattr(params, sidecar, "")
        if path_str:
            p = Path(path_str)
            if not p.is_absolute():
                p = par_path.parent / p
            if not p.exists():
                if verbose:
                    print(f"[native][integrate][bridge] {sidecar} "
                          f"'{path_str}' not found; clearing",
                          file=sys.stderr)
                setattr(params, sidecar, "")
    if verbose:
        print(f"[native][integrate] building pixel map "
              f"({params.NrPixelsY}×{params.NrPixelsZ} → "
              f"{params.n_r_bins}×{params.n_eta_bins} bins)…",
              file=sys.stderr)
    bm = build_map(params, verbose=False)
    pixmap = PixelMap(
        pxList=bm.pxList, counts=bm.counts, offsets=bm.offsets,
        map_header=None, nmap_header=None,
    )

    torch_dtype = torch.float32
    if verbose:
        print("[native][integrate] building CSR…", file=sys.stderr)
    geom = build_csr(
        pixmap, n_r=params.n_r_bins, n_eta=params.n_eta_bins,
        n_pixels_y=params.NrPixelsY, n_pixels_z=params.NrPixelsZ,
        device=device, dtype=torch_dtype,
        bc_y=params.BC_y, bc_z=params.BC_z,
        build_modes=(integration_mode,),
    )

    image = _load_image(img_path)
    img_t = torch.as_tensor(image.astype(np.float64),
                            dtype=torch_dtype).to(device)
    if dark_file:
        dark = _load_image(Path(dark_file))
        dark_t = torch.as_tensor(dark.astype(np.float64),
                                 dtype=torch_dtype).to(device)
        img_t = img_t - dark_t

    if verbose:
        print(f"[native][integrate] integrating (mode={integration_mode})…",
              file=sys.stderr)
    int2d_t = integrate(img_t, geom, mode=integration_mode, normalize=True)
    prof_t = profile_1d(int2d_t, geom, mode="area_weighted")

    int2d = int2d_t.detach().cpu().numpy().astype(np.float64)
    prof = prof_t.detach().cpu().numpy().astype(np.float64)
    area_per_bin = geom.area_per_bin.detach().cpu().numpy().astype(np.float64)

    # Build the legacy REtaMap analytically.
    px = float(params.pxY)
    retamap = _build_legacy_retamap(
        n_r=params.n_r_bins, n_eta=params.n_eta_bins,
        RMin=params.RMin, RBinSize=params.RBinSize,
        EtaMin=params.EtaMin, EtaBinSize=params.EtaBinSize,
        Lsd=params.Lsd, px=px, Wavelength=params.Wavelength,
        area_per_bin=area_per_bin,
    )

    # Output filenames: mirror MIDAS conventions so existing zoo readers
    # (refine_v2.py CONFIG dict, plot_lineout_results.py) work unchanged.
    stem = out_name or f"{img_path.name}.caked.hdf"
    zarr_path = out_dir / f"{stem}.zarr.zip"
    lineout_xy = out_dir / f"{img_path.name}.analysis.MIDAS_lineout.xy"
    lineout_bin = out_dir / "lineout.bin"

    if verbose:
        print(f"[native][integrate] writing zarr.zip → {zarr_path.name}",
              file=sys.stderr)
    _write_legacy_zarr_zip(
        zarr_path, int2d=int2d, retamap=retamap,
        Lsd=params.Lsd, Wavelength=params.Wavelength,
        BC_y=params.BC_y, BC_z=params.BC_z, px=px,
    )

    # Native lineout: (R, intensity) → also emit 2θ-axis xy for plotters.
    R = r_axis(n_r=params.n_r_bins, RMin=params.RMin, RBinSize=params.RBinSize)
    two_theta_deg = np.degrees(np.arctan2(R * px, params.Lsd))
    pairs = np.empty(params.n_r_bins * 2, dtype=np.float64)
    pairs[0::2] = R
    pairs[1::2] = prof
    lineout_bin.write_bytes(pairs.tobytes())
    np.savetxt(lineout_xy, np.column_stack([two_theta_deg, prof]),
               fmt="%.6f %.6f", header="2theta_deg  intensity", comments="# ")

    if verbose:
        print(f"[native][integrate] done. "
              f"zarr={zarr_path.name}  lineout={lineout_xy.name}",
              file=sys.stderr)

    return {
        "tool": "midas_integrate_2d_to_1d",
        "engine": "native",
        "engine_version": mi.__version__,
        "engine_details": {
            "device": str(device),
            "integration_mode": integration_mode,
            "n_r": int(params.n_r_bins),
            "n_eta": int(params.n_eta_bins),
            "RMin": float(params.RMin), "RMax": float(params.RMax),
            "RBinSize": float(params.RBinSize),
            "EtaMin": float(params.EtaMin), "EtaMax": float(params.EtaMax),
            "EtaBinSize": float(params.EtaBinSize),
            "bridge_substitutions": bridge_log,
        },
        "status": "success",
        "input_image": str(img_path),
        "calibration_file": str(par_path),
        "result_folder": str(out_dir),
        "lineout_xy": str(lineout_xy),
        "lineout_bin": str(lineout_bin),
        "zarr_zip": str(zarr_path),
        "csv_files": None,
        "message": (f"Native integration complete. zarr={zarr_path.name} "
                    f"({params.n_r_bins}×{params.n_eta_bins}); "
                    f"lineout={lineout_xy.name}"),
    }
