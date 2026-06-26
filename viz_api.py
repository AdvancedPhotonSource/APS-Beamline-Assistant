"""
viz_api.py — Server-side Plotly conversion for MIDAS analysis outputs.

Reads MIDAS data files (lineout XY, calibrant CSV, caked zarr, peaks HDF5)
and returns Plotly-compatible JSON dicts for the web UI VizPanel.

No FastAPI dependency — pure data→Plotly conversion functions.
"""

from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Lineout viewer  (*_lineout.xy  +  optional *_peaks.csv)
# ---------------------------------------------------------------------------

def _load_lineout_xy(path: Path) -> dict:
    """Parse MIDAS _lineout.xy file.

    Supports two formats:
    - 2-column: 2theta, intensity
    - 4-column: 2theta, raw, background, corrected
    """
    tth, raw, bg, corr = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                tth.append(float(parts[0]))
                raw.append(float(parts[1]))
                bg.append(float(parts[2]))
                corr.append(float(parts[3]))
            elif len(parts) >= 2:
                tth.append(float(parts[0]))
                corr.append(float(parts[1]))
    return {
        'tth': np.array(tth),
        'raw': np.array(raw) if raw else None,
        'bg': np.array(bg) if bg else None,
        'corr': np.array(corr),
    }


def _load_peaks_csv(path: Path) -> list[dict]:
    """Parse MIDAS _peaks.csv (CSV with # header line)."""
    rows = []
    header = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                header = [h.strip() for h in line.lstrip('# ').split(',')]
                continue
            if not line:
                continue
            parts = line.split(',')
            if header and len(parts) >= len(header):
                row = {}
                for h, v in zip(header, parts):
                    try:
                        row[h] = float(v.strip())
                    except ValueError:
                        row[h] = v.strip()
                rows.append(row)
    return rows


def lineout_to_plotly(
    xy_file: str,
    peaks_csv: str | None = None,
    show_raw: bool = True,
    show_bg: bool = True,
    log_y: bool = False,
) -> dict:
    """Convert lineout XY + optional peaks CSV to Plotly JSON.

    Returns {"plotly": {data, layout}, "tables": [...]}
    """
    xy_path = Path(xy_file)
    if not xy_path.exists():
        return {"error": f"File not found: {xy_file}"}

    d = _load_lineout_xy(xy_path)
    traces = []

    # Corrected/intensity (always shown)
    label = "Corrected" if d['raw'] is not None else "Intensity"
    traces.append({
        "x": d['tth'].tolist(),
        "y": d['corr'].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": label,
        "line": {"color": "#2196F3", "width": 1.5},
    })

    if show_raw and d['raw'] is not None:
        traces.append({
            "x": d['tth'].tolist(),
            "y": d['raw'].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Raw",
            "line": {"color": "#9E9E9E", "width": 1, "dash": "dot"},
        })

    if show_bg and d['bg'] is not None:
        traces.append({
            "x": d['tth'].tolist(),
            "y": d['bg'].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Background",
            "line": {"color": "#FF9800", "width": 1, "dash": "dash"},
        })

    # Peak markers
    tables = []
    if peaks_csv:
        peaks_path = Path(peaks_csv)
        if peaks_path.exists():
            peaks = _load_peaks_csv(peaks_path)
            if peaks:
                peak_x = [p.get('center_2theta', 0) for p in peaks]
                # Find nearest corrected intensity at each peak position
                peak_y = []
                for px in peak_x:
                    idx = np.argmin(np.abs(d['tth'] - px))
                    peak_y.append(float(d['corr'][idx]))

                traces.append({
                    "x": peak_x,
                    "y": peak_y,
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Peaks",
                    "marker": {"color": "#F44336", "size": 8, "symbol": "triangle-down"},
                })

                # Build peak table
                table_data = {}
                for key in ['center_2theta', 'area', 'FWHM', 'sig_centideg2',
                            'gam_centideg', 'eta', 'd-spacing', 'chi_sq']:
                    vals = [p.get(key, '') for p in peaks]
                    if any(isinstance(v, (int, float)) for v in vals):
                        table_data[key] = [round(v, 5) if isinstance(v, float) else v for v in vals]
                tables.append({"title": "Peak Parameters", "data": table_data})

    layout = {
        "title": {"text": xy_path.stem, "font": {"size": 14}},
        "xaxis": {"title": "2θ (deg)", "showgrid": True, "gridcolor": "#333"},
        "yaxis": {
            "title": "Intensity",
            "showgrid": True,
            "gridcolor": "#333",
            "type": "log" if log_y else "linear",
        },
        "template": "plotly_dark",
        "legend": {"x": 0.98, "y": 0.98, "xanchor": "right"},
        "margin": {"t": 40, "b": 50, "l": 60, "r": 20},
        "hovermode": "x unified",
    }

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": tables,
        "title": f"Lineout: {xy_path.stem}",
    }


# ---------------------------------------------------------------------------
# 2. Calibrant QC viewer  (*_corr.csv)
# ---------------------------------------------------------------------------

_CORR_COLUMNS = [
    'Eta', 'Strain', 'RadFit', 'EtaCalc', 'DiffCalc', 'RadCalc',
    'Ideal2Theta', 'Outlier', 'YRawCorr', 'ZRawCorr', 'RingNr',
    'RadGlobal', 'IdealR', 'Fit2Theta', 'IdealA', 'FitA',
    'DeltaR', 'DeltaA',
]


def _load_corr_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """Parse calibrant _corr.csv (space-separated, auto-detect header)."""
    with open(path) as f:
        lines = f.readlines()

    # Find header line starting with %Eta
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('%Eta') or stripped.startswith('% Eta'):
            data_start = i
            break

    if data_start is not None:
        data = np.genfromtxt(str(path), skip_header=data_start + 1)
    else:
        data = np.genfromtxt(str(path), skip_header=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    cols = _CORR_COLUMNS[:ncols]
    return data, cols


def calibrant_to_plotly(
    corr_file: str,
    x_axis: str = "Eta",
    y_axis: str = "Strain",
    color_by: str = "RingNr",
    filter_outliers: bool = True,
) -> dict:
    """Convert calibrant _corr.csv to Plotly scatter JSON."""
    corr_path = Path(corr_file)
    if not corr_path.exists():
        return {"error": f"File not found: {corr_file}"}

    data, cols = _load_corr_csv(corr_path)

    col_idx = {name: i for i, name in enumerate(cols)}
    if x_axis not in col_idx or y_axis not in col_idx:
        return {"error": f"Column not found. Available: {cols}"}

    # Filter outliers
    if filter_outliers and 'Outlier' in col_idx:
        mask = data[:, col_idx['Outlier']] == 0
        data = data[mask]

    x = data[:, col_idx[x_axis]]
    y = data[:, col_idx[y_axis]]

    # Scale strain to microstrain if plotting strain
    y_label = y_axis
    if y_axis == 'Strain':
        y = y * 1e6
        y_label = 'Strain (µε)'

    # Color by ring number or other column
    if color_by in col_idx:
        color = data[:, col_idx[color_by]]
    else:
        color = np.zeros(len(x))

    # Build per-ring traces for legend
    traces = []
    unique_rings = np.unique(color)
    colors_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]
    for i, ring in enumerate(unique_rings):
        mask = color == ring
        c = colors_palette[int(ring) % len(colors_palette)]
        traces.append({
            "x": x[mask].tolist(),
            "y": y[mask].tolist(),
            "type": "scatter",
            "mode": "markers",
            "name": f"Ring {int(ring)}",
            "marker": {"color": c, "size": 4, "opacity": 0.7},
        })

    # Add ideal lattice parameter reference line if plotting FitA vs Eta
    if y_axis == 'FitA' and 'IdealA' in col_idx:
        ideal_a = float(np.median(data[:, col_idx['IdealA']]))
        traces.append({
            "x": [float(x.min()), float(x.max())],
            "y": [ideal_a, ideal_a],
            "type": "scatter",
            "mode": "lines",
            "name": f"Ideal a = {ideal_a:.4f} Å",
            "line": {"color": "#00E676", "width": 2, "dash": "dash"},
        })

    layout = {
        "title": {"text": corr_path.stem, "font": {"size": 14}},
        "xaxis": {"title": x_axis + (" (deg)" if x_axis == "Eta" else ""), "showgrid": True, "gridcolor": "#333"},
        "yaxis": {"title": y_label, "showgrid": True, "gridcolor": "#333"},
        "template": "plotly_dark",
        "legend": {"x": 0.98, "y": 0.98, "xanchor": "right"},
        "margin": {"t": 40, "b": 50, "l": 70, "r": 20},
        "hovermode": "closest",
    }

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [],
        "title": f"Calibrant QC: {corr_path.stem}",
    }


# ---------------------------------------------------------------------------
# 3. Caked heatmap viewer  (*.caked.hdf.zarr.zip)
# ---------------------------------------------------------------------------

def _load_caked_zarr(path: Path) -> dict:
    """Load axes and intensity frames from caked zarr file."""
    import zarr
    z = zarr.open(str(path), mode='r')

    retamap = z['REtaMap'][:]
    tth_axis = retamap[1][:, 0]     # 2theta axis
    eta_axis = retamap[2][0, :]     # eta axis

    frames = {}
    frame_keys = []
    if 'OmegaSumFrame' in z:
        osf = z['OmegaSumFrame']
        frame_keys = sorted(osf.keys(), key=lambda k: int(k.split('_')[-1]))
        for fk in frame_keys:
            frames[fk] = osf[fk][:]
    elif 'SumFrames' in z:
        frame_keys = ['SumFrames']
        frames['SumFrames'] = z['SumFrames'][:]

    return {
        'tth_axis': tth_axis,
        'eta_axis': eta_axis,
        'frame_keys': frame_keys,
        'frames': frames,
    }


def caked_to_plotly(
    zarr_file: str,
    frame: int = -1,
    colorscale: str = "Viridis",
    log_intensity: bool = False,
) -> dict:
    """Convert caked zarr to Plotly 2D heatmap JSON."""
    zarr_path = Path(zarr_file)
    if not zarr_path.exists():
        return {"error": f"File not found: {zarr_file}"}

    d = _load_caked_zarr(zarr_path)
    if not d['frame_keys']:
        return {"error": "No frames found in zarr file"}

    # Select frame
    fk = d['frame_keys'][frame]
    intensity = d['frames'][fk].astype(float)

    if log_intensity:
        intensity = np.log10(np.maximum(intensity, 1e-10))

    # Detector gaps / dead pixels are NaN (and log10 can yield ±inf). Strict JSON
    # cannot encode NaN/Inf, and Plotly renders null as a gap — exactly what we
    # want for missing pixels — so map every non-finite value to None.
    z = np.where(np.isfinite(intensity), intensity, None)

    traces = [{
        "z": z.tolist(),
        "x": d['eta_axis'].tolist(),
        "y": d['tth_axis'].tolist(),
        "type": "heatmap",
        "colorscale": colorscale,
        "colorbar": {"title": "log₁₀(I)" if log_intensity else "Intensity"},
        "hovertemplate": "2θ: %{y:.3f}°<br>η: %{x:.1f}°<br>I: %{z:.1f}<extra></extra>",
    }]

    layout = {
        "title": {"text": f"{zarr_path.stem} — {fk}", "font": {"size": 14}},
        "xaxis": {"title": "η (deg)", "showgrid": False},
        "yaxis": {"title": "2θ (deg)", "showgrid": False, "autorange": "reversed"},
        "template": "plotly_dark",
        "margin": {"t": 40, "b": 50, "l": 70, "r": 20},
    }

    info = {
        "frames_available": len(d['frame_keys']),
        "current_frame": fk,
        "shape": f"{intensity.shape[0]} × {intensity.shape[1]}",
        "2θ_range": f"{float(d['tth_axis'].min()):.2f} – {float(d['tth_axis'].max()):.2f}°",
        "η_range": f"{float(d['eta_axis'].min()):.1f} – {float(d['eta_axis'].max()):.1f}°",
    }

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [{"title": "Caked Data Info", "data": info}],
        "title": f"Caked: {zarr_path.stem}",
    }


# ---------------------------------------------------------------------------
# 4. Integrator peak analysis  (*.caked.hdf.zarr.zip → peak fitting)
# ---------------------------------------------------------------------------

def integrator_peaks_to_plotly(
    zarr_file: str,
    corr_csv: str | None = None,
    frame: int = -1,
    min_height_frac: float = 0.05,
    prominence_frac: float = 0.03,
) -> dict:
    """Fit peaks in caked zarr data, return scatter of fitted 2θ vs η.

    This is a simplified version — finds peaks per η-bin using scipy,
    but does NOT do full pseudo-Voigt fitting (too slow for web).
    Instead reports peak positions from find_peaks.
    """
    from scipy.signal import find_peaks

    zarr_path = Path(zarr_file)
    if not zarr_path.exists():
        return {"error": f"File not found: {zarr_file}"}

    d = _load_caked_zarr(zarr_path)
    if not d['frame_keys']:
        return {"error": "No frames found in zarr file"}

    fk = d['frame_keys'][frame]
    intensity = d['frames'][fk].astype(float)
    tth_axis = d['tth_axis']
    eta_axis = d['eta_axis']

    imax = intensity.max()
    min_height = min_height_frac * imax
    prominence = prominence_frac * imax

    # Find peaks per eta bin
    peak_tth = []
    peak_eta = []
    peak_intensity = []

    for j in range(len(eta_axis)):
        profile = intensity[:, j]
        if np.max(profile) <= 0:
            continue
        indices, props = find_peaks(profile, height=min_height, prominence=prominence)
        for idx in indices:
            peak_tth.append(float(tth_axis[idx]))
            peak_eta.append(float(eta_axis[j]))
            peak_intensity.append(float(profile[idx]))

    if not peak_tth:
        return {"error": "No peaks found above threshold"}

    # Load ideal 2θ reference lines from corr.csv if provided
    ideal_lines = []
    if corr_csv:
        corr_path = Path(corr_csv)
        if corr_path.exists():
            data, cols = _load_corr_csv(corr_path)
            col_idx = {name: i for i, name in enumerate(cols)}
            if 'Ideal2Theta' in col_idx:
                ideal_tth = np.unique(np.round(data[:, col_idx['Ideal2Theta']], 3))
                ideal_lines = ideal_tth.tolist()

    traces = [{
        "x": peak_eta,
        "y": peak_tth,
        "type": "scatter",
        "mode": "markers",
        "name": "Detected Peaks",
        "marker": {
            "color": peak_intensity,
            "colorscale": "Viridis",
            "size": 4,
            "opacity": 0.7,
            "colorbar": {"title": "Intensity"},
        },
        "hovertemplate": "η: %{x:.1f}°<br>2θ: %{y:.3f}°<br>I: %{marker.color:.1f}<extra></extra>",
    }]

    # Add ideal 2θ reference lines
    for tth_val in ideal_lines:
        traces.append({
            "x": [float(eta_axis.min()), float(eta_axis.max())],
            "y": [tth_val, tth_val],
            "type": "scatter",
            "mode": "lines",
            "name": f"Ideal 2θ={tth_val:.3f}°",
            "line": {"color": "#00E676", "width": 1, "dash": "dot"},
            "showlegend": False,
        })

    layout = {
        "title": {"text": f"Peak Analysis: {zarr_path.stem}", "font": {"size": 14}},
        "xaxis": {"title": "η (deg)", "showgrid": True, "gridcolor": "#333"},
        "yaxis": {"title": "2θ (deg)", "showgrid": True, "gridcolor": "#333"},
        "template": "plotly_dark",
        "margin": {"t": 40, "b": 50, "l": 70, "r": 20},
        "hovermode": "closest",
    }

    info = {
        "total_peaks": len(peak_tth),
        "frame": fk,
        "2θ_range": f"{min(peak_tth):.3f} – {max(peak_tth):.3f}°",
        "η_range": f"{min(peak_eta):.1f} – {max(peak_eta):.1f}°",
    }
    if ideal_lines:
        info["ideal_rings"] = len(ideal_lines)

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [{"title": "Peak Analysis Info", "data": info}],
        "title": f"Peaks: {zarr_path.stem}",
    }


# ---------------------------------------------------------------------------
# 5. File discovery — find analysis outputs in a directory
# ---------------------------------------------------------------------------

def discover_viz_files(directory: str) -> dict:
    """Scan a directory for MIDAS analysis output files, grouped by viewer type."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return {"error": f"Not a directory: {directory}"}

    results = {
        "lineout": [],
        "calibrant": [],
        "caked": [],
        "peaks_h5": [],
        "grains": [],
        "spots": [],
        "microstructure": [],
    }

    for f in sorted(dir_path.iterdir()):
        name = f.name
        if name.endswith('_lineout.xy') or name.endswith('.lineout.xy'):
            peaks_csv = None
            peaks_candidate = f.with_name(name.replace('_lineout.xy', '_peaks.csv').replace('.lineout.xy', '_peaks.csv'))
            if peaks_candidate.exists():
                peaks_csv = str(peaks_candidate)
            results["lineout"].append({"file": str(f), "peaks_csv": peaks_csv})

        elif name.endswith('_corr.csv') or name.endswith('.corr.csv'):
            results["calibrant"].append({"file": str(f)})

        elif name.endswith('.caked.hdf.zarr.zip'):
            results["caked"].append({"file": str(f)})

        elif name.endswith('_caked_peaks.h5'):
            results["peaks_h5"].append({"file": str(f)})

        elif name == 'Grains.csv' or name.startswith('Grains') and name.endswith('.csv'):
            results["grains"].append({"file": str(f)})

        elif name == 'SpotMatrix.csv' or name.startswith('SpotMatrix') and name.endswith('.csv'):
            results["spots"].append({"file": str(f)})

        elif name.endswith('.mic') or name.endswith('.map'):
            results["microstructure"].append({"file": str(f)})

    return results


# ---------------------------------------------------------------------------
# 6. FF-HEDM grain map  (Grains.csv → 3D scatter)
# ---------------------------------------------------------------------------

def _load_grains_csv(path: Path) -> np.ndarray:
    """Parse MIDAS Grains.csv (whitespace-separated, 9-line header)."""
    return np.genfromtxt(str(path), skip_header=9)


def grains_to_plotly(
    grains_file: str,
    color_by: str = "confidence",
) -> dict:
    """Convert Grains.csv to 3D scatter plot of grain centroids."""
    grains_path = Path(grains_file)
    if not grains_path.exists():
        return {"error": f"File not found: {grains_file}"}

    try:
        data = _load_grains_csv(grains_path)
    except Exception as e:
        return {"error": f"Failed to parse Grains.csv: {e}"}

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_grains = data.shape[0]
    n_cols = data.shape[1]

    x = data[:, 10]
    y = data[:, 11]
    z = data[:, 12]

    color_options = {
        "confidence": (23, "Confidence"),
        "grain_radius": (22, "Grain Radius (µm)"),
        "strain_error": (-5, "RMS Strain Error"),
        "euler0": (-3, "Euler φ₁ (rad)"),
        "euler1": (-2, "Euler Φ (rad)"),
        "euler2": (-1, "Euler φ₂ (rad)"),
        "grain_id": (0, "Grain ID"),
    }

    col_idx, color_label = color_options.get(color_by, (23, "Confidence"))
    if col_idx < 0:
        col_idx = n_cols + col_idx
    color = data[:, col_idx] if col_idx < n_cols else np.zeros(n_grains)

    # Size by grain radius (column 22) if available
    sizes = data[:, 22] if n_cols > 22 else np.full(n_grains, 5.0)
    sizes = np.clip(sizes / np.max(sizes) * 15, 3, 20) if np.max(sizes) > 0 else np.full(n_grains, 5.0)

    hover = [
        f"Grain {int(data[i, 0])}<br>"
        f"X: {x[i]:.1f}<br>Y: {y[i]:.1f}<br>Z: {z[i]:.1f}<br>"
        f"Conf: {data[i, 23]:.3f}" if n_cols > 23 else f"Grain {int(data[i, 0])}"
        for i in range(n_grains)
    ]

    traces = [{
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "type": "scatter3d",
        "mode": "markers",
        "name": "Grains",
        "marker": {
            "color": color.tolist(),
            "colorscale": "Viridis",
            "size": sizes.tolist(),
            "opacity": 0.8,
            "colorbar": {"title": color_label, "len": 0.6},
        },
        "hovertext": hover,
        "hoverinfo": "text",
    }]

    layout = {
        "title": {"text": f"FF-HEDM Grain Map ({n_grains} grains)", "font": {"size": 14}},
        "scene": {
            "xaxis": {"title": "X (µm)"},
            "yaxis": {"title": "Y (µm)"},
            "zaxis": {"title": "Z (µm)"},
            "aspectmode": "data",
        },
        "template": "plotly_dark",
        "margin": {"t": 40, "b": 10, "l": 10, "r": 10},
    }

    # Summary table
    info = {
        "num_grains": n_grains,
        "x_range": f"{float(x.min()):.1f} – {float(x.max()):.1f} µm",
        "y_range": f"{float(y.min()):.1f} – {float(y.max()):.1f} µm",
        "z_range": f"{float(z.min()):.1f} – {float(z.max()):.1f} µm",
    }
    if n_cols > 23:
        info["confidence_range"] = f"{float(data[:, 23].min()):.3f} – {float(data[:, 23].max()):.3f}"

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [{"title": "Grain Map Info", "data": info}],
        "title": f"Grains: {grains_path.stem} ({n_grains})",
    }


# ---------------------------------------------------------------------------
# 7. Spot matrix / reciprocal space  (SpotMatrix.csv → 3D scatter)
# ---------------------------------------------------------------------------

def spot_matrix_to_plotly(
    spots_file: str,
    color_by: str = "ringNr",
) -> dict:
    """Convert SpotMatrix.csv to 2D scatter of diffraction spots."""
    spots_path = Path(spots_file)
    if not spots_path.exists():
        return {"error": f"File not found: {spots_file}"}

    try:
        import pandas as pd
        df = pd.read_csv(str(spots_path), sep=r'\s+', header=0)
    except Exception as e:
        return {"error": f"Failed to parse SpotMatrix.csv: {e}"}

    if df.empty:
        return {"error": "SpotMatrix.csv is empty"}

    # Column name mapping (flexible — handle both header styles)
    cols = df.columns.tolist()
    n_spots = len(df)

    # Use omega vs eta scatter as the default view
    omega_col = next((c for c in cols if c.lower() in ('omega', 'ome')), None)
    eta_col = next((c for c in cols if c.lower() == 'eta'), None)
    ttheta_col = next((c for c in cols if c.lower() in ('theta', 'ttheta', '2theta')), None)
    ring_col = next((c for c in cols if c.lower() in ('ringnr', 'ring_nr', 'ring')), None)
    grain_col = next((c for c in cols if c.lower() == 'grainid'), None)
    strain_col = next((c for c in cols if 'strain' in c.lower()), None)

    if eta_col is None or ttheta_col is None:
        # Fallback to column indices
        if len(cols) >= 12:
            eta_col = cols[6]
            ttheta_col = cols[10]
            ring_col = cols[7]
        else:
            return {"error": f"SpotMatrix.csv has {len(cols)} columns, expected ≥12"}

    x = df[eta_col].values
    y = df[ttheta_col].values * 2  # theta → 2theta

    # Color
    color_map = {
        "ringNr": (ring_col, "Ring Number"),
        "strain": (strain_col, "Strain Error"),
        "grainId": (grain_col, "Grain ID"),
        "omega": (omega_col, "Omega (deg)"),
    }
    c_col, c_label = color_map.get(color_by, (ring_col, "Ring Number"))
    color = df[c_col].values.tolist() if c_col else [0] * n_spots

    traces = [{
        "x": x.tolist(),
        "y": y.tolist(),
        "type": "scatter",
        "mode": "markers",
        "name": "Spots",
        "marker": {
            "color": color,
            "colorscale": "Viridis",
            "size": 3,
            "opacity": 0.6,
            "colorbar": {"title": c_label},
        },
        "hovertemplate": f"η: %{{x:.1f}}°<br>2θ: %{{y:.3f}}°<br>{c_label}: %{{marker.color}}<extra></extra>",
    }]

    layout = {
        "title": {"text": f"Spot Matrix ({n_spots} spots)", "font": {"size": 14}},
        "xaxis": {"title": "η (deg)", "showgrid": True, "gridcolor": "#333"},
        "yaxis": {"title": "2θ (deg)", "showgrid": True, "gridcolor": "#333"},
        "template": "plotly_dark",
        "margin": {"t": 40, "b": 50, "l": 70, "r": 20},
        "hovermode": "closest",
    }

    info = {
        "total_spots": n_spots,
        "columns": cols,
    }
    if grain_col:
        info["unique_grains"] = int(df[grain_col].nunique())
    if ring_col:
        info["ring_numbers"] = sorted(df[ring_col].dropna().unique().astype(int).tolist())

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [{"title": "Spot Matrix Info", "data": info}],
        "title": f"Spots: {spots_path.stem} ({n_spots})",
    }


# ---------------------------------------------------------------------------
# 8. NF-HEDM microstructure  (.mic → 2D scatter)
# ---------------------------------------------------------------------------

def _load_mic_file(path: Path) -> tuple[np.ndarray, str]:
    """Load .mic (text) or .map (binary) microstructure file."""
    if path.suffix == '.map':
        with open(path, 'rb') as f:
            mic_size_x = int(np.fromfile(f, dtype=np.double, count=1)[0])
            mic_size_y = int(np.fromfile(f, dtype=np.double, count=1)[0])
            _ref_x = np.fromfile(f, dtype=np.double, count=1)[0]
            _ref_y = np.fromfile(f, dtype=np.double, count=1)[0]
            raw = np.fromfile(f, dtype=np.double)
        # Reshape: each voxel has 12 values like the text format
        n_cols = 12
        n_rows = len(raw) // n_cols
        data = raw[:n_rows * n_cols].reshape(n_rows, n_cols)
        return data, 'map'
    else:
        data = np.genfromtxt(str(path), skip_header=4)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data, 'mic'


def microstructure_to_plotly(
    mic_file: str,
    color_by: str = "confidence",
    min_confidence: float = 0.0,
) -> dict:
    """Convert .mic/.map to 2D orientation map scatter plot."""
    mic_path = Path(mic_file)
    if not mic_path.exists():
        return {"error": f"File not found: {mic_file}"}

    try:
        data, fmt = _load_mic_file(mic_path)
    except Exception as e:
        return {"error": f"Failed to parse {mic_path.name}: {e}"}

    n_cols = data.shape[1]
    if n_cols < 11:
        return {"error": f"Expected ≥11 columns, got {n_cols}"}

    x = data[:, 3]
    y = data[:, 4]
    confidence = data[:, 10]

    # Filter by confidence
    mask = confidence > min_confidence
    x = x[mask]
    y = y[mask]
    data_filt = data[mask]

    color_options = {
        "confidence": (10, "Confidence"),
        "grain_id": (0, "Grain ID"),
        "euler0": (7, "Euler φ₁ (rad)"),
        "euler1": (8, "Euler Φ (rad)"),
        "euler2": (9, "Euler φ₂ (rad)"),
        "phase": (11, "Phase Nr"),
    }

    col_idx, color_label = color_options.get(color_by, (10, "Confidence"))
    color = data_filt[:, col_idx] if col_idx < n_cols else np.zeros(len(x))

    # IPF-like coloring from Euler angles
    if color_by.startswith("euler") or color_by == "ipf":
        # Simple RGB from Euler angles for orientation visualization
        e0 = data_filt[:, 7] if n_cols > 7 else np.zeros(len(x))
        e1 = data_filt[:, 8] if n_cols > 8 else np.zeros(len(x))
        e2 = data_filt[:, 9] if n_cols > 9 else np.zeros(len(x))
        r = ((e0 % np.pi) / np.pi * 255).astype(int)
        g = ((e1 % np.pi) / np.pi * 255).astype(int)
        b = ((e2 % np.pi) / np.pi * 255).astype(int)
        colors = [f'rgb({r[i]},{g[i]},{b[i]})' for i in range(len(x))]

        traces = [{
            "x": x.tolist(),
            "y": y.tolist(),
            "type": "scatter",
            "mode": "markers",
            "name": "Orientation",
            "marker": {
                "color": colors,
                "size": 3,
                "opacity": 0.9,
            },
            "hovertemplate": "X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
        }]
    else:
        traces = [{
            "x": x.tolist(),
            "y": y.tolist(),
            "type": "scatter",
            "mode": "markers",
            "name": color_label,
            "marker": {
                "color": color.tolist(),
                "colorscale": "Viridis",
                "size": 3,
                "opacity": 0.9,
                "colorbar": {"title": color_label},
            },
            "hovertemplate": f"X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>{color_label}: %{{marker.color:.3f}}<extra></extra>",
        }]

    layout = {
        "title": {"text": f"NF Microstructure: {mic_path.name}", "font": {"size": 14}},
        "xaxis": {"title": "X (µm)", "showgrid": False, "scaleanchor": "y", "scaleratio": 1},
        "yaxis": {"title": "Y (µm)", "showgrid": False},
        "template": "plotly_dark",
        "margin": {"t": 40, "b": 50, "l": 60, "r": 20},
        "hovermode": "closest",
    }

    info = {
        "total_voxels": int(data.shape[0]),
        "displayed": int(len(x)),
        "min_confidence_filter": min_confidence,
        "format": fmt,
    }
    if n_cols > 10:
        info["confidence_range"] = f"{float(confidence[mask].min()):.3f} – {float(confidence[mask].max()):.3f}"

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [{"title": "Microstructure Info", "data": info}],
        "title": f"NF: {mic_path.stem}",
    }


# ---------------------------------------------------------------------------
# 9. Lineout comparison  (overlay multiple lineouts + ideal rings)
# ---------------------------------------------------------------------------

def lineout_comparison_to_plotly(
    files: list[str],
    param_file: str | None = None,
    log_y: bool = False,
) -> dict:
    """Overlay multiple lineout XY files on a single plot."""
    traces = []
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0',
              '#00BCD4', '#FFEB3B', '#795548']

    for i, fpath in enumerate(files):
        p = Path(fpath)
        if not p.exists():
            continue
        d = _load_lineout_xy(p)
        traces.append({
            "x": d['tth'].tolist(),
            "y": d['corr'].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": p.stem,
            "line": {"color": colors[i % len(colors)], "width": 1.5},
        })

    if not traces:
        return {"error": "No valid lineout files found"}

    # Load ideal ring positions from parameter file if provided
    if param_file:
        param_path = Path(param_file)
        hkls_path = param_path.parent / 'hkls.csv'
        if hkls_path.exists():
            try:
                hkls = np.genfromtxt(str(hkls_path), skip_header=1)
                if hkls.ndim == 2 and hkls.shape[1] >= 1:
                    ideal_tth = hkls[:, 0]  # first column is 2theta
                    for tth in ideal_tth[:20]:
                        traces.append({
                            "x": [float(tth), float(tth)],
                            "y": [0, 1],
                            "type": "scatter",
                            "mode": "lines",
                            "name": f"Ring 2θ={tth:.3f}°",
                            "line": {"color": "#00E676", "width": 1, "dash": "dot"},
                            "showlegend": False,
                            "yref": "paper",
                        })
            except Exception:
                pass

    layout = {
        "title": {"text": "Lineout Comparison", "font": {"size": 14}},
        "xaxis": {"title": "2θ (deg)", "showgrid": True, "gridcolor": "#333"},
        "yaxis": {
            "title": "Intensity",
            "showgrid": True,
            "gridcolor": "#333",
            "type": "log" if log_y else "linear",
        },
        "template": "plotly_dark",
        "legend": {"x": 0.98, "y": 0.98, "xanchor": "right"},
        "margin": {"t": 40, "b": 50, "l": 60, "r": 20},
        "hovermode": "x unified",
    }

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [],
        "title": f"Comparison: {len(files)} lineouts",
    }


# ---------------------------------------------------------------------------
# 10. Caked peaks with lattice parameter analysis  (_caked_peaks.h5)
# ---------------------------------------------------------------------------

def caked_peaks_to_plotly(
    peaks_h5_file: str,
    zarr_file: str | None = None,
    frame: int = -1,
) -> dict:
    """Convert _caked_peaks.h5 to peak overlay plot with lattice parameters."""
    import h5py

    h5_path = Path(peaks_h5_file)
    if not h5_path.exists():
        return {"error": f"File not found: {peaks_h5_file}"}

    try:
        with h5py.File(str(h5_path), 'r') as hf:
            if 'peaks' not in hf:
                return {"error": "No 'peaks' group in HDF5 file"}

            pk_group = hf['peaks']
            frame_keys = sorted(pk_group.keys())
            if not frame_keys:
                return {"error": "No frames in peaks data"}

            fk = frame_keys[frame]
            frame_group = pk_group[fk]

            all_peaks = []
            for eta_key in sorted(frame_group.keys(), key=lambda k: int(k.split('_')[-1])):
                eta_data = frame_group[eta_key]
                if isinstance(eta_data, h5py.Dataset):
                    peak_arr = eta_data[:]
                    if peak_arr.ndim == 2:
                        for row in peak_arr:
                            all_peaks.append(row)
                elif isinstance(eta_data, h5py.Group):
                    for pk_key in eta_data.keys():
                        ds = eta_data[pk_key]
                        if isinstance(ds, h5py.Dataset):
                            all_peaks.append(ds[:])
    except Exception as e:
        return {"error": f"Failed to read peaks HDF5: {e}"}

    if not all_peaks:
        return {"error": "No peak data found in HDF5 file"}

    peaks = np.array(all_peaks)
    n_peaks = len(peaks)

    # Build scatter: center_2theta vs eta_bin index
    traces = [{
        "x": list(range(n_peaks)),
        "y": peaks[:, 0].tolist() if peaks.shape[1] > 0 else [],
        "type": "scatter",
        "mode": "markers",
        "name": "Fitted Peaks",
        "marker": {"color": "#2196F3", "size": 5, "opacity": 0.7},
    }]

    layout = {
        "title": {"text": f"Caked Peaks: {h5_path.stem}", "font": {"size": 14}},
        "xaxis": {"title": "Peak Index", "showgrid": True, "gridcolor": "#333"},
        "yaxis": {"title": "2θ (deg)", "showgrid": True, "gridcolor": "#333"},
        "template": "plotly_dark",
        "margin": {"t": 40, "b": 50, "l": 70, "r": 20},
    }

    info = {"total_peaks": n_peaks, "frame": fk, "columns_per_peak": peaks.shape[1] if peaks.ndim == 2 else 0}

    return {
        "plotly": {"data": traces, "layout": layout},
        "tables": [{"title": "Caked Peaks Info", "data": info}],
        "title": f"Caked Peaks: {h5_path.stem}",
    }
