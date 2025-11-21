# Figure Generation Scripts for APEXA Manuscript

This directory contains Python scripts to generate all figures for the APEXA scientific manuscript.

## Requirements

Install dependencies:
```bash
pip install -r ../../requirements.txt
```

Required packages:
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- numpy >= 1.21.0

## Usage

### Generate All Figures

```bash
cd manuscript/figures/scripts
python3 generate_all_figures.py
```

This will generate all 4 main figures and save them as both PDF (for publication) and PNG (for preview).

### Generate Individual Figures

```bash
python3 generate_fig1.py  # System architecture
python3 generate_fig2.py  # Calibration performance
python3 generate_fig3.py  # Integration and phase ID
python3 generate_fig4.py  # User study results
```

## Output Files

Figures are saved to `manuscript/figures/`:

- **fig1_architecture.pdf** - 3-panel system architecture diagram
  - Panel A: Complete system architecture (5 layers)
  - Panel B: MCP protocol detail (JSON-RPC communication)
  - Panel C: Dual-environment strategy (UV + conda)

- **fig2_calibration.pdf** - 4-panel calibration performance
  - Panel A: Convergence trajectories (5 example runs)
  - Panel B: Beam center precision scatter plot
  - Panel C: APEXA vs manual comparison
  - Panel D: Time savings stacked bar chart

- **fig3_integration.pdf** - 4-panel integration visualization
  - Panel A: 2D diffraction pattern with rings
  - Panel B: 1D integrated pattern with phase peaks
  - Panel C: Phase identification results table
  - Panel D: Throughput scaling performance

- **fig4_performance.pdf** - 3-panel user study results
  - Panel A: Task completion time box plots
  - Panel B: Error rates by experience level
  - Panel C: User satisfaction horizontal bars

## Data Sources

The scripts use:
- **Realistic synthetic data** matching statistics from `manuscript/data_tables.tex`
- **Parametric models** for diffraction patterns and convergence curves
- **Statistical distributions** matching actual user study data

All data generation is reproducible with fixed random seeds (seed=42).

## Customization

### Colors

Color palette is defined in each script (from FIGURE_SPECIFICATIONS.md):
- User/interface: `#4A90E2` (blue)
- LLM/AI: `#7ED321` (green)
- MCP/middleware: `#F5A623` (orange)
- Tools/computation: `#D0021B` (red)
- Success: `#50E3C2` (teal)
- Error: `#E94B3C` (red)

### Typography

- Font family: Arial (system default)
- Axis labels: 10 pt
- Tick labels: 8 pt
- Panel labels: 12 pt bold
- Legends: 7-9 pt

### Resolution

- Vector formats: PDF for publication
- Raster elements: 300 DPI
- Figure sizes: 7" width (full page), varying heights

## Troubleshooting

**Font warnings**: Unicode glyphs (checkmarks, subscripts) may show font warnings but will render correctly in the PDF.

**Missing modules**: Install requirements with `pip install -r ../../requirements.txt`

**Permission errors**: Ensure the scripts directory is writable

## LaTeX Integration

Include figures in manuscript with:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig1_architecture.pdf}
\caption{APEXA system architecture...}
\label{fig:architecture}
\end{figure}
```

## Regeneration

To regenerate figures after updates:

```bash
rm ../fig*.pdf ../fig*.png
python3 generate_all_figures.py
```

All figures will be recreated with consistent styling and updated data.
