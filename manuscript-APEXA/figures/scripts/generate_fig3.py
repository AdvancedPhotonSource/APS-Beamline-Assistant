#!/usr/bin/env python3
"""
Generate Figure 3: Real-Time Integration and Phase Identification
Four-panel visualization showing 2D image, 1D pattern, phase ID, and throughput scaling
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns

sns.set_style("whitegrid")

def create_figure():
    """Create the complete 4-panel figure"""
    fig = plt.figure(figsize=(7, 6))

    # Create grid with different sizes for panels
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: 2D Diffraction Image
    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax_a)

    # Panel B: Integrated 1D Pattern
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_b(ax_b)

    # Panel C: Phase Identification Results
    ax_c = fig.add_subplot(gs[1, 0])
    draw_panel_c(ax_c)

    # Panel D: Throughput Scaling
    ax_d = fig.add_subplot(gs[1, 1])
    draw_panel_d(ax_d)

    return fig

def generate_2d_diffraction_pattern():
    """Generate synthetic 2D diffraction pattern with Debye-Scherrer rings"""
    size = 512
    center_x, center_y = 256, 256

    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Start with background
    image = np.random.poisson(50, (size, size)).astype(float)

    # Add Debye-Scherrer rings for Ti-6Al-4V (alpha-Ti HCP)
    ring_radii = [80, 120, 160, 195, 230, 270]  # Approximate ring positions
    ring_intensities = [5000, 3500, 2800, 2200, 1800, 1400]

    for radius, intensity in zip(ring_radii, ring_intensities):
        # Ring with some azimuthal variation (texture)
        ring_width = 3
        ring_mask = np.abs(r - radius) < ring_width

        # Add azimuthal texture
        theta = np.arctan2(y - center_y, x - center_x)
        texture = 1 + 0.3 * np.sin(3 * theta) + 0.2 * np.cos(5 * theta)

        image += ring_mask * intensity * texture * np.random.uniform(0.8, 1.2)

    # Add Poisson noise
    image = np.random.poisson(image)

    return image, (center_x, center_y)

def generate_1d_pattern():
    """Generate synthetic 1D integrated pattern for Ti-6Al-4V"""
    # 2theta range in degrees
    two_theta = np.linspace(5, 50, 1000)

    # Background (polynomial)
    background = 100 + 50 * np.exp(-two_theta / 10)

    # Add peaks for alpha-Ti (HCP) and beta-Ti (BCC)
    peaks_alpha = [
        (10.2, 1500, 0.3),   # (100)
        (15.4, 1200, 0.3),   # (002)
        (18.3, 2000, 0.3),   # (101)
        (24.8, 800, 0.3),    # (102)
        (28.6, 1400, 0.3),   # (110)
        (32.1, 900, 0.3),    # (103)
        (35.5, 1100, 0.3),   # (200)
        (39.2, 750, 0.3),    # (112)
        (42.8, 600, 0.3),    # (201)
    ]

    peaks_beta = [
        (19.5, 300, 0.4),    # (110) - minority phase
        (32.8, 200, 0.4),    # (200)
    ]

    intensity = background.copy()

    def gaussian(x, pos, height, width):
        return height * np.exp(-((x - pos) / width) ** 2)

    # Add alpha-Ti peaks
    for pos, height, width in peaks_alpha:
        intensity += gaussian(two_theta, pos, height, width)

    # Add beta-Ti peaks
    for pos, height, width in peaks_beta:
        intensity += gaussian(two_theta, pos, height, width)

    # Add noise
    intensity += np.random.normal(0, 50, len(two_theta))

    return two_theta, intensity, peaks_alpha, peaks_beta

def draw_panel_a(ax):
    """Panel A: 2D Diffraction Image"""
    image, (cx, cy) = generate_2d_diffraction_pattern()

    # Display image
    im = ax.imshow(image, cmap='viridis', origin='lower', interpolation='bilinear',
                   vmin=0, vmax=np.percentile(image, 99.5))

    # Add beam center crosshairs
    ax.axhline(y=cy, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=cx, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)

    # Scale bar
    scale_length = 50  # pixels
    scale_x, scale_y = 30, 30
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y],
            'w-', linewidth=3)
    ax.text(scale_x + scale_length/2, scale_y + 15, '50 px',
            color='white', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.set_xlabel('Detector X (pixels)', fontsize=9)
    ax.set_ylabel('Detector Y (pixels)', fontsize=9)
    ax.set_title('Ti-6Al-4V Diffraction Pattern', fontsize=9, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Intensity (counts)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Panel label
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

def draw_panel_b(ax):
    """Panel B: Integrated 1D Pattern"""
    two_theta, intensity, peaks_alpha, peaks_beta = generate_1d_pattern()

    # Plot integrated pattern
    ax.plot(two_theta, intensity, 'k-', linewidth=1.5, label='Integrated pattern')

    # Add vertical lines for identified peaks
    for pos, height, width in peaks_alpha[:6]:  # Show first 6 alpha peaks
        ax.axvline(x=pos, color='#2ca02c', linestyle='--', linewidth=1, alpha=0.6)

    for pos, height, width in peaks_beta:
        ax.axvline(x=pos, color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.6)

    # Labels for phases
    ax.text(0.05, 0.95, 'α-Ti (HCP)', transform=ax.transAxes,
            fontsize=8, color='#2ca02c', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.text(0.05, 0.88, 'β-Ti (BCC)', transform=ax.transAxes,
            fontsize=8, color='#ff7f0e', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('2θ (degrees)', fontsize=9)
    ax.set_ylabel('Intensity (a.u.)', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(50, 5000)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Azimuthally Integrated Pattern', fontsize=9, fontweight='bold')

    # Panel label
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

def draw_panel_c(ax):
    """Panel C: Phase Identification Results Table"""
    ax.axis('off')

    # Create table data
    phases = ['α-Ti (HCP)', 'β-Ti (BCC)']
    space_groups = ['P6₃/mmc', 'Im-3m']
    matched = ['12/14', '2/3']
    confidence = ['98%', '85%']

    # Table header
    ax.text(0.5, 0.95, 'Phase Identification Results', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='center')

    # Column headers
    col_headers = ['Phase', 'Space Group', 'Peaks', 'Confidence']
    col_x = [0.05, 0.35, 0.63, 0.82]

    y_pos = 0.80
    for x, header in zip(col_x, col_headers):
        ax.text(x, y_pos, header, transform=ax.transAxes,
                fontsize=8, fontweight='bold')

    # Header line
    ax.plot([0.03, 0.97], [y_pos - 0.03, y_pos - 0.03],
            'k-', linewidth=1.5, transform=ax.transAxes)

    # Table rows
    colors = ['#2ca02c', '#ff7f0e']
    for i, (phase, sg, match, conf) in enumerate(zip(phases, space_groups, matched, confidence)):
        y_row = y_pos - 0.15 - i * 0.15

        # Phase name with color
        ax.text(col_x[0], y_row, phase, transform=ax.transAxes,
                fontsize=9, color=colors[i], fontweight='bold')

        # Space group
        ax.text(col_x[1], y_row, sg, transform=ax.transAxes,
                fontsize=9)

        # Peaks matched
        ax.text(col_x[2], y_row, match, transform=ax.transAxes,
                fontsize=9)

        # Confidence
        ax.text(col_x[3], y_row, conf, transform=ax.transAxes,
                fontsize=9, fontweight='bold')

    # Add crystal structure icons (simplified)
    # HCP representation
    hcp_x, hcp_y = 0.25, 0.20
    hexagon_points = []
    for angle in np.linspace(0, 2*np.pi, 7):
        hexagon_points.append([hcp_x + 0.08*np.cos(angle),
                               hcp_y + 0.08*np.sin(angle)])
    hexagon = plt.Polygon(hexagon_points, fill=False, edgecolor='#2ca02c',
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(hexagon)
    ax.text(hcp_x, hcp_y - 0.13, 'HCP', transform=ax.transAxes,
            fontsize=7, ha='center', color='#2ca02c')

    # BCC representation
    bcc_x, bcc_y = 0.75, 0.20
    square = plt.Rectangle((bcc_x - 0.06, bcc_y - 0.06), 0.12, 0.12,
                           fill=False, edgecolor='#ff7f0e',
                           linewidth=2, transform=ax.transAxes)
    ax.add_patch(square)
    # Center atom
    ax.plot(bcc_x, bcc_y, 'o', markersize=6, color='#ff7f0e',
            transform=ax.transAxes)
    ax.text(bcc_x, bcc_y - 0.13, 'BCC', transform=ax.transAxes,
            fontsize=7, ha='center', color='#ff7f0e')

    # Panel label
    ax.text(-0.05, 0.95, 'C', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

def draw_panel_d(ax):
    """Panel D: Throughput Scaling Performance"""
    # Data from Table 3 in data_tables.tex
    cores = np.array([1, 2, 4, 8, 16, 32, 64])
    time_per_frame = np.array([4.2, 2.1, 1.1, 0.56, 0.29, 0.15, 0.09])
    throughput = 1 / time_per_frame  # frames per second
    speedup = cores / time_per_frame / (1 / 4.2)  # relative to 1 core

    # Ideal linear scaling
    ideal_throughput = throughput[0] * cores

    # Plot throughput
    ax.plot(cores, throughput, 'o-', linewidth=2.5, markersize=8,
            color='#1f77b4', label='APEXA Performance')

    # Ideal scaling reference
    ax.plot(cores, ideal_throughput, '--', linewidth=2, color='gray',
            alpha=0.7, label='Ideal Linear Scaling')

    # Annotate efficiency at 32 cores
    efficiency_32 = speedup[5] / 32 * 100
    ax.annotate(f'{speedup[5]:.1f}× speedup\n({efficiency_32:.0f}% efficient)',
                xy=(32, throughput[5]),
                xytext=(25, 8),
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_xlabel('Number of CPU Cores', fontsize=9)
    ax.set_ylabel('Integration Rate (frames/second)', fontsize=9)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(cores)
    ax.set_xticklabels(cores)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Parallel Integration Scaling', fontsize=9, fontweight='bold')

    # Panel label
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

def main():
    """Generate and save Figure 3"""
    fig = create_figure()

    # Save as PDF
    output_path = '/Users/b324240/Git/beamline-assistant-dev/manuscript/figures/fig3_integration.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure 3 saved to: {output_path}")

    # Also save as PNG for preview
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Preview saved to: {png_path}")

    plt.close()

if __name__ == '__main__':
    main()
