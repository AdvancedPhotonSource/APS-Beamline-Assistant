#!/usr/bin/env python3
"""
Generate Figure 1: APEXA System Architecture
Three-panel diagram showing system overview, MCP protocol, and dual-environment strategy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color palette from specifications
COLORS = {
    'user': '#4A90E2',      # Blue - User/interface
    'llm': '#7ED321',       # Green - LLM/AI
    'mcp': '#F5A623',       # Orange - MCP/middleware
    'tools': '#D0021B',     # Red - Tools/computations
    'success': '#50E3C2',   # Teal
    'error': '#E94B3C'      # Red
}

def create_figure():
    """Create the complete 3-panel figure"""
    fig = plt.figure(figsize=(7, 6))

    # Panel A: Full System Architecture (top, larger)
    ax_a = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
    draw_panel_a(ax_a)

    # Panel B: MCP Protocol Detail (bottom left)
    ax_b = plt.subplot2grid((3, 2), (1, 0), colspan=1, rowspan=2)
    draw_panel_b(ax_b)

    # Panel C: Dual Environment (bottom right)
    ax_c = plt.subplot2grid((3, 2), (1, 1), colspan=1, rowspan=2)
    draw_panel_c(ax_c)

    plt.tight_layout()
    return fig

def draw_panel_a(ax):
    """Panel A: Complete System Architecture with 5 layers"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Add panel label
    ax.text(-0.5, 10, 'A', fontsize=12, fontweight='bold', va='top')

    # Layer positions (y coordinates, top to bottom)
    layers = {
        'user': 9,
        'llm': 7.5,
        'mcp': 6,
        'tools': 4.5,
        'data': 3
    }

    box_height = 0.8
    box_width = 8
    x_center = 1

    # Layer 1: User Interface
    user_box = FancyBboxPatch((x_center, layers['user']-box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.1", edgecolor='black',
                               facecolor=COLORS['user'], alpha=0.3, linewidth=1.5)
    ax.add_patch(user_box)
    ax.text(x_center + box_width/2, layers['user'], 'User Interface\n(CLI Terminal)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Layer 2: LLM Reasoning Engine
    llm_box = FancyBboxPatch((x_center, layers['llm']-box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.1", edgecolor='black',
                              facecolor=COLORS['llm'], alpha=0.3, linewidth=1.5)
    ax.add_patch(llm_box)
    ax.text(x_center + box_width/2, layers['llm'], 'LLM Reasoning Engine\n(Claude Sonnet 4.5)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Layer 3: MCP Servers (3 boxes)
    mcp_width = 2.5
    mcp_boxes = ['filesystem', 'executor', 'midas']
    for i, name in enumerate(mcp_boxes):
        x_pos = x_center + i * 2.7
        mcp_box = FancyBboxPatch((x_pos, layers['mcp']-box_height/2), mcp_width, box_height,
                                  boxstyle="round,pad=0.05", edgecolor='black',
                                  facecolor=COLORS['mcp'], alpha=0.3, linewidth=1.5)
        ax.add_patch(mcp_box)
        ax.text(x_pos + mcp_width/2, layers['mcp'], f'MCP\n{name}',
                ha='center', va='center', fontsize=8)

    # Layer 4: Analysis Tools (3 boxes)
    tool_width = 2.5
    tools = ['MIDAS', 'GSAS-II', 'pyFAI']
    for i, name in enumerate(tools):
        x_pos = x_center + i * 2.7
        tool_box = FancyBboxPatch((x_pos, layers['tools']-box_height/2), tool_width, box_height,
                                   boxstyle="round,pad=0.05", edgecolor='black',
                                   facecolor=COLORS['tools'], alpha=0.3, linewidth=1.5)
        ax.add_patch(tool_box)
        ax.text(x_pos + tool_width/2, layers['tools'], name,
                ha='center', va='center', fontsize=8, fontweight='bold')

    # Layer 5: Data Layer
    data_box = FancyBboxPatch((x_center, layers['data']-box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.1", edgecolor='black',
                               facecolor='lightgray', alpha=0.3, linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(x_center + box_width/2, layers['data'], 'Data Layer\n(Images, Calibrations, Results)',
            ha='center', va='center', fontsize=9)

    # Arrows showing data flow (solid) and control flow (dashed)
    arrow_props_solid = dict(arrowstyle='->', lw=2, color='black')
    arrow_props_dashed = dict(arrowstyle='->', lw=1.5, color='gray', linestyle='dashed')

    # User -> LLM (solid)
    ax.annotate('', xy=(5, layers['llm']+box_height/2), xytext=(5, layers['user']-box_height/2),
                arrowprops=arrow_props_solid)

    # LLM -> MCP (dashed control)
    ax.annotate('', xy=(3.5, layers['mcp']+box_height/2), xytext=(3.5, layers['llm']-box_height/2),
                arrowprops=arrow_props_dashed)

    # MCP -> Tools (solid)
    ax.annotate('', xy=(3.5, layers['tools']+box_height/2), xytext=(3.5, layers['mcp']-box_height/2),
                arrowprops=arrow_props_solid)

    # Tools -> Data (solid)
    ax.annotate('', xy=(3.5, layers['data']+box_height/2), xytext=(3.5, layers['tools']-box_height/2),
                arrowprops=arrow_props_solid)

    # Example text flow
    ax.text(x_center + box_width/2, layers['user'] + 1.2,
            '"autocalibrate CeO2 at 61keV"',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def draw_panel_b(ax):
    """Panel B: MCP Protocol Detail showing JSON-RPC communication"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Add panel label
    ax.text(-0.5, 10, 'B', fontsize=12, fontweight='bold', va='top')
    ax.text(5, 9.5, 'MCP Protocol Detail', ha='center', fontsize=10, fontweight='bold')

    # LLM box
    llm_box = FancyBboxPatch((0.5, 7), 4, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=COLORS['llm'], alpha=0.3, linewidth=1.5)
    ax.add_patch(llm_box)
    ax.text(2.5, 7.75, 'LLM', ha='center', va='center', fontsize=9, fontweight='bold')

    # MCP Server box
    mcp_box = FancyBboxPatch((5.5, 7), 4, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=COLORS['mcp'], alpha=0.3, linewidth=1.5)
    ax.add_patch(mcp_box)
    ax.text(7.5, 7.75, 'MCP Server', ha='center', va='center', fontsize=9, fontweight='bold')

    # Request arrow
    ax.annotate('', xy=(5.5, 7.9), xytext=(4.5, 7.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(5, 8.3, 'Request', ha='center', fontsize=7, color='blue')

    # Response arrow
    ax.annotate('', xy=(4.5, 7.6), xytext=(5.5, 7.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(5, 7.2, 'Response', ha='center', fontsize=7, color='green')

    # JSON Request example
    request_text = '''JSON-RPC Request:
{
  "method": "midas_auto_calibrate",
  "params": {
    "image_file": "CeO2.tif",
    "energy": 61.332,
    "stopping_strain": 0.00004
  }
}'''
    ax.text(0.3, 5.5, request_text, fontsize=6, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            va='top')

    # JSON Response example
    response_text = '''JSON Response:
{
  "success": true,
  "results": {
    "beam_center": [865.47, 702.86],
    "distance_mm": 651.118,
    "final_strain": 3.8e-5,
    "iterations": 5
  }
}'''
    ax.text(0.3, 2.5, response_text, fontsize=6, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            va='top')

def draw_panel_c(ax):
    """Panel C: Dual-Environment Strategy"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Add panel label
    ax.text(-0.5, 10, 'C', fontsize=12, fontweight='bold', va='top')
    ax.text(5, 9.5, 'Dual-Environment Strategy', ha='center', fontsize=10, fontweight='bold')

    # UV Environment box (top)
    uv_box = FancyBboxPatch((0.5, 6.5), 9, 2.3, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightblue', alpha=0.2, linewidth=2)
    ax.add_patch(uv_box)
    ax.text(1, 8.5, 'UV Environment', fontsize=9, fontweight='bold', color='blue')

    # UV contents
    ax.text(1.5, 7.8, '• MCP servers', fontsize=7)
    ax.text(1.5, 7.5, '• Client code (argo_mcp_client.py)', fontsize=7)
    ax.text(1.5, 7.2, '• Orchestration logic', fontsize=7)
    ax.text(1.5, 6.9, '• No heavy dependencies', fontsize=7)

    # Conda Environment box (bottom)
    conda_box = FancyBboxPatch((0.5, 2.5), 9, 2.3, boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='lightgreen', alpha=0.2, linewidth=2)
    ax.add_patch(conda_box)
    ax.text(1, 4.5, 'Conda midas_env', fontsize=9, fontweight='bold', color='green')

    # Conda contents
    ax.text(1.5, 3.8, '• zarr, diplib, h5py', fontsize=7)
    ax.text(1.5, 3.5, '• AutoCalibrateZarr.py', fontsize=7)
    ax.text(1.5, 3.2, '• MIDAS C++ binaries', fontsize=7)
    ax.text(1.5, 2.9, '• Heavy analysis dependencies', fontsize=7)

    # Arrow showing automatic detection
    ax.annotate('', xy=(5, 2.5), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color=COLORS['mcp']))

    # Detection function
    ax.text(6.5, 5.5, 'find_midas_python()', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['mcp'], alpha=0.3))

    # Highlight box
    highlight_text = 'No manual activation needed!'
    ax.text(5, 1.5, highlight_text, ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    # Add checkmark
    ax.text(8.5, 1.5, '✓', fontsize=16, color='green', fontweight='bold')

def main():
    """Generate and save Figure 1"""
    fig = create_figure()

    # Save as PDF
    output_path = '/Users/b324240/Git/beamline-assistant-dev/manuscript/figures/fig1_architecture.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure 1 saved to: {output_path}")

    # Also save as PNG for preview
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Preview saved to: {png_path}")

    plt.close()

if __name__ == '__main__':
    main()
