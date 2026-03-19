#!/usr/bin/env python3
"""
Create Nature-quality publication figure for Africa
c) Economic Loss (USD) Map - CALIBRATED VERSION
d) Priority Index Map

Both panels focused on full Africa region

UPDATED: December 2024
- Now uses CALIBRATED economic loss map (economic_loss_calibrated.tif)
- Only shows pixels above De Groote 2008 field-validated thresholds
- Corrects for documentation intensity bias in occurrence data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import Colorbar
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import rasterio
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# CHANGE 2: Set working directory to STRIGA_ANALYSIS for data file access
import os
os.chdir('/Users/francktonle/Downloads/STRIGA_ANALYSIS')

# ==============================================================================
# ADJUSTABLE POSITIONING PARAMETERS - EASY TO MODIFY
# ==============================================================================

# MAP EXTENT PARAMETERS FOR PANEL B (SSA)
SSA_WEST = -18                 # Western boundary
SSA_EAST = 52                  # Eastern boundary
SSA_SOUTH = -35                # Southern boundary
SSA_NORTH = 32                 # Northern boundary (increased to show all SSA countries)

# COLORBAR POSITIONING (as fraction of axes)
COLORBAR_X_POS = 0.109          # Horizontal position from left (0.0 = left edge, 1.0 = right edge)
COLORBAR_Y_POS = 0.03          # Vertical position from bottom (0.0 = bottom, 1.0 = top)
COLORBAR_WIDTH = 0.04          # Width as fraction of axes (0.04 = 4% of axes width)
COLORBAR_HEIGHT = 0.35         # Height as fraction of axes (0.35 = 35% of axes height)

# TITLE POSITIONING (as fraction of axes)
TITLE_X_POS = 0.065             # Horizontal position from left
TITLE_Y_POS = 0.41             # Vertical position from bottom (above colorbar)
TITLE_BOX_ALPHA = 0.9          # Transparency of white background box (0 = transparent, 1 = opaque)
TITLE_FONTSIZE = 12            # Font size for titles (CHANGE 1: was 10)
TITLE_LINESPACING = 1.2        # Line spacing for multi-line titles

# WHITE BACKGROUND RECTANGLE FOR LEGEND AREA
LEGEND_BG_X = 0.055            # X position of white background rectangle
LEGEND_BG_Y = 0.01             # Y position of white background rectangle
LEGEND_BG_WIDTH = 0.16         # Width of white background
LEGEND_BG_HEIGHT = 0.468        # Height of white background
LEGEND_BG_ALPHA = 0.95         # Transparency of background
LEGEND_BG_ENABLED = True       # Set to False to disable white background

# PANEL LABEL POSITIONING
PANEL_LABEL_X = 0.02           # Horizontal position from left
PANEL_LABEL_Y = 0.98           # Vertical position from bottom
PANEL_LABEL_SIZE = 16          # Font size for 'a' and 'b' labels (CHANGE 1: was 14)

# TICK LABEL SIZES
COLORBAR_TICK_SIZE = 10        # Size of colorbar tick labels (CHANGE 1: was 8)
MAP_TICK_SIZE = 11             # Size of map axis tick labels (CHANGE 1: was 9)

# ==============================================================================
# END OF ADJUSTABLE PARAMETERS
# ==============================================================================

# Check available fonts and select appropriate one
available_fonts = [f.name for f in fm.fontManager.ttflist]

# Priority order for Nature-like fonts
font_preferences = ['Harding', 'Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
selected_font = 'DejaVu Sans'  # Default fallback

for font in font_preferences:
    if font in available_fonts:
        selected_font = font
        print(f"Using font: {selected_font}")
        break

# Set publication-quality defaults for Nature
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [selected_font]
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['pdf.fonttype'] = 42  # Ensure editable text in PDFs (Nature requirement)

def load_and_prepare_data():
    """Load raster data and boundaries

    UPDATED: Now uses CALIBRATED economic loss map that only shows
    pixels above De Groote 2008 field-validated thresholds.
    """

    # Load Africa boundaries and convert to WGS84 for both panels
    africa = gpd.read_file('Africa.shp')
    africa = africa.to_crs('EPSG:4326')  # Convert to WGS84 to match raster data

    # Load CALIBRATED economic loss data (Panel C)
    # This map only includes pixels above country-specific thresholds
    # calibrated to De Groote 2008 field-validated prevalence estimates
    econ_path = 'analysis_outputs/economic_loss_calibrated.tif'
    print(f"  Loading CALIBRATED economic loss: {econ_path}")
    with rasterio.open(econ_path) as src:
        econ_data = src.read(1)
        econ_transform = src.transform
        econ_bounds = src.bounds

    # Load CALIBRATED priority index data (Panel D)
    priority_path = 'analysis_outputs/priority_index_calibrated.tif'
    print(f"  Loading CALIBRATED priority index: {priority_path}")
    with rasterio.open(priority_path) as src:
        priority_data = src.read(1)
        priority_transform = src.transform
        priority_bounds = src.bounds

    # Handle nodata (-9999 is used in calibrated maps)
    econ_data = np.where((econ_data <= 0) | (econ_data == -9999), np.nan, econ_data)
    priority_data = np.where((priority_data <= 0) | (priority_data == -9999), np.nan, priority_data)

    # Print statistics for validation
    valid_econ = econ_data[~np.isnan(econ_data)]
    valid_priority = priority_data[~np.isnan(priority_data)]
    print(f"  Economic loss: {len(valid_econ):,} pixels with valid data")
    print(f"  Total economic loss: ${np.nansum(econ_data)/1e9:.2f} billion")
    print(f"  Priority index: {len(valid_priority):,} pixels with valid data")
    print(f"  Priority range: {np.nanmin(priority_data):.1f} - {np.nanmax(priority_data):.1f}")

    return africa, econ_data, econ_transform, econ_bounds, priority_data, priority_transform, priority_bounds

def format_longitude_label(x):
    """Format longitude labels with proper negative signs"""
    if x == 0:
        return '0°'
    elif x < 0:
        return f'{int(x)}°'  # Keep negative sign for west
    else:
        return f'{int(x)}°'   # Positive for east

def format_latitude_label(y):
    """Format latitude labels with proper negative signs"""
    if y == 0:
        return '0°'
    elif y < 0:
        return f'{int(y)}°'  # Keep negative sign for south
    else:
        return f'{int(y)}°'   # Positive for north

def add_legend_background(ax):
    """Add white background rectangle for legend/colorbar area"""
    if LEGEND_BG_ENABLED:
        rect = mpatches.Rectangle(
            (LEGEND_BG_X, LEGEND_BG_Y),
            LEGEND_BG_WIDTH,
            LEGEND_BG_HEIGHT,
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='none',
            alpha=LEGEND_BG_ALPHA,
            zorder=2
        )
        ax.add_patch(rect)

def create_ssa_figure():
    """Create publication-quality figure - Both panels show Africa data"""

    # Load data
    print("Loading data...")
    africa, econ_data, econ_transform, econ_bounds, priority_data, priority_transform, priority_bounds = load_and_prepare_data()

    # Create figure with specific dimensions
    fig = plt.figure(figsize=(14, 6.5))

    # Create gridspec for precise control of spacing
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Use full Africa extent for both panels (same as panels A & B)
    extent_africa = [-20, 55, -35, 38]  # Full Africa extent to match other panels

    # ============= PANEL C: ECONOMIC LOSS (USD) - AFRICA =============
    ax1.set_xlim(extent_africa[0], extent_africa[1])
    ax1.set_ylim(extent_africa[2], extent_africa[3])
    ax1.set_aspect('equal', adjustable='box')  # Ensure square aspect

    # Apply log10 transformation first
    econ_data_log = np.log10(np.maximum(econ_data, 1))  # Ensure minimum of $1
    econ_valid_log = econ_data_log[~np.isnan(econ_data_log) & (econ_data_log > 0)]

    # Create data-driven colormap for economic loss with smooth gradation
    # Based on analysis: better yellow-orange transitions, red for top 3%
    colors_econ = [
        # 10 Yellows with better variation (0-75th percentile)
        '#ffffd9',  # Very light cream-yellow
        '#ffffb8',  # Light yellow
        '#ffff97',  # Soft yellow
        '#ffff76',  # Light-medium yellow
        '#ffff55',  # Medium yellow
        '#ffff34',  # Medium-bright yellow
        '#fffc00',  # Bright yellow
        '#fff400',  # Strong yellow
        '#ffec00',  # Deep yellow
        '#ffe400',  # Golden yellow

        # 7 Oranges with smooth transition (75-97th percentile)
        '#ffdc00',  # Yellow-orange
        '#ffd400',  # Light gold
        '#ffcc00',  # Gold
        '#ffb800',  # Light orange
        '#ffa400',  # Orange
        '#ff9000',  # Medium orange
        '#ff7c00',  # Deep orange

        # 3 Reds for critical areas (97-100th percentile)
        '#ff5500',  # Red-orange
        '#ff2200',  # Light red
        '#cc0000',  # Dark red
    ]

    # Calculate percentile positions with better distribution
    n_colors = len(colors_econ)
    positions_econ = np.zeros(n_colors)

    # Yellow positions (0-10): Use square root spacing for better low-value differentiation
    for i in range(10):
        # Map to 0-75th percentile with sqrt spacing for better spread
        pct = (np.sqrt(i / 9)) * 75
        positions_econ[i] = np.percentile(econ_valid_log, pct)

    # Orange positions (10-17): Linear spacing in 75-97 range
    for i in range(7):
        pct = 75 + (i / 6) * 22  # Linear from 75 to 97
        positions_econ[10 + i] = np.percentile(econ_valid_log, pct)

    # Red positions (17-20): Compressed at top for critical areas
    positions_econ[17] = np.percentile(econ_valid_log, 97)
    positions_econ[18] = np.percentile(econ_valid_log, 98.5)
    positions_econ[19] = np.percentile(econ_valid_log, 99.5)

    # Normalize positions to [0, 1] and ensure exact bounds
    positions_econ = (positions_econ - positions_econ.min()) / (positions_econ.max() - positions_econ.min())
    # Ensure exact 0 and 1 bounds to avoid colormap errors
    positions_econ[0] = 0.0
    positions_econ[-1] = 1.0

    cmap_econ = LinearSegmentedColormap.from_list('economic_loss',
                                                   list(zip(positions_econ, colors_econ)), N=256)

    # Set bounds based on data analysis to ensure only top 3% gets red
    # Use actual data range for proper color distribution
    vmin_econ = np.percentile(econ_valid_log, 1)  # Start at 1st percentile
    vmax_econ = np.percentile(econ_valid_log, 99.9)  # End at 99.9th percentile

    # Plot economic loss with log scale
    im1 = ax1.imshow(econ_data_log,
                     extent=[econ_bounds.left, econ_bounds.right,
                            econ_bounds.bottom, econ_bounds.top],
                     cmap=cmap_econ,
                     vmin=vmin_econ, vmax=vmax_econ,
                     interpolation='nearest',
                     alpha=1.0)

    # Add Africa boundaries for economic loss map
    africa.boundary.plot(ax=ax1, linewidth=0.5, color='black', alpha=0.7)

    # Add grid
    ax1.grid(True, linestyle='-', linewidth=0.25, alpha=0.3, color='gray', zorder=1)

    # Set tick positions for SSA (same as panel B)
    x_ticks = [-10, 0, 10, 20, 30, 40, 50]
    y_tick_interval = 10
    y_ticks = list(range(int(SSA_SOUTH/10)*10, int(SSA_NORTH/10)*10 + y_tick_interval, y_tick_interval))
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)

    # Format axis labels
    ax1.set_xticklabels([format_longitude_label(x) for x in x_ticks], fontsize=MAP_TICK_SIZE)
    ax1.set_yticklabels([format_latitude_label(y) for y in y_ticks], fontsize=MAP_TICK_SIZE)

    # Add white background for legend area
    add_legend_background(ax1)

    # Add panel label
    ax1.text(PANEL_LABEL_X, PANEL_LABEL_Y, 'c', transform=ax1.transAxes,
             fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left', zorder=10)

    # Add colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax1 = inset_axes(ax1,
                      width=f"{COLORBAR_WIDTH*100}%",
                      height=f"{COLORBAR_HEIGHT*100}%",
                      loc='lower left',
                      bbox_to_anchor=(COLORBAR_X_POS, COLORBAR_Y_POS, 1, 1),
                      bbox_transform=ax1.transAxes,
                      borderpad=0)

    cbar1 = plt.colorbar(im1, cax=cax1, orientation='vertical')

    # Set specific tick values at meaningful economic thresholds
    # Using log10 scale: $100, $1K, $10K, $100K, $1M
    tick_values_log = [np.log10(100), np.log10(1000), np.log10(10000),
                       np.log10(100000), np.log10(1000000)]
    tick_values_log = [tv for tv in tick_values_log if vmin_econ <= tv <= vmax_econ]

    cbar1.set_ticks(tick_values_log)
    cbar1.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=3, width=0.5, pad=2)

    # Format as USD values with consistent notation
    tick_labels = []
    for tv in tick_values_log:
        usd_val = 10**tv
        if usd_val >= 1e6:  # 1M and above
            tick_labels.append(f'${int(usd_val/1e6)}M')
        elif usd_val >= 1e3:  # 1K and above
            tick_labels.append(f'${int(usd_val/1e3)}K')
        else:
            tick_labels.append(f'${int(usd_val)}')

    cbar1.ax.set_yticklabels(tick_labels, fontsize=COLORBAR_TICK_SIZE)
    cbar1.outline.set_linewidth(0.5)

    # Add title
    if LEGEND_BG_ENABLED:
        ax1.text(TITLE_X_POS, TITLE_Y_POS, 'Economic loss\n(USD)',
                 transform=ax1.transAxes,
                 fontsize=TITLE_FONTSIZE,
                 va='bottom', ha='left',
                 fontweight='normal',
                 linespacing=TITLE_LINESPACING,
                 zorder=10)

    # ============= PANEL D: PRIORITY INDEX (AFRICA) =============
    ax2.set_xlim(extent_africa[0], extent_africa[1])
    ax2.set_ylim(extent_africa[2], extent_africa[3])
    ax2.set_aspect('equal', adjustable='box')  # Ensure square aspect to match Panel C

    # Create data-driven colormap for priority index
    # Green-Blue-Purple scheme for maximum differentiation from economic loss
    # Based on analysis: only top 5% gets purple
    colors_priority = [
        # 11 Greens for bottom 70% - light to medium green
        '#f0fff0', '#e0ffe0', '#d0ffd0', '#c0ffc0', '#b0ffb0',
        '#a0ffa0', '#90ff90', '#80ff80', '#70ff70', '#60ff60',
        '#50ff50',
        # 6 Blues for 70-95th percentile - transition colors
        '#40e0d0', '#30c0e0', '#20a0f0', '#1080ff', '#0060ff',
        '#0040ff',
        # 2 Purples for 95-98th percentile
        '#4020ff', '#6000ff',
        # 1 Deep purple for >98th percentile (critical intervention zones)
        '#8000ff'
    ]

    # Use square root transformation to better show variation in low values
    # Since 79% of data is below 20, this helps visualize the distribution
    priority_sqrt = np.sqrt(priority_data)
    priority_valid_sqrt = priority_sqrt[~np.isnan(priority_sqrt)]

    # Calculate percentile positions for colors
    n_colors = len(colors_priority)
    positions_priority = np.zeros(n_colors)

    # 11 colors for 0-70th percentile
    for i in range(11):
        positions_priority[i] = np.percentile(priority_valid_sqrt, i * 70 / 10)

    # 6 colors for 70-95th percentile
    for i in range(6):
        positions_priority[11 + i] = np.percentile(priority_valid_sqrt, 70 + i * 25 / 5)

    # 2 colors for 95-98th percentile
    positions_priority[17] = np.percentile(priority_valid_sqrt, 95)
    positions_priority[18] = np.percentile(priority_valid_sqrt, 96.5)

    # 1 color for >98th percentile
    positions_priority[19] = np.percentile(priority_valid_sqrt, 98)

    # Normalize positions to [0, 1] and ensure exact bounds
    positions_priority = (positions_priority - positions_priority.min()) / (positions_priority.max() - positions_priority.min())
    # Ensure exact 0 and 1 bounds to avoid colormap errors
    positions_priority[0] = 0.0
    positions_priority[-1] = 1.0

    # Use data-driven range
    vmin_priority = np.percentile(priority_valid_sqrt, 1)  # Start at 1st percentile
    vmax_priority = np.percentile(priority_valid_sqrt, 99.5)  # End at 99.5th percentile

    # Create continuous colormap for priority index with data-driven positions
    cmap_priority = LinearSegmentedColormap.from_list('priority',
                                                       list(zip(positions_priority, colors_priority)), N=256)

    # Plot priority index (using sqrt-transformed data)
    im2 = ax2.imshow(priority_sqrt,
                     extent=[priority_bounds.left, priority_bounds.right,
                            priority_bounds.bottom, priority_bounds.top],
                     cmap=cmap_priority,
                     vmin=vmin_priority,
                     vmax=vmax_priority,
                     interpolation='nearest',
                     alpha=1.0)

    # Add Africa boundaries
    africa.boundary.plot(ax=ax2, linewidth=0.5, color='black', alpha=0.7)

    # Add grid
    ax2.grid(True, linestyle='-', linewidth=0.25, alpha=0.3, color='gray', zorder=1)

    # Set tick positions for SSA view (will auto-adjust based on extent)
    x_ticks_ssa = [-10, 0, 10, 20, 30, 40, 50]
    # Dynamically set y-ticks based on SSA extent parameters
    y_tick_interval = 10
    y_ticks_ssa = list(range(int(SSA_SOUTH/10)*10, int(SSA_NORTH/10)*10 + y_tick_interval, y_tick_interval))
    ax2.set_xticks(x_ticks_ssa)
    ax2.set_yticks(y_ticks_ssa)

    # Format axis labels
    ax2.set_xticklabels([format_longitude_label(x) for x in x_ticks_ssa], fontsize=MAP_TICK_SIZE)
    ax2.set_yticklabels([format_latitude_label(y) for y in y_ticks_ssa], fontsize=MAP_TICK_SIZE)

    # Add white background for legend area
    add_legend_background(ax2)

    # Add panel label
    ax2.text(PANEL_LABEL_X, PANEL_LABEL_Y, 'd', transform=ax2.transAxes,
             fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left', zorder=10)

    # Add colorbar
    cax2 = inset_axes(ax2,
                      width=f"{COLORBAR_WIDTH*100}%",
                      height=f"{COLORBAR_HEIGHT*100}%",
                      loc='lower left',
                      bbox_to_anchor=(COLORBAR_X_POS, COLORBAR_Y_POS, 1, 1),
                      bbox_transform=ax2.transAxes,
                      borderpad=0)

    # Create continuous colorbar
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='vertical')

    # Set meaningful ticks for priority index
    # Using sqrt scale: 0, 10, 25, 50, 75, 100 become 0, 3.16, 5, 7.07, 8.66, 10
    actual_priority_values = [0, 10, 25, 50, 75, 100]
    tick_values_sqrt = [np.sqrt(v) for v in actual_priority_values]
    tick_values_sqrt = [tv for tv in tick_values_sqrt if vmin_priority <= tv <= vmax_priority]

    # Set these specific ticks
    cbar2.set_ticks(tick_values_sqrt)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=3, width=0.5, pad=2)

    # Format as actual priority values (not sqrt values)
    tick_labels = []
    for tv in tick_values_sqrt:
        actual_val = tv**2  # Convert back from sqrt
        tick_labels.append(f'{int(actual_val)}')

    cbar2.ax.set_yticklabels(tick_labels, fontsize=COLORBAR_TICK_SIZE)
    cbar2.outline.set_linewidth(0.5)

    # Add title
    if LEGEND_BG_ENABLED:
        ax2.text(TITLE_X_POS, TITLE_Y_POS, 'Priority index\n(0\u2013100)',
                 transform=ax2.transAxes,
                 fontsize=TITLE_FONTSIZE,
                 va='bottom', ha='left',
                 fontweight='normal',
                 linespacing=TITLE_LINESPACING,
                 zorder=10)

    # Fine-tune layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save figure with optimized settings for publication quality but reasonable size
    # CHANGE 2: Updated output paths for revision
    output_path = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED/figure_economic_priority_ssa_calibrated_v2.png'
    # 400 DPI is sufficient for Nature publications
    plt.savefig(output_path, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"CALIBRATED figure saved to: {output_path}")

    # Also save as PDF (vector format, typically smaller)
    output_path_pdf = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED/figure_economic_priority_ssa_calibrated_v2.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF version saved to: {output_path_pdf}")

    # CHANGE 3: Removed plt.show() for non-interactive execution

    # Print statistics
    print("\nData Statistics (SSA only):")
    print(f"Economic loss: ${np.nanmin(econ_data):,.0f} - ${np.nanmax(econ_data):,.0f}")
    print(f"  Using log scale percentile range for visualization")
    print(f"Priority index: {np.nanmin(priority_data):.1f} - {np.nanmax(priority_data):.1f}")
    print(f"  Using percentile range: {vmin_priority:.1f} - {vmax_priority:.1f} for visualization")

    return fig

if __name__ == "__main__":
    print("="*60)
    print("Creating CALIBRATED Economic Loss & Priority Index figure for SSA")
    print("="*60)
    print("\nCALIBRATION INFO:")
    print("  - Using De Groote 2008 field-validated thresholds")
    print("  - Only pixels above country-specific thresholds shown")
    print("  - Corrects for documentation intensity bias")
    print("="*60)
    print(f"\nUsing font: {selected_font}")
    print("="*60)
    print("CURRENT SSA MAP EXTENT SETTINGS:")
    print(f"  West: {SSA_WEST}°")
    print(f"  East: {SSA_EAST}°")
    print(f"  South: {SSA_SOUTH}°")
    print(f"  North: {SSA_NORTH}°")
    print("To adjust: modify SSA_WEST, SSA_EAST, SSA_SOUTH, SSA_NORTH at top of file")
    print("="*60)

    fig = create_ssa_figure()

    print("\n" + "="*60)
    print("CALIBRATED FIGURE GENERATION COMPLETE")
    print("="*60)
