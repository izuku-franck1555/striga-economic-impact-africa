#!/usr/bin/env python3
"""
Create Nature-quality publication figure for Africa
a) Agroecological Similarity Map
b) Infestation Rate Map - CALIBRATED VERSION

Both panels focused on full Africa region

UPDATED: December 2024
- Now uses CALIBRATED infestation rate map
- Only shows pixels above De Groote 2008 field-validated thresholds
- Corrects for documentation intensity bias in occurrence data

REVISION R1 (v2): March 2026
- CHANGE 1: Panel a colormap changed from diverging red-green to YlOrBr (R1.3.1)
- CHANGE 2: Font sizes increased for readability (R1.4)
- CHANGE 3: Working directory set explicitly; output paths updated
- CHANGE 4: plt.show() removed for non-interactive execution
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

# CHANGE 3: Set working directory explicitly
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
TITLE_FONTSIZE = 12            # CHANGE 2: Font size for titles (was 10)
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
PANEL_LABEL_SIZE = 16          # CHANGE 2: Font size for 'a' and 'b' labels (was 14)

# TICK LABEL SIZES
COLORBAR_TICK_SIZE = 10        # CHANGE 2: Size of colorbar tick labels (was 8)
MAP_TICK_SIZE = 11             # CHANGE 2: Size of map axis tick labels (was 9)

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

    UPDATED: Now uses CALIBRATED infestation rate map that only shows
    pixels above De Groote 2008 field-validated thresholds.
    """

    # Load FULL Africa boundaries for both panels
    africa = gpd.read_file('Africa.shp')
    africa = africa.to_crs('EPSG:4326')

    # Load CALIBRATED infestation rate
    # This map only includes pixels above country-specific thresholds
    ir_path = 'analysis_outputs/infestation_rate_calibrated.tif'
    print(f"  Loading CALIBRATED infestation rate: {ir_path}")
    with rasterio.open(ir_path) as src:
        ir_data = src.read(1)
        ir_transform = src.transform
        ir_bounds = src.bounds

    # Load similarity layer (unchanged - this is input data)
    with rasterio.open('similarity-layer.tif') as src:
        sim_data = src.read(1)
        sim_transform = src.transform
        sim_bounds = src.bounds

    # Handle nodata (-9999 is used in calibrated maps)
    ir_data = np.where((ir_data <= 0) | (ir_data == -9999), np.nan, ir_data)
    sim_data = np.where(sim_data <= 0, np.nan, sim_data)

    # Print statistics for validation
    valid_ir = ir_data[~np.isnan(ir_data)]
    print(f"  Infestation rate: {len(valid_ir):,} pixels with valid data")
    print(f"  IR range: {np.nanmin(ir_data)*100:.1f}% - {np.nanmax(ir_data)*100:.1f}%")

    return africa, ir_data, ir_transform, ir_bounds, sim_data, sim_transform, sim_bounds

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
    """Create publication-quality figure - Full Africa for both panels"""

    # Load data
    print("Loading data...")
    africa, ir_data, ir_transform, ir_bounds, sim_data, sim_transform, sim_bounds = load_and_prepare_data()

    # Create figure with specific dimensions
    fig = plt.figure(figsize=(14, 6.5))

    # Create gridspec for precise control of spacing
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Use full Africa extent for BOTH panels
    extent_africa = [-20, 55, -35, 38]  # Full Africa for both panels

    # ============= PANEL A: AGROECOLOGICAL SIMILARITY (FULL AFRICA) =============
    ax1.set_xlim(extent_africa[0], extent_africa[1])
    ax1.set_ylim(extent_africa[2], extent_africa[3])
    ax1.set_aspect('equal', adjustable='box')  # Ensure square aspect

    # CHANGE 1: Use YlOrBr sequential colormap instead of diverging red-green (R1.3.1)
    cmap_sim = plt.cm.YlOrBr

    # Calculate percentile-based range for better visualization
    sim_valid = sim_data[~np.isnan(sim_data)]
    vmin_sim = np.percentile(sim_valid, 5)   # 5th percentile
    vmax_sim = np.percentile(sim_valid, 95)  # 95th percentile

    # Plot similarity with percentile-based scale
    im1 = ax1.imshow(sim_data,
                     extent=[sim_bounds.left, sim_bounds.right,
                            sim_bounds.bottom, sim_bounds.top],
                     cmap=cmap_sim,
                     vmin=vmin_sim, vmax=vmax_sim,
                     interpolation='nearest',
                     alpha=1.0)

    # Add FULL AFRICA boundaries for similarity map
    africa.boundary.plot(ax=ax1, linewidth=0.5, color='black', alpha=0.7)

    # Add grid
    ax1.grid(True, linestyle='-', linewidth=0.25, alpha=0.3, color='gray', zorder=1)

    # Set tick positions for full Africa
    x_ticks = [-10, 0, 10, 20, 30, 40, 50]
    y_ticks = [-30, -20, -10, 0, 10, 20, 30]  # Full Africa range
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)

    # Format axis labels
    ax1.set_xticklabels([format_longitude_label(x) for x in x_ticks], fontsize=MAP_TICK_SIZE)
    ax1.set_yticklabels([format_latitude_label(y) for y in y_ticks], fontsize=MAP_TICK_SIZE)

    # Add white background for legend area
    add_legend_background(ax1)

    # Add panel label
    ax1.text(PANEL_LABEL_X, PANEL_LABEL_Y, 'a', transform=ax1.transAxes,
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
    # Create 5 ticks avoiding the max value
    # Use 0%, 20%, 40%, 60%, 80% of the range
    tick_positions = [0.0, 0.2, 0.4, 0.6, 0.8]
    tick_values = [vmin_sim + pos * (vmax_sim - vmin_sim) for pos in tick_positions]
    cbar1.set_ticks(tick_values)
    cbar1.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=3, width=0.5, pad=2)
    tick_labels = [f'{int(v*100)}' for v in tick_values]
    cbar1.ax.set_yticklabels(tick_labels, fontsize=COLORBAR_TICK_SIZE)
    cbar1.outline.set_linewidth(0.5)

    # Add title
    if LEGEND_BG_ENABLED:
        ax1.text(TITLE_X_POS, TITLE_Y_POS, 'Agroecological\nsimilarity (%)',
                 transform=ax1.transAxes,
                 fontsize=TITLE_FONTSIZE,
                 va='bottom', ha='left',
                 fontweight='normal',
                 linespacing=TITLE_LINESPACING,
                 zorder=10)

    # ============= PANEL B: INFESTATION RATE (FULL AFRICA) =============
    ax2.set_xlim(extent_africa[0], extent_africa[1])
    ax2.set_ylim(extent_africa[2], extent_africa[3])
    ax2.set_aspect('equal', adjustable='box')  # Ensure square aspect to match Panel A

    # 20-color gradient: 15 yellows → 3 oranges → 2 reds
    # Balanced distribution for periphery-epicenter visualization
    base_colors = [
        # Yellows (15 colors - Maximum differentiation from ivory to gold)
        '#fffff0',   # 1. Ivory (almost white) - far peripheral zones
        '#ffffd4',   # 2. Light cream - outer dispersal
        '#ffffb8',   # 3. Cream - early dispersal
        '#ffff9c',   # 4. Pale yellow - low risk zones
        '#ffff80',   # 5. Light yellow
        '#ffff64',   # 6. Soft yellow - moderate dispersal
        '#ffff48',   # 7. Medium yellow
        '#ffff2c',   # 8. Yellow - active dispersal
        '#ffff10',   # 9. Bright yellow
        '#fff400',   # 10. Strong yellow - significant presence
        '#ffe800',   # 11. Deep yellow
        '#ffdc00',   # 12. Golden yellow - approaching moderate
        '#ffd000',   # 13. Gold
        '#ffc400',   # 14. Deep gold - near intervention
        '#ffb800',   # 15. Dark gold (approaching orange)

        # Oranges (3 colors - intervention zones)
        '#ffaa00',   # 16. Light orange - moderate severity
        '#ff9900',   # 17. Orange - high severity
        '#ff8800',   # 18. Deep orange - very high severity

        # Reds (2 colors - critical epicenters)
        '#ff6600',   # 19. Red-orange - severe epicenters
        '#ff3300'    # 20. Red (critical epicenter cores only)
    ]

    # Apply power transformation FIRST
    power_factor = 0.15
    ir_data_transformed = np.power(ir_data, power_factor)

    # Calculate percentiles on transformed data
    ir_transformed_valid = ir_data_transformed[~np.isnan(ir_data_transformed)]
    vmin_transformed = np.percentile(ir_transformed_valid, 2)
    vmax_transformed = np.percentile(ir_transformed_valid, 85)

    # Create ADAPTIVE colormap based on data distribution
    # This ensures each color represents equal amount of data
    data_in_range = ir_transformed_valid[(ir_transformed_valid >= vmin_transformed) &
                                         (ir_transformed_valid <= vmax_transformed)]

    # For adaptive coloring with CONTINUOUS colorbar
    # Create a smooth colormap with colors positioned at quantiles

    # Calculate quantile positions for 20 colors
    n_colors = 20
    # Create positions at quantiles to ensure equal data representation
    quantile_points = np.linspace(0, 100, n_colors)
    quantile_values = np.percentile(data_in_range, quantile_points)

    # Normalize quantile values to [0, 1] using the actual data range
    # This ensures positions properly span [0, 1]
    actual_min = quantile_values[0]   # 0th percentile of data_in_range
    actual_max = quantile_values[-1]  # 100th percentile of data_in_range

    # Avoid division by zero in edge cases
    if actual_max - actual_min > 0:
        positions = (quantile_values - actual_min) / (actual_max - actual_min)
    else:
        # Fallback to evenly spaced positions if data is constant
        positions = np.linspace(0, 1, n_colors)

    # Create continuous colormap with colors positioned at quantiles
    # This gives us smooth transitions while maintaining adaptive distribution
    cmap_ir = LinearSegmentedColormap.from_list('adaptive_infestation_continuous',
                                                list(zip(positions, base_colors)),
                                                N=256)

    # Print diagnostic info about adaptive coloring
    print("\n" + "="*60)
    print("ADAPTIVE COLORMAP DISTRIBUTION (Continuous):")
    print("="*60)
    print(f"Total color anchors: {n_colors}")
    print(f"Data range (transformed): {vmin_transformed:.3f} to {vmax_transformed:.3f}")
    print(f"Quantile positions: {len(positions)}")
    print("\nColor anchor points at data quantiles:")
    for i in range(n_colors):
        # Convert back to original scale for interpretation
        orig_val = np.power(quantile_values[i], 1/power_factor) * 100
        # Updated for 15 yellows, 3 oranges, 2 reds
        if i < 15:
            color_type = "Yellow"
        elif i < 18:
            color_type = "Orange"
        else:
            color_type = "Red"
        percentile = quantile_points[i]
        print(f"  Color {i+1:2d} ({color_type:6s}) at {percentile:3.0f}th percentile: {orig_val:5.1f}% infestation")

    # Plot infestation rate with adaptive continuous colormap
    im2 = ax2.imshow(ir_data_transformed,
                     extent=[ir_bounds.left, ir_bounds.right,
                            ir_bounds.bottom, ir_bounds.top],
                     cmap=cmap_ir,
                     vmin=vmin_transformed,
                     vmax=vmax_transformed,
                     interpolation='nearest',
                     alpha=1.0)

    # Add SSA boundaries
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
    ax2.text(PANEL_LABEL_X, PANEL_LABEL_Y, 'b', transform=ax2.transAxes,
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

    # Set exactly 5 colorbar ticks avoiding the max value
    # Use positions at 0%, 20%, 40%, 60%, 80% of the range
    tick_positions = [0.0, 0.2, 0.4, 0.6, 0.8]
    tick_values_transformed = [vmin_transformed + pos * (vmax_transformed - vmin_transformed)
                              for pos in tick_positions]

    # Set only these 5 ticks
    cbar2.set_ticks(tick_values_transformed)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=3, width=0.5, pad=2)

    # Convert back to actual percentages for labels
    # Since we used power transformation, convert back
    tick_labels_pct = []
    for tv in tick_values_transformed:
        # Inverse transformation: if y = x^0.15, then x = y^(1/0.15)
        original_val = np.power(tv, 1/power_factor)
        tick_labels_pct.append(f'{int(original_val*100)}')

    cbar2.ax.set_yticklabels(tick_labels_pct, fontsize=COLORBAR_TICK_SIZE)
    cbar2.outline.set_linewidth(0.5)

    # Add title
    if LEGEND_BG_ENABLED:
        ax2.text(TITLE_X_POS, TITLE_Y_POS, 'Infestation risk\nindex',
                 transform=ax2.transAxes,
                 fontsize=TITLE_FONTSIZE,
                 va='bottom', ha='left',
                 fontweight='normal',
                 linespacing=TITLE_LINESPACING,
                 zorder=10)

    # Fine-tune layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save figure with optimized settings for publication quality but reasonable size
    # CHANGE 3: Updated output paths for revision
    output_path = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED/figure_ssa_nature_quality_calibrated_v2.png'
    # 400 DPI is sufficient for Nature publications and reduces file size significantly
    plt.savefig(output_path, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"CALIBRATED figure saved to: {output_path}")

    # Also save as PDF (vector format, smaller file)
    output_path_pdf = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED/figure_ssa_nature_quality_calibrated_v2.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF version saved to: {output_path_pdf}")

    # CHANGE 4: plt.show() removed for non-interactive execution

    # Print statistics
    print("\nData Statistics (SSA only):")
    print(f"Agroecological similarity: {np.nanmin(sim_data)*100:.1f}% - {np.nanmax(sim_data)*100:.1f}%")
    print(f"  Using percentile range: {vmin_sim*100:.1f}% - {vmax_sim*100:.1f}% for visualization")
    print(f"Infestation rate: {np.nanmin(ir_data)*100:.1f}% - {np.nanmax(ir_data)*100:.1f}%")
    print(f"  Transformed range: {vmin_transformed:.3f} - {vmax_transformed:.3f}")

    return fig

if __name__ == "__main__":
    print("="*60)
    print("Creating CALIBRATED Nature publication figure...")
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
