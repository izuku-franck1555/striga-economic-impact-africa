#!/usr/bin/env python3
"""
Create Nature-quality Supplementary Figure S1:
  a) Striga occurrence points (S. hermonthica in red, S. asiatica in blue)
  b) Maize production across Africa (SPAM 2020, log color scale)

Visual style matches Figure 2 (create_nature_quality_maps_ssa.py).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# Set working directory for data access
os.chdir('/Users/francktonle/Downloads/STRIGA_ANALYSIS')

# ==============================================================================
# ADJUSTABLE POSITIONING PARAMETERS — MATCHED TO FIGURE 2
# ==============================================================================

# MAP EXTENT — IDENTICAL TO FIGURE 2
extent_africa = [-20, 55, -35, 38]

# COLORBAR POSITIONING (panel b only) — back to left (original position)
COLORBAR_X_POS = 0.109
COLORBAR_Y_POS = 0.03
COLORBAR_WIDTH = 0.04
COLORBAR_HEIGHT = 0.35

# TITLE POSITIONING (above colorbar)
TITLE_X_POS = 0.065
TITLE_Y_POS = 0.41
TITLE_FONTSIZE = 12
TITLE_LINESPACING = 1.2

# WHITE BACKGROUND RECTANGLE FOR COLORBAR AREA (panel b)
LEGEND_BG_X = 0.055
LEGEND_BG_Y = 0.01
LEGEND_BG_WIDTH = 0.16
LEGEND_BG_HEIGHT = 0.468
LEGEND_BG_ALPHA = 0.95

# PANEL LABEL POSITIONING
PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
PANEL_LABEL_SIZE = 16

# TICK LABEL SIZES — MATCHED TO FIGURE 2 v2
COLORBAR_TICK_SIZE = 10
MAP_TICK_SIZE = 11

# OCCURRENCE LEGEND (panel a) — positioned in lower-right
OCCUR_LEGEND_BG_X = 0.72
OCCUR_LEGEND_BG_Y = 0.01
OCCUR_LEGEND_BG_WIDTH = 0.26
OCCUR_LEGEND_BG_HEIGHT = 0.15

# ==============================================================================
# FONT SETUP — IDENTICAL TO FIGURE 2
# ==============================================================================

available_fonts = [f.name for f in fm.fontManager.ttflist]
font_preferences = ['Harding', 'Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
selected_font = 'DejaVu Sans'

for font in font_preferences:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [selected_font]
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['pdf.fonttype'] = 42  # Ensure editable text in PDFs (Nature requirement)

# ==============================================================================
# HELPER FUNCTIONS — IDENTICAL TO FIGURE 2
# ==============================================================================

def format_longitude_label(x):
    if x == 0:
        return '0°'
    elif x < 0:
        return f'{int(x)}°'
    else:
        return f'{int(x)}°'

def format_latitude_label(y):
    if y == 0:
        return '0°'
    elif y < 0:
        return f'{int(y)}°'
    else:
        return f'{int(y)}°'

def add_legend_background(ax, x=LEGEND_BG_X, y=LEGEND_BG_Y,
                          width=LEGEND_BG_WIDTH, height=LEGEND_BG_HEIGHT,
                          alpha=LEGEND_BG_ALPHA):
    """Add white background rectangle for legend/colorbar area."""
    rect = mpatches.Rectangle(
        (x, y), width, height,
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='none',
        alpha=alpha,
        zorder=2
    )
    ax.add_patch(rect)


def add_scale_bar(ax, x_frac, y_frac, length_km=500, n_segments=2,
                  bar_height=0.008, fontsize=10):
    """Add a professional segmented scale bar using axes-relative positioning.

    Uses transAxes so placement never conflicts with data elements.
    Draws alternating black/white segments like cartographic standards.
    """
    from matplotlib.transforms import Bbox, TransformedBbox

    # Get data coordinates at the desired axes-fraction position
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0_data = xlim[0] + x_frac * (xlim[1] - xlim[0])
    y0_data = ylim[0] + y_frac * (ylim[1] - ylim[0])

    # Convert km to degrees at this latitude
    lat_rad = np.radians(y0_data)
    deg_per_km = 1.0 / (111.0 * np.cos(lat_rad))
    total_deg = length_km * deg_per_km
    seg_deg = total_deg / n_segments

    # Bar height in data coordinates
    data_height = bar_height * (ylim[1] - ylim[0])

    # Draw alternating segments
    for i in range(n_segments):
        x_start = x0_data + i * seg_deg
        color = 'black' if i % 2 == 0 else 'white'
        rect = mpatches.FancyBboxPatch(
            (x_start, y0_data), seg_deg, data_height,
            boxstyle='square,pad=0',
            facecolor=color, edgecolor='black', linewidth=0.8,
            zorder=8
        )
        ax.add_patch(rect)

    # End labels: 0 and total km
    label_y = y0_data - data_height * 1.0
    ax.text(x0_data, label_y, '0', ha='center', va='top',
            fontsize=fontsize, zorder=8)
    ax.text(x0_data + total_deg, label_y, f'{length_km} km',
            ha='center', va='top', fontsize=fontsize, zorder=8)


def add_north_arrow(ax, x_frac, y_frac, size=0.06, fontsize=11):
    """Add a compass-style north arrow using axes-relative positioning.

    Draws a filled/unfilled dual-triangle needle with N label.
    """
    # Convert axes fractions to data coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = xlim[0] + x_frac * (xlim[1] - xlim[0])
    cy = ylim[0] + y_frac * (ylim[1] - ylim[0])

    # Arrow dimensions in data coordinates
    arrow_h = size * (ylim[1] - ylim[0])
    half_w = arrow_h * 0.22

    # Filled (east) half — black
    tri_filled = plt.Polygon(
        [(cx, cy + arrow_h), (cx + half_w, cy), (cx, cy + arrow_h * 0.3)],
        closed=True, facecolor='black', edgecolor='black', linewidth=0.8, zorder=8
    )
    # Open (west) half — white with black border
    tri_open = plt.Polygon(
        [(cx, cy + arrow_h), (cx - half_w, cy), (cx, cy + arrow_h * 0.3)],
        closed=True, facecolor='white', edgecolor='black', linewidth=0.8, zorder=8
    )
    ax.add_patch(tri_filled)
    ax.add_patch(tri_open)

    # N label
    ax.text(cx, cy + arrow_h + arrow_h * 0.15, 'N',
            ha='center', va='bottom', fontsize=fontsize,
            fontweight='bold', zorder=8)

# ==============================================================================
# MAIN FIGURE CREATION
# ==============================================================================

def create_figure_s1():
    """Create publication-quality Supplementary Figure S1."""

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")

    # Africa boundaries
    africa = gpd.read_file('Africa.shp')
    africa = africa.to_crs('EPSG:4326')

    # Striga occurrence points
    hermonthica = gpd.read_file('preprocessed_data/seed_points_hermonthica.geojson')
    asiatica = gpd.read_file('asiatica.geojson')

    # Ensure CRS match
    if hermonthica.crs != 'EPSG:4326':
        hermonthica = hermonthica.to_crs('EPSG:4326')
    if asiatica.crs != 'EPSG:4326':
        asiatica = asiatica.to_crs('EPSG:4326')

    n_herm = len(hermonthica)
    n_asia = len(asiatica)
    n_total = n_herm + n_asia
    print(f"  S. hermonthica: {n_herm:,} records")
    print(f"  S. asiatica:    {n_asia:,} records")
    print(f"  Total:          {n_total:,} records")

    # Maize production raster
    with rasterio.open('preprocessed_data/maize_africa_masked.tif') as src:
        maize_data = src.read(1).astype(float)
        maize_transform = src.transform
        maize_bounds = src.bounds
        maize_nodata = src.nodata

    # Handle nodata
    if maize_nodata is not None:
        maize_data[maize_data == maize_nodata] = np.nan
    maize_data[maize_data <= 0] = np.nan

    maize_valid = maize_data[~np.isnan(maize_data)]
    print(f"  Maize raster: {len(maize_valid):,} valid pixels")
    print(f"  Maize range: {np.nanmin(maize_data):.1f} – {np.nanmax(maize_data):.1f} tonnes")

    # ------------------------------------------------------------------
    # 2. Create figure — same dimensions as Figure 2
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Tick positions — identical to Figure 2
    x_ticks = [-10, 0, 10, 20, 30, 40, 50]
    y_ticks = [-30, -20, -10, 0, 10, 20, 30]

    # ------------------------------------------------------------------
    # PANEL a: STRIGA OCCURRENCES
    # ------------------------------------------------------------------
    ax1.set_xlim(extent_africa[0], extent_africa[1])
    ax1.set_ylim(extent_africa[2], extent_africa[3])
    ax1.set_aspect('equal', adjustable='box')

    # Africa background — light gray fill
    africa.plot(ax=ax1, facecolor='#f0f0f0', edgecolor='none', zorder=0)
    # Country boundaries
    africa.boundary.plot(ax=ax1, linewidth=0.5, color='black', alpha=0.7, zorder=1)

    # Plot occurrence points
    # Asiatica first (blue, behind) then hermonthica (red, on top)
    ax1.scatter(asiatica.geometry.x, asiatica.geometry.y,
                s=8, c='#2166ac', marker='^', alpha=0.7, edgecolors='none',
                label=r'$\it{S.\ asiatica}$', zorder=3)
    ax1.scatter(hermonthica.geometry.x, hermonthica.geometry.y,
                s=8, c='#b2182b', marker='o', alpha=0.7, edgecolors='none',
                label=r'$\it{S.\ hermonthica}$', zorder=4)

    # Grid — identical to Figure 2
    ax1.grid(True, linestyle='-', linewidth=0.25, alpha=0.3, color='gray', zorder=1)

    # Ticks
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels([format_longitude_label(x) for x in x_ticks], fontsize=MAP_TICK_SIZE)
    ax1.set_yticklabels([format_latitude_label(y) for y in y_ticks], fontsize=MAP_TICK_SIZE)

    # Species legend — compact, with clean frame, lower-right
    legend = ax1.legend(loc='lower right',
                        fontsize=10, frameon=True, fancybox=False,
                        edgecolor='#666666', facecolor='white',
                        framealpha=0.92, borderpad=0.5,
                        markerscale=2.2, handletextpad=0.5,
                        labelspacing=0.5)
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(10)

    # Scale bar — bottom-left, well inside the map
    add_scale_bar(ax1, x_frac=0.05, y_frac=0.06, length_km=500)
    # North arrow — top-right corner
    add_north_arrow(ax1, x_frac=0.92, y_frac=0.82)

    # Panel label — identical positioning to Figure 2
    ax1.text(PANEL_LABEL_X, PANEL_LABEL_Y, 'a', transform=ax1.transAxes,
             fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left', zorder=10)

    # ------------------------------------------------------------------
    # PANEL b: MAIZE PRODUCTION
    # ------------------------------------------------------------------
    ax2.set_xlim(extent_africa[0], extent_africa[1])
    ax2.set_ylim(extent_africa[2], extent_africa[3])
    ax2.set_aspect('equal', adjustable='box')

    # Africa background — white fill
    africa.plot(ax=ax2, facecolor='white', edgecolor='none', zorder=0)
    # Country boundaries
    africa.boundary.plot(ax=ax2, linewidth=0.5, color='black', alpha=0.7, zorder=1)

    # Maize raster with log-scale coloring
    vmin_maize = np.nanmin(maize_data[maize_data > 0])
    vmax_maize = np.nanmax(maize_data)

    im2 = ax2.imshow(maize_data,
                     extent=[maize_bounds.left, maize_bounds.right,
                             maize_bounds.bottom, maize_bounds.top],
                     cmap='YlGn',
                     norm=LogNorm(vmin=vmin_maize, vmax=vmax_maize),
                     interpolation='nearest',
                     alpha=1.0,
                     zorder=2)

    # Re-draw boundaries on top of raster
    africa.boundary.plot(ax=ax2, linewidth=0.5, color='black', alpha=0.7, zorder=3)

    # Grid
    ax2.grid(True, linestyle='-', linewidth=0.25, alpha=0.3, color='gray', zorder=1)

    # Ticks
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    ax2.set_xticklabels([format_longitude_label(x) for x in x_ticks], fontsize=MAP_TICK_SIZE)
    ax2.set_yticklabels([format_latitude_label(y) for y in y_ticks], fontsize=MAP_TICK_SIZE)

    # White background for colorbar area — identical to Figure 2
    add_legend_background(ax2)

    # Colorbar — identical approach to Figure 2
    cax2 = inset_axes(ax2,
                      width=f"{COLORBAR_WIDTH * 100}%",
                      height=f"{COLORBAR_HEIGHT * 100}%",
                      loc='lower left',
                      bbox_to_anchor=(COLORBAR_X_POS, COLORBAR_Y_POS, 1, 1),
                      bbox_transform=ax2.transAxes,
                      borderpad=0)

    cbar2 = plt.colorbar(im2, cax=cax2, orientation='vertical')

    # Format colorbar ticks as readable values on log scale
    import matplotlib.ticker as ticker

    tick_values = [1, 10, 100, 1000, 10000, 100000]
    tick_labels = ['1', '10', '100', '1K', '10K', '100K']
    # Filter ticks to those within data range
    valid_ticks = [(v, l) for v, l in zip(tick_values, tick_labels)
                   if vmin_maize <= v <= vmax_maize]
    if valid_ticks:
        tv, tl = zip(*valid_ticks)
        cbar2.set_ticks(list(tv))
        cbar2.ax.set_yticklabels(list(tl), fontsize=COLORBAR_TICK_SIZE)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=3, width=0.5, pad=2)
    cbar2.outline.set_linewidth(0.5)

    # Colorbar title — same approach as Figure 2
    ax2.text(TITLE_X_POS, TITLE_Y_POS, 'Maize production\n(tonnes)',
             transform=ax2.transAxes,
             fontsize=TITLE_FONTSIZE,
             va='bottom', ha='left',
             fontweight='normal',
             linespacing=TITLE_LINESPACING,
             zorder=10)

    # Scale bar — just right of the colorbar, close to it
    add_scale_bar(ax2, x_frac=0.22, y_frac=0.06, length_km=500)
    # North arrow — top-right corner
    add_north_arrow(ax2, x_frac=0.92, y_frac=0.82)

    # Panel label
    ax2.text(PANEL_LABEL_X, PANEL_LABEL_Y, 'b', transform=ax2.transAxes,
             fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left', zorder=10)

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    output_dir = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED'
    output_png = os.path.join(output_dir, 'figure_s1_revised.png')
    output_pdf = os.path.join(output_dir, 'figure_s1_revised.pdf')

    plt.savefig(output_png, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nPNG saved: {output_png}")

    plt.savefig(output_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF saved: {output_pdf}")

    plt.close(fig)
    print("\nFigure S1 generation complete.")
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Figure S1 — Supplementary Figure")
    print("  Panel a: Striga occurrence points")
    print("  Panel b: Maize production (SPAM 2020)")
    print("=" * 60)
    print(f"Font: {selected_font}")
    print(f"Extent: {extent_africa}")
    print("=" * 60)

    fig = create_figure_s1()

    print("\n" + "=" * 60)
    print("FIGURE S1 GENERATION COMPLETE")
    print("=" * 60)
