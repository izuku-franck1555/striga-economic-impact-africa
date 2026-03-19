#!/usr/bin/env python3
"""
Supplementary Figure S2: Benin zoomed — infestation rate at 10 km pixel resolution.

Purpose: Demonstrate that Figure 2's model predictions are gridded spatial
data at 10 km resolution, not point observations. Requested by Reviewer 1.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/francktonle/Downloads/STRIGA_ANALYSIS')

# ==============================================================================
# PARAMETERS
# ==============================================================================

BENIN_EXTENT = [0.5, 4.0, 6.0, 12.5]  # [xmin, xmax, ymin, ymax]

PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
PANEL_LABEL_SIZE = 16
COLORBAR_TICK_SIZE = 10
MAP_TICK_SIZE = 11
TITLE_FONTSIZE = 12

# ==============================================================================
# FONT SETUP
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
plt.rcParams['pdf.fonttype'] = 42

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_longitude_label(x):
    if x == 0: return '0°'
    return f'{int(x)}°'

def format_latitude_label(y):
    if y == 0: return '0°'
    return f'{int(y)}°'


def add_scale_bar(ax, x_frac, y_frac, length_km=100, n_segments=2,
                  bar_height=0.008, fontsize=10):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + x_frac * (xlim[1] - xlim[0])
    y0 = ylim[0] + y_frac * (ylim[1] - ylim[0])
    deg_per_km = 1.0 / (111.0 * np.cos(np.radians(y0)))
    total_deg = length_km * deg_per_km
    seg_deg = total_deg / n_segments
    data_height = bar_height * (ylim[1] - ylim[0])
    for i in range(n_segments):
        color = 'black' if i % 2 == 0 else 'white'
        rect = mpatches.FancyBboxPatch(
            (x0 + i * seg_deg, y0), seg_deg, data_height,
            boxstyle='square,pad=0', facecolor=color, edgecolor='black',
            linewidth=0.8, zorder=8)
        ax.add_patch(rect)
    label_y = y0 - data_height * 1.0
    ax.text(x0, label_y, '0', ha='center', va='top', fontsize=fontsize, zorder=8)
    ax.text(x0 + total_deg, label_y, f'{length_km} km',
            ha='center', va='top', fontsize=fontsize, zorder=8)


def add_north_arrow(ax, x_frac, y_frac, size=0.06, fontsize=11):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = xlim[0] + x_frac * (xlim[1] - xlim[0])
    cy = ylim[0] + y_frac * (ylim[1] - ylim[0])
    arrow_h = size * (ylim[1] - ylim[0])
    half_w = arrow_h * 0.22
    tri_filled = plt.Polygon(
        [(cx, cy + arrow_h), (cx + half_w, cy), (cx, cy + arrow_h * 0.3)],
        closed=True, facecolor='black', edgecolor='black', linewidth=0.8, zorder=8)
    tri_open = plt.Polygon(
        [(cx, cy + arrow_h), (cx - half_w, cy), (cx, cy + arrow_h * 0.3)],
        closed=True, facecolor='white', edgecolor='black', linewidth=0.8, zorder=8)
    ax.add_patch(tri_filled)
    ax.add_patch(tri_open)
    ax.text(cx, cy + arrow_h + arrow_h * 0.15, 'N',
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold', zorder=8)


# ==============================================================================
# MAIN
# ==============================================================================

def create_figure_s2():
    print("Loading data...")

    africa = gpd.read_file('Africa.shp').to_crs('EPSG:4326')
    benin = africa[africa['NAME'] == 'Benin']
    neighbors = africa[africa['NAME'].isin(
        ['Togo', 'Burkina Faso', 'Niger', 'Nigeria', 'Ghana'])]

    hermonthica = gpd.read_file('preprocessed_data/seed_points_hermonthica.geojson')
    asiatica = gpd.read_file('asiatica.geojson')
    if hermonthica.crs != 'EPSG:4326':
        hermonthica = hermonthica.to_crs('EPSG:4326')
    if asiatica.crs != 'EPSG:4326':
        asiatica = asiatica.to_crs('EPSG:4326')

    # Filter to Benin extent
    def in_extent(gdf, ext):
        x, y = gdf.geometry.x, gdf.geometry.y
        return gdf[(x >= ext[0]) & (x <= ext[1]) & (y >= ext[2]) & (y <= ext[3])]

    herm = in_extent(hermonthica, BENIN_EXTENT)
    asia = in_extent(asiatica, BENIN_EXTENT)
    print(f"  Benin area: {len(herm)} S. hermonthica, {len(asia)} S. asiatica")

    # Load IR raster — windowed read for Benin
    with rasterio.open('analysis_outputs/infestation_rate_calibrated.tif') as src:
        win = from_bounds(BENIN_EXTENT[0], BENIN_EXTENT[2],
                          BENIN_EXTENT[1], BENIN_EXTENT[3], src.transform)
        ir_data = src.read(1, window=win).astype(float)
        win_transform = rasterio.windows.transform(win, src.transform)
        # Calculate bounds for imshow extent
        win_left = win_transform.c
        win_top = win_transform.f
        win_right = win_left + ir_data.shape[1] * win_transform.a
        win_bottom = win_top + ir_data.shape[0] * win_transform.e

    # Handle nodata
    ir_data[(ir_data <= 0) | (ir_data == -9999)] = np.nan
    valid = ir_data[~np.isnan(ir_data)]
    print(f"  IR pixels in Benin: {len(valid):,}")
    print(f"  IR range: {np.nanmin(ir_data):.3f} – {np.nanmax(ir_data):.3f}")

    # Create figure — single panel, square-ish
    fig, ax = plt.subplots(1, 1, figsize=(7, 8))

    ax.set_xlim(BENIN_EXTENT[0], BENIN_EXTENT[1])
    ax.set_ylim(BENIN_EXTENT[2], BENIN_EXTENT[3])
    ax.set_aspect('equal', adjustable='box')

    # Light gray background for all land
    africa.plot(ax=ax, facecolor='#f0f0f0', edgecolor='none', zorder=0)

    # IR raster — crisp pixels with nearest interpolation
    im = ax.imshow(ir_data,
                   extent=[win_left, win_right, win_bottom, win_top],
                   cmap='YlOrRd', vmin=0, vmax=1,
                   interpolation='nearest', alpha=1.0, zorder=2)

    # Boundaries — Benin highlighted, neighbors lighter
    neighbors.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.4, zorder=3)
    benin.boundary.plot(ax=ax, linewidth=1.5, color='black', alpha=0.9, zorder=4)

    # Occurrence points
    if len(asia) > 0:
        ax.scatter(asia.geometry.x, asia.geometry.y,
                   s=25, c='#0072B2', marker='^', alpha=0.8, edgecolors='white',
                   linewidth=0.5, label=r'$\it{S.\ asiatica}$', zorder=5)
    if len(herm) > 0:
        ax.scatter(herm.geometry.x, herm.geometry.y,
                   s=25, c='#D55E00', marker='o', alpha=0.8, edgecolors='white',
                   linewidth=0.5, label=r'$\it{S.\ hermonthica}$', zorder=6)

    # Grid
    ax.grid(True, linestyle='-', linewidth=0.25, alpha=0.3, color='gray', zorder=1)

    # Ticks
    x_ticks = [1, 2, 3, 4]
    y_ticks = [7, 8, 9, 10, 11, 12]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([format_longitude_label(x) for x in x_ticks], fontsize=MAP_TICK_SIZE)
    ax.set_yticklabels([format_latitude_label(y) for y in y_ticks], fontsize=MAP_TICK_SIZE)

    # Colorbar — outside the map on the right, like Figure S1 panel b
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                             fontsize=COLORBAR_TICK_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=3, width=0.5, pad=2)
    cbar.set_label('Infestation risk index', fontsize=TITLE_FONTSIZE, labelpad=8)
    cbar.outline.set_linewidth(0.5)

    # Legend
    legend = ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=False,
                       edgecolor='#666666', facecolor='white', framealpha=0.92,
                       borderpad=0.5, markerscale=1.5, handletextpad=0.5,
                       labelspacing=0.5)
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(10)

    # Scale bar (100 km for this zoom level) — bottom-left
    add_scale_bar(ax, x_frac=0.05, y_frac=0.04, length_km=100)

    # North arrow — top-right
    add_north_arrow(ax, x_frac=0.92, y_frac=0.85)

    # Save
    output_dir = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED'
    output_png = os.path.join(output_dir, 'figure_s2_zoom.png')
    output_pdf = os.path.join(output_dir, 'figure_s2_zoom.pdf')

    plt.savefig(output_png, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nPNG saved: {output_png}")
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PDF saved: {output_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Figure S2 — Benin zoomed (10 km resolution)")
    print("=" * 60)
    create_figure_s2()
    print("\nFIGURE S2 COMPLETE")
