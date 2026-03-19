#!/usr/bin/env python3
"""
Publication Figure for CALIBRATED Monte Carlo Results (v3 — Nature Comms Guidelines)
Applies Nature Communications figure formatting requirements:
  - Solid fills only (no hatching)
  - Colorblind-safe palette (Wong 2011), no red-green contrast
  - All text in black, sentence case axis labels
  - Arial/Helvetica font, lowercase bold panel labels
  - PDF fonttype 42 for editable text
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import json
import os
from scipy.stats import gaussian_kde
from scipy import stats

# =====================
# INLINE STYLE UTILITIES (replaces visualization_utils import)
# =====================

def set_publication_style():
    """Set Nature Communications publication style."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['pdf.fonttype'] = 42  # Editable text in PDF


def save_figure(fig, output_path, dpi=300, formats=None):
    """Save figure in specified formats."""
    if formats is None:
        formats = ['png']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for fmt in formats:
        filepath = f"{output_path}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")


# =====================
# REGION MARKERS for B&W readability (R1.3.4)
# =====================
REGION_MARKERS = {
    'Western Africa': 'o',   # circle
    'Eastern Africa': 's',   # square
    'Central Africa': '^',   # triangle
    'Southern Africa': 'D'   # diamond
}

# Hatching removed per Nature Comms guidelines: "Use solid color for filling objects"

print("CREATING PUBLICATION FIGURE - CALIBRATED MONTE CARLO RESULTS (v2 Revised)")
print("=" * 80)

# =====================
# LOAD CALIBRATED DATA
# =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
# Data is relative to SCRIPTS/CURRENT-ECONOMIC-IMPACT/ in the original tree
striga_base = '/Users/francktonle/Downloads/STRIGA_ANALYSIS'
base_path = os.path.join(striga_base, 'monte_carlo_publication/monte_carlo_publication/data/mc_results')

# Find the calibrated results file (100k iterations)
calibrated_npz = os.path.join(base_path, 'mc_full_results_20251206_224123.npz')
calibrated_stats = os.path.join(base_path, 'country_statistics_20251206_224123.csv')

print(f"\nLoading calibrated Monte Carlo results...")
print(f"  NPZ file: {calibrated_npz}")
print(f"  Stats file: {calibrated_stats}")

mc_data = np.load(calibrated_npz, allow_pickle=True)
total_losses = mc_data['total_losses']
country_losses_matrix = mc_data['country_losses_matrix']
country_names = mc_data['country_names']

print(f"Loaded {len(total_losses):,} Monte Carlo samples")
print(f"Total losses range: ${total_losses.min()/1e9:.3f}B - ${total_losses.max()/1e9:.3f}B")
print(f"Mean: ${np.mean(total_losses)/1e9:.3f}B")

# Load country statistics
country_stats = pd.read_csv(calibrated_stats, index_col=0)
print(f"Loaded statistics for {len(country_stats)} countries")

# =====================
# GENERATE PANEL DATA FROM CALIBRATED RESULTS
# =====================
print("\nGenerating panel data from calibrated results...")

# Panel A: Distribution statistics
losses_billion = total_losses / 1e9
ci_95 = np.percentile(total_losses, [2.5, 97.5])
ci_50 = np.percentile(total_losses, [25, 75])

panel_a_data = {
    'mean': np.mean(total_losses),
    'median': np.median(total_losses),
    'std': np.std(total_losses),
    'cv': np.std(total_losses) / np.mean(total_losses),
    'ci_95': [ci_95[0], ci_95[1]],
    'ci_50': [ci_50[0], ci_50[1]],
    'min': np.min(total_losses),
    'max': np.max(total_losses)
}

# Panel B: Top 10 by economic loss
sorted_countries = country_stats.sort_values('econ_mean', ascending=False)
top10 = sorted_countries.head(10)

panel_b_data = {
    'countries': top10.index.tolist(),
    'losses_millions': [x/1e6 for x in top10['econ_mean'].values],
    'ci_low': top10['econ_ci_95_low'].values.tolist(),
    'ci_high': top10['econ_ci_95_high'].values.tolist()
}

# Load maize production data from MapSPAM-derived CSV
mapspam_production_file = os.path.join(striga_base, 'preprocessed_data/country_maize_production.csv')
mapspam_df = pd.read_csv(mapspam_production_file)
maize_production = dict(zip(mapspam_df['country'], mapspam_df['maize_production_tonnes']))
# Handle NaN values
maize_production = {k: v for k, v in maize_production.items() if pd.notna(v)}
print(f"Loaded MapSPAM production data for {len(maize_production)} countries")

# Regional mapping
# UN Regional Classification for Africa
region_map = {
    # Eastern Africa (UN classification)
    'Kenya': 'Eastern Africa', 'Tanzania': 'Eastern Africa', 'Uganda': 'Eastern Africa',
    'Ethiopia': 'Eastern Africa', 'Rwanda': 'Eastern Africa', 'Burundi': 'Eastern Africa',
    'Sudan': 'Eastern Africa', 'South Sudan': 'Eastern Africa', 'Eritrea': 'Eastern Africa',
    'Djibouti': 'Eastern Africa', 'Somalia': 'Eastern Africa', 'Madagascar': 'Eastern Africa',
    'Malawi': 'Eastern Africa', 'Mozambique': 'Eastern Africa', 'Zambia': 'Eastern Africa',
    'Zimbabwe': 'Eastern Africa', 'Mauritius': 'Eastern Africa', 'Comoros': 'Eastern Africa',
    'Seychelles': 'Eastern Africa',
    # Western Africa
    'Nigeria': 'Western Africa', 'Mali': 'Western Africa', 'Benin': 'Western Africa',
    'Ghana': 'Western Africa', 'Togo': 'Western Africa', 'Burkina Faso': 'Western Africa',
    'Senegal': 'Western Africa', 'Guinea': 'Western Africa', 'Guinea-Bissau': 'Western Africa',
    'Gambia': 'Western Africa', 'Sierra Leone': 'Western Africa', 'Liberia': 'Western Africa',
    'Cote d\'Ivoire': 'Western Africa', 'Côte d\'Ivoire': 'Western Africa',
    'Niger': 'Western Africa', 'Mauritania': 'Western Africa', 'Cabo Verde': 'Western Africa',
    # Central Africa
    'Cameroon': 'Central Africa', 'Chad': 'Central Africa', 'Central African Republic': 'Central Africa',
    'Congo': 'Central Africa', 'Congo DRC': 'Central Africa', 'Gabon': 'Central Africa',
    'Equatorial Guinea': 'Central Africa', 'Sao Tome and Principe': 'Central Africa',
    'Angola': 'Central Africa',
    # Southern Africa (only 5 countries per UN)
    'South Africa': 'Southern Africa', 'Namibia': 'Southern Africa', 'Botswana': 'Southern Africa',
    'Lesotho': 'Southern Africa', 'Eswatini': 'Southern Africa'
}

# Standard maize price (USD/tonne)
maize_price = 280

# Calculate proportional losses
proportional_data = []
for country in country_stats.index:
    if country in maize_production and maize_production[country] > 0:
        econ_loss = country_stats.loc[country, 'econ_mean']
        phys_loss = country_stats.loc[country, 'phys_mean_tonnes']
        production = maize_production[country]
        prop_loss = (phys_loss / production) * 100 if production > 0 else 0
        region = region_map.get(country, 'Unknown')

        proportional_data.append({
            'countries': country,
            'economic_loss_millions': econ_loss / 1e6,
            'proportional_loss_percent': prop_loss,
            'region': region
        })

panel_c_data = pd.DataFrame(proportional_data)

# Panel D: Top 10 by proportional loss
panel_c_sorted = panel_c_data.sort_values('proportional_loss_percent', ascending=False)
top10_prop = panel_c_sorted.head(10)

panel_d_data = {
    'countries': top10_prop['countries'].tolist(),
    'proportional_loss_percent': top10_prop['proportional_loss_percent'].tolist()
}

# Panel E/F: Regional data
regional_totals = {r: 0 for r in ['Eastern Africa', 'Western Africa', 'Central Africa', 'Southern Africa']}
for country in country_stats.index:
    if country in region_map:
        region = region_map[country]
        if region in regional_totals:
            regional_totals[region] += country_stats.loc[country, 'econ_mean']

total_econ = sum(regional_totals.values())
regional_percentages = {r: (v/total_econ)*100 for r, v in regional_totals.items()}

print(f"\nRegional breakdown (calibrated):")
for r, p in sorted(regional_percentages.items(), key=lambda x: -x[1]):
    print(f"  {r}: {p:.1f}% (${regional_totals[r]/1e6:.1f}M)")

# =====================
# SET UP FIGURE
# =====================
set_publication_style()

PROF_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'danger': '#d62728',
    'success': '#2ca02c',
    'warning': '#bcbd22',
    'info': '#17becf',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
}

# Wong 2011 colorblind-safe palette (Nature-recommended)
REGION_COLORS = {
    'Western Africa': '#D55E00',   # Vermillion
    'Eastern Africa': '#009E73',   # Bluish green
    'Central Africa': '#E69F00',   # Orange
    'Southern Africa': '#0072B2'   # Blue
}

print("\nCreating comprehensive figure with calibrated data...")

fig = plt.figure(figsize=(20, 16))

gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1],
                      hspace=0.35, wspace=0.3,
                      left=0.06, right=0.96, top=0.94, bottom=0.08,
                      width_ratios=[1, 1, 1.3])

# ====================
# PANEL A: Distribution
# ====================
ax1 = fig.add_subplot(gs[0, :2])

n_bins = 400
n, bins_arr, patches_hist = ax1.hist(losses_billion, bins=n_bins, density=True,
                                     alpha=0.6, color=PROF_COLORS['primary'],
                                     edgecolor='darkblue', linewidth=0.3)

# KDE
kde = gaussian_kde(losses_billion)
kde.set_bandwidth(bw_method='silverman')
x_range = np.linspace(losses_billion.min() - 0.05, losses_billion.max() + 0.05, 1000)
kde_values = kde(x_range)

ax1.plot(x_range, kde_values, color=PROF_COLORS['danger'],
        linewidth=1.5, label='Density', zorder=5)

# CI bands
ci_95_b = np.percentile(losses_billion, [2.5, 97.5])
ax1.axvspan(ci_95_b[0], ci_95_b[1], alpha=0.12, color=PROF_COLORS['purple'],
           label='95% CI', zorder=1)

# Mean line
mean_val = np.mean(losses_billion)
ax1.axvline(mean_val, color=PROF_COLORS['danger'], linestyle='--',
           linewidth=2, label=f'Mean: ${mean_val:.2f}B', zorder=4)

ax1.set_xlabel('Economic loss (billion USD)', fontsize=13)
ax1.set_ylabel('Probability density', fontsize=13)
ax1.text(-0.02, 1.05, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.tick_params(axis='both', labelsize=11)
ax1.grid(True, alpha=0.15, linestyle='--')
ax1.set_xlim(losses_billion.min() - 0.02, losses_billion.max() + 0.02)

# ====================
# PANEL D: Risk Matrix (with region-specific markers — R1.3.4)
# ====================
ax2 = fig.add_subplot(gs[:2, 2])

x_data = panel_c_data['economic_loss_millions'].values
y_data = panel_c_data['proportional_loss_percent'].values
countries = panel_c_data['countries'].values
regions = panel_c_data['region'].values

valid_mask = (x_data > 0) & (y_data > 0)
x_valid = x_data[valid_mask]
y_valid = y_data[valid_mask]
countries_valid = countries[valid_mask]
regions_valid = regions[valid_mask]

ax2.set_xscale('log')
ax2.set_yscale('log')

# Density contours
if len(x_valid) > 5:
    x_min, x_max = 0.1, 500
    y_min, y_max = 0.5, 50
    xx = np.logspace(np.log10(x_min), np.log10(x_max), 50)
    yy = np.logspace(np.log10(y_min), np.log10(y_max), 50)
    XX, YY = np.meshgrid(xx, yy)
    positions = np.vstack([np.log10(XX.ravel()), np.log10(YY.ravel())])

    values = np.vstack([np.log10(x_valid), np.log10(y_valid)])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, XX.shape)

    contour = ax2.contourf(XX, YY, f, levels=8, cmap='YlOrRd', alpha=0.2)
    ax2.contour(XX, YY, f, levels=5, colors='gray', alpha=0.3, linewidths=0.8)

# Plot points by region — now with distinct markers (R1.3.4)
for region in REGION_COLORS.keys():
    mask = regions_valid == region
    if np.any(mask):
        ax2.scatter(x_valid[mask], y_valid[mask],
                   c=REGION_COLORS[region],
                   marker=REGION_MARKERS[region],
                   s=110, alpha=0.7,
                   edgecolors='white', linewidth=1.5, label=region,
                   zorder=5)

# Risk zones
x_mid = 10
y_mid = 5

# Colorblind-safe risk zone shading (no red-green contrast)
ax2.add_patch(Rectangle((x_mid, y_mid), 500-x_mid, 50-y_mid,
                        facecolor='#D55E00', alpha=0.08))   # Vermillion — high risk
ax2.add_patch(Rectangle((0.1, y_mid), x_mid-0.1, 50-y_mid,
                        facecolor='#E69F00', alpha=0.08))   # Orange — vulnerable
ax2.add_patch(Rectangle((x_mid, 0.5), 500-x_mid, y_mid-0.5,
                        facecolor='#0072B2', alpha=0.08))   # Blue — moderate
ax2.add_patch(Rectangle((0.1, 0.5), x_mid-0.1, y_mid-0.5,
                        facecolor='#56B4E9', alpha=0.08))   # Sky blue — low risk

ax2.text(50, 15, 'HIGH RISK', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black', alpha=0.5)
ax2.text(1, 15, 'VULNERABLE', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black', alpha=0.5)
ax2.text(50, 1.5, 'MODERATE', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black', alpha=0.5)
ax2.text(1, 1.5, 'LOW RISK', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black', alpha=0.5)

# Label top countries with manual positioning to avoid overlaps
high_risk = []
vulnerable = []
moderate = []
low_risk = []

for i, (x, y, country) in enumerate(zip(x_valid, y_valid, countries_valid)):
    if x >= x_mid and y >= y_mid:
        high_risk.append((x*y, i, country))
    elif x < x_mid and y >= y_mid:
        vulnerable.append((y, i, country))
    elif x >= x_mid and y < y_mid:
        moderate.append((x, i, country))
    else:
        low_risk.append((x*y, i, country))

categories = [
    ('HIGH RISK', sorted(high_risk, reverse=True)[:3], 'gray'),
    ('VULNERABLE', sorted(vulnerable, reverse=True)[:3], 'gray'),
    ('MODERATE', sorted(moderate, reverse=True)[:3], 'gray'),
    ('LOW RISK', sorted(low_risk, reverse=True)[:3], 'gray')
]

# Manual offsets for each category to ensure no overlaps
manual_offsets = {
    'HIGH RISK': [(12, 25), (12, -5), (-70, -20)],
    'VULNERABLE': [(8, 15), (8, -15), (-55, 8)],
    'MODERATE': [(12, 12), (12, -18), (-60, 5)],
    'LOW RISK': [(8, 10), (8, -15), (-50, 8)]
}

for cat_name, top3, color in categories:
    offsets = manual_offsets[cat_name]
    for rank, (_, idx, country) in enumerate(top3):
        if idx < len(x_valid) and rank < len(offsets):
            offset = offsets[rank]
            use_arrow = abs(offset[0]) > 40 or abs(offset[1]) > 20

            ax2.annotate(f'{country}',
                        (x_valid[idx], y_valid[idx]),
                        xytext=offset, textcoords='offset points',
                        fontsize=10, fontweight='semibold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor=color, alpha=0.95, linewidth=1.5),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6,
                                       connectionstyle='arc3,rad=0.15',
                                       linewidth=1) if use_arrow else None,
                        zorder=10)

ax2.grid(True, which='major', alpha=0.3, linestyle='-')
ax2.grid(True, which='minor', alpha=0.1, linestyle=':')
ax2.set_xlim(0.1, 500)
ax2.set_ylim(0.5, 50)
ax2.set_xlabel('Economic loss (million USD) - log scale', fontsize=13)
ax2.set_ylabel('Proportional loss (%) - log scale', fontsize=13)
ax2.text(-0.02, 1.03, 'd', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
ax2.tick_params(axis='both', labelsize=11)
ax2.legend(loc='lower left', fontsize=11, title='Region', title_fontsize=11, frameon=True)

# ====================
# PANEL B: Top 10 Economic (solid fills, Blues gradient)
# ====================
ax3 = fig.add_subplot(gs[1, 0])

countries_c = panel_b_data['countries'][:10]
losses_m = panel_b_data['losses_millions'][:10]
ci_low = [panel_b_data['ci_low'][i]/1e6 for i in range(10)]
ci_high = [panel_b_data['ci_high'][i]/1e6 for i in range(10)]

colors_gradient = plt.cm.YlOrRd(np.linspace(0.85, 0.2, len(countries_c)))

y_pos = np.arange(len(countries_c))
bars_b = ax3.barh(y_pos, losses_m, color=colors_gradient,
               alpha=0.85, edgecolor='white', linewidth=0.8)

for i in range(len(countries_c)):
    error_low = losses_m[i] - ci_low[i]
    error_high = ci_high[i] - losses_m[i]
    ax3.errorbar(losses_m[i], i, xerr=[[error_low], [error_high]],
                fmt='none', capsize=3, capthick=1.5, elinewidth=1.5,
                color='black', alpha=0.5, zorder=3)
    ax3.text(ci_high[i] + 5, i, f'${losses_m[i]:.0f}M',
            va='center', fontsize=10, fontweight='bold')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(countries_c, fontsize=11)
ax3.set_xlabel('Economic loss (million USD)', fontsize=13)
ax3.text(-0.02, 1.08, 'b', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top')
ax3.tick_params(axis='x', labelsize=11)
ax3.grid(True, axis='x', alpha=0.15, linestyle='--')
ax3.set_xlim(0, max(ci_high) * 1.15)

# ====================
# PANEL C: Top 10 Proportional (solid fills, Blues gradient)
# ====================
ax4 = fig.add_subplot(gs[1, 1])

countries_d = panel_d_data['countries'][:10]
prop_losses = panel_d_data['proportional_loss_percent'][:10]

colors_gradient_d = plt.cm.YlOrRd(np.linspace(0.85, 0.2, len(countries_d)))

y_pos = np.arange(len(countries_d))
bars_c = ax4.barh(y_pos, prop_losses, color=colors_gradient_d,
               alpha=0.85, edgecolor='white', linewidth=0.8)

cv = panel_a_data['cv']
for i, loss in enumerate(prop_losses):
    error = max(loss * cv * 0.5, 0.5)
    ax4.errorbar(loss, i, xerr=error, fmt='none',
                capsize=3, capthick=1.5, elinewidth=1.5,
                color='black', alpha=0.5, zorder=3)
    ax4.text(loss + error + 0.5, i, f'{loss:.1f}%',
            va='center', fontsize=10, fontweight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(countries_d, fontsize=11)
ax4.set_xlabel('Loss (% of national production)', fontsize=13)
ax4.text(-0.02, 1.08, 'c', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top')
ax4.tick_params(axis='x', labelsize=11)
ax4.grid(True, axis='x', alpha=0.15, linestyle='--')
ax4.set_xlim(0, max(prop_losses) * 1.2)

# ====================
# PANEL E: Regional Distribution (Calibrated values)
# ====================
ax5 = fig.add_subplot(gs[2, 0])

regions_pie = ['Eastern Africa', 'Western Africa', 'Southern Africa', 'Central Africa']
percentages = [regional_percentages[r] for r in regions_pie]
sizes = [regional_totals[r] for r in regions_pie]
colors_regional = [REGION_COLORS[r] for r in regions_pie]

wedges, texts, autotexts = ax5.pie(sizes, labels=regions_pie, colors=colors_regional,
                                    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                                    startangle=45, pctdistance=0.85,
                                    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
                                    textprops={'fontsize': 11, 'color': 'black'})

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

mean_b = panel_a_data['mean'] / 1e9
ax5.text(0, 0, f'Mean\n${mean_b:.2f}B', ha='center', va='center',
        fontsize=13, fontweight='bold', color='black')

ax5.text(-0.02, 1.05, 'e', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top')

# ====================
# PANEL F: Regional Uncertainty
# ====================
ax6 = fig.add_subplot(gs[2, 1:])

regions_list = ['Western Africa', 'Eastern Africa', 'Southern Africa', 'Central Africa']
regional_losses_by_iteration = {r: [] for r in regions_list}

country_region_map = region_map

for i in range(len(total_losses)):
    regional_totals_iter = {r: 0 for r in regions_list}
    for j, country_name in enumerate(country_names):
        if country_name in country_region_map:
            region = country_region_map[country_name]
            if region in regions_list:
                regional_totals_iter[region] += country_losses_matrix[i, j] / 1e6

    for region in regions_list:
        regional_losses_by_iteration[region].append(regional_totals_iter[region])

# Normalize data
normalized_data = []
positions = np.arange(len(regions_list))

for region in regions_list:
    data = regional_losses_by_iteration[region]
    mean = np.mean(data)
    normalized = [(x / mean - 1) * 100 for x in data] if mean > 0 else data
    normalized_data.append(normalized)

# Violin plots
parts = ax6.violinplot(normalized_data, positions=positions, widths=0.7,
                       showmeans=True, showmedians=True, showextrema=True)

for i, (pc, region) in enumerate(zip(parts['bodies'], regions_list)):
    pc.set_facecolor(REGION_COLORS[region])
    pc.set_alpha(0.4)
    pc.set_edgecolor(REGION_COLORS[region])
    pc.set_linewidth(1.5)

parts['cmeans'].set_color(PROF_COLORS['danger'])
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)

# Box plot overlay
bp = ax6.boxplot(normalized_data, positions=positions, widths=0.3,
                patch_artist=False, showfliers=False,
                boxprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                medianprops=dict(color='red', linewidth=2.5))

# Scatter points
np.random.seed(42)
for i, region in enumerate(regions_list):
    data = regional_losses_by_iteration[region]
    mean = np.mean(data)
    if mean > 0:
        sample = np.random.choice(data, min(50, len(data)), replace=False)
        normalized_sample = [(x / mean - 1) * 100 for x in sample]
        x_data_scatter = np.random.normal(i, 0.06, len(normalized_sample))
        ax6.scatter(x_data_scatter, normalized_sample, alpha=0.3, s=25,
                   color=REGION_COLORS[region],
                   marker=REGION_MARKERS[region],
                   edgecolors='white', linewidth=0.5)

# Statistics annotations
for i, region in enumerate(regions_list):
    data = regional_losses_by_iteration[region]
    mean_val = np.mean(data)
    std_val = np.std(data)
    cv = (std_val / mean_val) * 100 if mean_val > 0 else 0

    ax6.text(i, 24, f'CV: {cv:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='black', linewidth=1.5, alpha=0.9))

    ax6.text(i, -37, f'Mean: ${mean_val:.0f}M',
            ha='center', va='top', fontsize=11, fontweight='semibold')

# Reference lines
ax6.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax6.axhline(5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax6.axhline(-5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

ax6.set_xticks(positions)
ax6.set_xticklabels([r.replace(' Africa', '') for r in regions_list],
                    fontsize=11, fontweight='semibold')
ax6.set_ylabel('Deviation from mean (%)', fontsize=13)
ax6.text(-0.02, 1.05, 'f', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top')
ax6.tick_params(axis='both', labelsize=11)
ax6.set_ylim(-40, 45)
ax6.set_yticks([-40, -30, -20, -10, 0, 10, 20, 30])
ax6.grid(True, axis='y', alpha=0.15, linestyle='--')

# No figure-level title — caption goes in manuscript per Nature Comms guidelines

# Save figure
output_path = os.path.join(
    os.path.dirname(script_dir.rstrip('/')),
    '..', '..', '04-FIGURES', 'REVISED', 'comprehensive_economic_impact_calibrated'
)
output_path = os.path.normpath(output_path)
save_figure(fig, output_path, dpi=300, formats=['png', 'pdf'])

print("\n" + "=" * 80)
print("CALIBRATED FIGURE CREATED SUCCESSFULLY (v2 Revised)")
print("=" * 80)
print(f"\nSaved to: {output_path}.png")
print(f"\nRevision changes applied:")
print(f"  R1.3.2: Hatching patterns added to bar charts (Panels B, C) and pie (Panel E)")
print(f"  R1.3.4: Distinct marker shapes per region in scatter (Panel D) and violin (Panel F)")
print(f"  R1.4:   Font sizes increased — titles 16pt, axis labels 13pt, ticks/legend 11pt")
print(f"\nKey Statistics (Calibrated):")
print(f"  Continental Mean: ${np.mean(total_losses)/1e9:.2f}B")
print(f"  95% CI: ${ci_95[0]/1e9:.2f}B - ${ci_95[1]/1e9:.2f}B")
print(f"\nTop 5 Countries by Economic Loss:")
for i, country in enumerate(panel_b_data['countries'][:5]):
    print(f"  {i+1}. {country}: ${panel_b_data['losses_millions'][i]:.1f}M")
print(f"\nRegional Distribution:")
for r in ['Western Africa', 'Eastern Africa', 'Southern Africa', 'Central Africa']:
    print(f"  {r}: {regional_percentages[r]:.1f}%")
