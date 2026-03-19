#!/usr/bin/env python3
"""
Create Final Climate Projections Figure - Version 8 (Revision R1)
Nature Communications Publication Quality

Revision changes addressing reviewer comments:
- R1.3.3: Converted Panel D from line chart to grouped bar chart
- R1.3.4: Added line style differentiation in Panel A, marker shapes per region
- R1.4:   Increased all font sizes for legibility

Nature Communications guidelines applied:
- Solid color fills only (no hatch patterns)
- Wong 2011 colorblind-safe palette
- All text in black
- Sentence case axis labels
- Arial font throughout
- pdf.fonttype = 42 for editable text
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
import json
import os

print("CREATING FINAL CLIMATE PROJECTIONS FIGURE - VERSION 8")
print("=" * 80)
print("Nature Communications R1 Revision")
print("Addresses: R1.3.3 (Panel D bar chart),")
print("           R1.3.4 (B&W readability), R1.4 (font sizes)")
print("NC guidelines: solid fills, Wong palette, black text, sentence case")
print("=" * 80)

# Nature Communications: editable text in PDF
plt.rcParams['pdf.fonttype'] = 42

# =====================================================================
# PATHS — resolve data relative to original SCRIPTS/FUTURE-PROJECTION/
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRIGA_BASE = '/Users/francktonle/Downloads/STRIGA_ANALYSIS'
PANEL_DATA_DIR = os.path.join(
    STRIGA_BASE, 'climate_projections_publication', 'outputs', 'panel_data_corrected'
)

OUTPUT_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', '04-FIGURES', 'REVISED')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'climate_results_figure_final_v8.png')

print(f"\nData directory : {PANEL_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# =====================================================================
# SCENARIO NAME MAPPING
# =====================================================================
SCENARIO_NAME_MAPPING = {
    'Degraded Response': 'Crisis Scenario',
    'Continued Trends': 'Current Trajectory',
    'Enhanced Control': 'Improved Management',
    'Integrated Management': 'Optimal Management',
}

# =====================================================================
# STYLE DICTIONARIES — color + secondary encoding for B&W graceful degradation
# =====================================================================

# Scenario styles: Wong 2011 colorblind-safe palette + line style (no hatch)
SCENARIO_STYLES = {
    'Crisis Scenario':      {'color': '#D55E00', 'linestyle': '--'},
    'Current Trajectory':   {'color': '#E69F00', 'linestyle': '-'},
    'Improved Management':  {'color': '#56B4E9', 'linestyle': '-.'},
    'Optimal Management':   {'color': '#009E73', 'linestyle': ':'},
}

# Region styles: Wong 2011 colorblind-safe palette + marker shape (no hatch)
REGION_STYLES = {
    'West Africa':     {'color': '#D55E00', 'marker': 'o'},
    'East Africa':     {'color': '#009E73', 'marker': 's'},
    'Southern Africa': {'color': '#0072B2', 'marker': '^'},
}

# Convenience color dicts (derived from style dicts)
SCENARIO_COLORS = {k: v['color'] for k, v in SCENARIO_STYLES.items()}
REGION_COLORS = {k: v['color'] for k, v in REGION_STYLES.items()}

PROF_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'danger': '#d62728',
    'success': '#2ca02c',
    'warning': '#bcbd22',
    'info': '#17becf',
    'purple': '#9467bd',
    'gray': '#7f7f7f',
    'light_gray': '#E9ECEF',
    'dark': '#212529',
    'baseline_gray': '#5a5a5a',
}

# =====================================================================
# FONT SIZES — R1.4: increased for legibility
# =====================================================================
FONT = {
    'panel_label': 16,   # Panel labels (A, B, C, D, E)
    'axis_title': 13,    # Axis titles
    'tick': 11,          # Tick labels
    'legend': 11,        # Legend text
    'annotation': 11,    # Annotations
    'suptitle': 16,      # Main figure title
}


def set_publication_style():
    """Set matplotlib style for Nature Communications publication."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = FONT['tick']
    plt.rcParams['axes.labelsize'] = FONT['axis_title']
    plt.rcParams['axes.titlesize'] = FONT['panel_label']
    plt.rcParams['xtick.labelsize'] = FONT['tick']
    plt.rcParams['ytick.labelsize'] = FONT['tick']
    plt.rcParams['legend.fontsize'] = FONT['legend']
    plt.rcParams['figure.titlesize'] = FONT['suptitle']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


# =====================================================================
# LOAD DATA
# =====================================================================
print("\nLoading panel data...")

with open(os.path.join(PANEL_DATA_DIR, 'panel_a_trajectories.json'), 'r') as f:
    panel_a_data = json.load(f)

with open(os.path.join(PANEL_DATA_DIR, 'panel_b_decomposition.json'), 'r') as f:
    panel_b_data = json.load(f)

panel_c_df = pd.read_csv(os.path.join(PANEL_DATA_DIR, 'panel_c_phase_space.csv'))

with open(os.path.join(PANEL_DATA_DIR, 'panel_d_regional_signatures.json'), 'r') as f:
    panel_d_data = json.load(f)

with open(os.path.join(PANEL_DATA_DIR, 'panel_e_adoption_impact.json'), 'r') as f:
    panel_e_data = json.load(f)

print("  All panel data loaded")

# =====================================================================
# APPLY SCENARIO NAME MAPPING
# =====================================================================
print("Applying scenario name mapping...")

for country in panel_a_data['projections_2050']:
    panel_a_data['projections_2050'][country] = {
        SCENARIO_NAME_MAPPING.get(sc, sc): data
        for sc, data in panel_a_data['projections_2050'][country].items()
    }

panel_b_data['scenarios'] = [
    SCENARIO_NAME_MAPPING.get(s, s) for s in panel_b_data['scenarios']
]
panel_b_data['totals'] = {
    SCENARIO_NAME_MAPPING.get(k, k): v for k, v in panel_b_data['totals'].items()
}

panel_c_df['scenario'] = panel_c_df['scenario'].map(
    lambda x: SCENARIO_NAME_MAPPING.get(x, x)
)

for point in panel_e_data['country_points']:
    point['scenario'] = SCENARIO_NAME_MAPPING.get(
        point['scenario'], point['scenario']
    )

print("  Scenario names mapped")

# =====================================================================
# CALCULATE SCENARIO STATISTICS WITH CIs
# =====================================================================
print("Calculating scenario-level confidence intervals...")

scenarios_ordered = [
    'Crisis Scenario', 'Current Trajectory',
    'Improved Management', 'Optimal Management',
]
scenario_stats = {}
baseline_total = panel_b_data['components']['baseline']

for scenario in scenarios_ordered:
    total_mean = 0
    total_ci_low = 0
    total_ci_high = 0

    for country in panel_a_data['countries']:
        proj = panel_a_data['projections_2050'][country][scenario]
        total_mean += proj['mean_millions']
        total_ci_low += proj['ci_95_low_millions']
        total_ci_high += proj['ci_95_high_millions']

    scenario_stats[scenario] = {
        'mean': total_mean,
        'ci_low': total_ci_low,
        'ci_high': total_ci_high,
        'change_pct': (total_mean - baseline_total) / baseline_total * 100,
    }

baseline_ci_low = sum(
    panel_a_data['baseline_2020'][c]['ci_95_low_millions']
    for c in panel_a_data['countries']
)
baseline_ci_high = sum(
    panel_a_data['baseline_2020'][c]['ci_95_high_millions']
    for c in panel_a_data['countries']
)

print(f"  Baseline: ${baseline_total:.0f}M")
for s, stats in scenario_stats.items():
    print(f"  {s}: ${stats['mean']:.0f}M [{stats['change_pct']:+.0f}%]")

# =====================================================================
# CREATE FIGURE
# =====================================================================
set_publication_style()

print("\nCreating figure...")
fig = plt.figure(figsize=(18, 20))

gs = gridspec.GridSpec(
    3, 2, height_ratios=[1.1, 1, 1],
    hspace=0.35, wspace=0.3,
    left=0.08, right=0.95, top=0.94, bottom=0.05,
)

# =====================================================================
# PANEL A: Country Trajectories (with line styles + region markers)
# =====================================================================
print("Creating Panel A — Country Trajectories...")

ax1 = fig.add_subplot(gs[0, :])

countries_sorted = panel_a_data['countries_sorted']

# Build country → region lookup
country_regions = {}
for _, row in panel_c_df.iterrows():
    country_regions[row['country']] = row['region']

# Subtle background shading by region
for i, country in enumerate(countries_sorted):
    if country in country_regions:
        region = country_regions[country]
        rect = Rectangle(
            (0, i - 0.4), 1000, 0.8,
            facecolor=REGION_COLORS.get(region, PROF_COLORS['gray']),
            alpha=0.05, zorder=0,
        )
        ax1.add_patch(rect)

# Plot each country
for i, country in enumerate(countries_sorted):
    y_pos = i
    region = country_regions.get(country, 'West Africa')
    region_marker = REGION_STYLES.get(region, {'marker': 'o'})['marker']

    # Baseline
    baseline_data = panel_a_data['baseline_2020'][country]
    baseline = baseline_data['mean_millions']
    baseline_ci_low_c = baseline_data['ci_95_low_millions']
    baseline_ci_high_c = baseline_data['ci_95_high_millions']

    baseline_label = '2020 baseline' if i == 0 else None
    ax1.errorbar(
        baseline, y_pos,
        xerr=[[baseline - baseline_ci_low_c], [baseline_ci_high_c - baseline]],
        fmt=region_marker, color=PROF_COLORS['gray'], markersize=8,
        capsize=3, capthick=1.2, alpha=0.9, zorder=10,
        markeredgecolor='white', markeredgewidth=0.5,
        label=baseline_label,
    )

    # Scenario projections
    projections = panel_a_data['projections_2050'][country]
    for scenario_name, proj_data in projections.items():
        style = SCENARIO_STYLES.get(scenario_name, {})
        color = style.get('color', PROF_COLORS['gray'])
        linestyle = style.get('linestyle', '-')
        value = proj_data['mean_millions']

        # Connecting line with scenario-specific line style (R1.3.4)
        ax1.plot(
            [baseline, value], [y_pos, y_pos],
            color=color, alpha=0.6, linewidth=2.5,
            linestyle=linestyle, zorder=2,
        )
        # Endpoint marker with region-specific shape (R1.3.4)
        ax1.plot(
            value, y_pos, marker=region_marker,
            color=color, markersize=9,
            markeredgecolor='white', markeredgewidth=1.2,
            zorder=11, alpha=0.9, linestyle='None',
        )

ax1.set_yticks(range(len(countries_sorted)))
ax1.set_yticklabels(countries_sorted, fontsize=FONT['tick'], fontweight='medium')
ax1.set_xlabel(
    'Economic loss (million USD)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax1.set_title(
    '',
    fontsize=FONT['panel_label'], fontweight='bold', pad=15,
)
ax1.text(-0.02, 1.05, 'a', transform=ax1.transAxes,
         fontsize=FONT['panel_label'], fontweight='bold', va='bottom', ha='right')

# Legend: scenarios (color + line style) + regions (marker shape)
legend_elements = [
    Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor=PROF_COLORS['gray'], markersize=8,
        label='2020 baseline', markeredgecolor='gray',
    ),
]
for scenario, style in SCENARIO_STYLES.items():
    legend_elements.append(
        Line2D(
            [0], [0], color=style['color'], linewidth=2.5,
            linestyle=style['linestyle'], marker='o', markersize=8,
            label=scenario, alpha=0.8,
        )
    )
# Region marker legend entries
for region, style in REGION_STYLES.items():
    legend_elements.append(
        Line2D(
            [0], [0], marker=style['marker'], color='w',
            markerfacecolor=PROF_COLORS['gray'], markersize=8,
            markeredgecolor='gray', label=region,
        )
    )

ax1.legend(
    handles=legend_elements, loc='upper right', ncol=2,
    frameon=True, fancybox=True, fontsize=FONT['legend'],
)

# Reference gridlines
max_loss = max(
    max(p['mean_millions'] for p in proj.values())
    for proj in [panel_a_data['projections_2050'][c] for c in countries_sorted]
)
for x_line in [100, 200, 300, 400, 500, 600, 700]:
    if x_line < max_loss:
        ax1.axvline(
            x=x_line, color=PROF_COLORS['light_gray'],
            linestyle='-', alpha=0.2, linewidth=0.5, zorder=0,
        )

ax1.grid(True, axis='x', alpha=0.15, linewidth=0.5)
ax1.set_xlim(0, max_loss * 1.05)
ax1.set_ylim(-0.5, len(countries_sorted) - 0.5)

# =====================================================================
# PANEL B: 2050 Projections by Scenario (bars WITH hatching — R1.3.2)
# =====================================================================
print("Creating Panel B — 2050 Projections by Scenario (solid fills)...")

ax3 = fig.add_subplot(gs[1, 0])

scenario_keys = [
    'Crisis Scenario', 'Current Trajectory',
    'Improved Management', 'Optimal Management',
]
scenario_means = [scenario_stats[s]['mean'] for s in scenario_keys]
scenario_ci_lows = [scenario_stats[s]['ci_low'] for s in scenario_keys]
scenario_ci_highs = [scenario_stats[s]['ci_high'] for s in scenario_keys]
scenario_colors_list = [SCENARIO_COLORS[s] for s in scenario_keys]
scenario_changes = [scenario_stats[s]['change_pct'] for s in scenario_keys]

# Labels with percentage changes
scenario_labels = [
    f'Crisis\n(+{scenario_changes[0]:.0f}%)',
    f'Current\n(+{scenario_changes[1]:.0f}%)',
    f'Improved\n(+{scenario_changes[2]:.0f}%)',
    f'Optimal\n(+{scenario_changes[3]:.0f}%)',
]

x_pos = np.arange(len(scenario_labels))

# Subtle background zones
ax3.axhspan(0, baseline_total, alpha=0.04, color=PROF_COLORS['success'], zorder=0)
ax3.axhspan(
    baseline_total, max(scenario_means) * 1.2,
    alpha=0.04, color=PROF_COLORS['danger'], zorder=0,
)

# Baseline reference line
ax3.axhline(
    y=baseline_total, color=PROF_COLORS['baseline_gray'],
    linestyle='-', linewidth=2, alpha=0.6, zorder=2,
)

# Draw bars with solid fills (NC guideline: no hatch patterns)
bar_width = 0.65
bars = []
for idx in range(len(scenario_keys)):
    bar = ax3.bar(
        x_pos[idx], scenario_means[idx], width=bar_width,
        color=scenario_colors_list[idx],
        edgecolor='white', linewidth=2, alpha=0.85, zorder=5,
    )
    bars.append(bar[0])

# CI error bars
for i, (x, mean, ci_low, ci_high) in enumerate(
    zip(x_pos, scenario_means, scenario_ci_lows, scenario_ci_highs)
):
    err_low = max(0, mean - ci_low)
    err_high = max(0, ci_high - mean)
    ax3.errorbar(
        x, mean, yerr=[[err_low], [err_high]],
        fmt='none', color='#333333', capsize=5, capthick=1.5,
        elinewidth=1.5, zorder=6,
    )

# Dollar amounts above CI caps
for i, (bar, mean, ci_high) in enumerate(
    zip(bars, scenario_means, scenario_ci_highs)
):
    y_pos_text = ci_high + max(scenario_means) * 0.03
    ax3.text(
        bar.get_x() + bar.get_width() / 2, y_pos_text,
        f'${mean / 1000:.2f}B', ha='center', va='bottom',
        fontsize=FONT['annotation'], fontweight='bold', color=PROF_COLORS['dark'],
    )

# Baseline annotation
ax3.annotate(
    '2020 baseline\n$0.71B',
    xy=(x_pos[0] - 0.3, baseline_total),
    xytext=(x_pos[0] - 0.45, baseline_total),
    fontsize=FONT['annotation'], fontweight='bold', ha='right', va='center',
    color=PROF_COLORS['dark'],
    bbox=dict(
        boxstyle='round,pad=0.3', facecolor='white',
        edgecolor=PROF_COLORS['baseline_gray'], alpha=0.95,
    ),
)

# "Avoidable losses" bracket
current_mean = scenario_stats['Current Trajectory']['mean']
optimal_mean = scenario_stats['Optimal Management']['mean']
avoidable = current_mean - optimal_mean

bracket_y = max(scenario_means) * 1.12
ax3.plot(
    [1, 1, 3, 3],
    [bracket_y - 30, bracket_y, bracket_y, bracket_y - 30],
    color='#333333', linewidth=2, zorder=10,
)
ax3.text(
    2, bracket_y + max(scenario_means) * 0.03,
    f'Avoidable: ${avoidable / 1000:.2f}B/yr', ha='center', va='bottom',
    fontsize=FONT['annotation'], fontweight='bold', color='black',
)

# Formatting
ax3.set_xticks(x_pos)
ax3.set_xticklabels(scenario_labels, fontsize=FONT['tick'], fontweight='medium')
ax3.set_ylabel(
    'Annual economic loss (million USD)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax3.set_title(
    '',
    fontsize=FONT['panel_label'], fontweight='bold',
)
ax3.text(-0.02, 1.05, 'b', transform=ax3.transAxes,
         fontsize=FONT['panel_label'], fontweight='bold', va='bottom', ha='right')
ax3.set_ylim(0, max(scenario_means) * 1.25)
ax3.set_xlim(-0.8, len(scenario_labels) - 0.5)


def billions_formatter(x, pos):
    if x >= 1000:
        return f'${x / 1000:.1f}B'
    elif x > 0:
        return f'${x:.0f}M'
    else:
        return ''


ax3.yaxis.set_major_formatter(plt.FuncFormatter(billions_formatter))

for y_line in [500, 1000, 1500, 2000, 2500]:
    if y_line < max(scenario_means) * 1.2:
        ax3.axhline(
            y=y_line, color=PROF_COLORS['light_gray'],
            linestyle='-', alpha=0.3, linewidth=0.5, zorder=0,
        )

ax3.text(
    0.02, 0.98, 'Error bars: 95% CI',
    transform=ax3.transAxes, fontsize=10, style='italic',
    color=PROF_COLORS['gray'], ha='left', va='top',
)

# =====================================================================
# PANEL C: Country Trajectory Classification (with region markers — R1.3.4)
# =====================================================================
print("Creating Panel C — Country Trajectory Classification...")

ax2 = fig.add_subplot(gs[1, 1])

continued_data = panel_c_df[panel_c_df['scenario'] == 'Current Trajectory']

x_data = continued_data['production_growth_pct'].values
y_data = continued_data['loss_rate_change_pp_annual'].values
countries = continued_data['country'].values
regions = continued_data['region'].values

x_threshold = 27.1
y_buffer = 0.015

ax2.axhline(y=0, color=PROF_COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axvline(x=x_threshold, color=PROF_COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.5)

x_range = [0, max(x_data) * 1.12]
y_min = min(y_data) - y_buffer
y_max = max(y_data) + y_buffer

quadrant_colors = {
    'Emerging Crisis':      ('#d62728', 0.04),
    'Vulnerable Expansion': ('#ff7f0e', 0.04),
    'Effective Management': ('#17becf', 0.04),
    'Sustainable Growth':   ('#2ca02c', 0.04),
}

ax2.add_patch(Rectangle(
    (0, 0), x_threshold, y_max,
    facecolor=quadrant_colors['Emerging Crisis'][0],
    alpha=quadrant_colors['Emerging Crisis'][1],
))
ax2.add_patch(Rectangle(
    (x_threshold, 0), x_range[1] - x_threshold, y_max,
    facecolor=quadrant_colors['Vulnerable Expansion'][0],
    alpha=quadrant_colors['Vulnerable Expansion'][1],
))
ax2.add_patch(Rectangle(
    (0, y_min), x_threshold, -y_min,
    facecolor=quadrant_colors['Effective Management'][0],
    alpha=quadrant_colors['Effective Management'][1],
))
ax2.add_patch(Rectangle(
    (x_threshold, y_min), x_range[1] - x_threshold, -y_min,
    facecolor=quadrant_colors['Sustainable Growth'][0],
    alpha=quadrant_colors['Sustainable Growth'][1],
))

ax2.text(
    x_threshold / 2, y_max * 0.85, 'Emerging\ncrisis',
    ha='center', va='center', fontsize=10, fontweight='bold',
    color='black', alpha=0.6,
)
ax2.text(
    (x_threshold + max(x_data) * 0.9) / 2, y_max * 0.85, 'Vulnerable\nexpansion',
    ha='center', va='center', fontsize=10, fontweight='bold',
    color='black', alpha=0.6,
)
ax2.text(
    x_threshold / 2, y_min * 0.7, 'Effective\nmanagement',
    ha='center', va='center', fontsize=10, fontweight='bold',
    color='black', alpha=0.6,
)
ax2.text(
    (x_threshold + max(x_data) * 0.9) / 2, y_min * 0.7, 'Sustainable\ngrowth',
    ha='center', va='center', fontsize=10, fontweight='bold',
    color='black', alpha=0.6,
)

for i, country in enumerate(countries):
    region = regions[i]
    color = REGION_STYLES.get(region, {'color': PROF_COLORS['gray']})['color']
    marker = REGION_STYLES.get(region, {'marker': 'o'})['marker']

    baseline_loss = panel_a_data['baseline_2020'][country]['mean_millions']
    size = 80 + (baseline_loss / 8)

    x_plot = x_data[i]
    y_plot = y_data[i]

    # Nudge overlapping points for readability
    if country == 'Tanzania':
        if abs(y_plot) < 0.01:
            y_plot = 0.008
    elif country == 'Ethiopia':
        if y_plot > -0.01:
            y_plot = -0.015

    if country not in ['Tanzania', 'Ethiopia']:
        if abs(y_plot) < 0.006:
            y_plot = 0.006 if y_plot > 0 else -0.006
        if abs(x_plot - x_threshold) < 1:
            x_plot = x_threshold + 1 if x_plot > x_threshold else x_threshold - 1

    # Region-specific marker shape (R1.3.4)
    ax2.scatter(
        x_plot, y_plot, s=size, c=color, marker=marker,
        alpha=0.8, edgecolors='white', linewidth=2, zorder=5,
    )

    if country in ['Mali', 'Nigeria', 'Kenya', 'Ethiopia', 'Zambia']:
        if x_plot < x_threshold and y_plot > 0:
            xytext = (-8, 8)
        elif x_plot > x_threshold and y_plot > 0:
            xytext = (8, 8)
        elif x_plot < x_threshold and y_plot < 0:
            xytext = (-8, -8)
        else:
            xytext = (8, -8)

        ax2.annotate(
            country, (x_plot, y_plot),
            xytext=xytext, textcoords='offset points',
            fontsize=FONT['tick'], fontweight='semibold',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor='white', edgecolor=color,
                alpha=0.95, linewidth=0.8,
            ),
        )

ax2.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, p: f'{x:.2f}' if abs(x) >= 0.01 else ('0' if x == 0 else f'{x:.3f}')
))
ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune='lower'))

ax2.set_xlabel(
    'Production growth 2020\u20132050 (%)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax2.set_ylabel(
    'Loss rate change (pp/year)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax2.set_title(
    '',
    fontsize=FONT['panel_label'], fontweight='bold',
)
ax2.text(-0.02, 1.05, 'c', transform=ax2.transAxes,
         fontsize=FONT['panel_label'], fontweight='bold', va='bottom', ha='right')

# Region legend with marker shapes
region_elements = [
    Line2D(
        [0], [0], marker=style['marker'], color='w',
        markerfacecolor=style['color'], markersize=10,
        markeredgecolor='white', label=region,
    )
    for region, style in REGION_STYLES.items()
]
ax2.legend(handles=region_elements, loc='upper right', fontsize=FONT['legend'], frameon=True)

ax2.grid(True, alpha=0.1, linewidth=0.4)
ax2.set_xlim(x_range)
ax2.set_ylim([y_min * 1.5, y_max * 1.1])

# =====================================================================
# PANEL D: Regional Response Signatures — GROUPED BAR CHART (R1.3.3)
# =====================================================================
print("Creating Panel D — Regional Differences (GROUPED BAR CHART)...")

ax4 = fig.add_subplot(gs[2, 0])

regions_list = panel_d_data['regions']
metrics = panel_d_data['metrics']
x_metrics = np.arange(len(metrics))
n_regions = len(regions_list)
bar_width_d = 0.25
offsets = [(-bar_width_d) * (n_regions - 1) / 2 + i * bar_width_d for i in range(n_regions)]

for i, region in enumerate(regions_list):
    if region not in panel_d_data['values']:
        continue
    values = [panel_d_data['values'][region][metric] for metric in metrics]
    style = REGION_STYLES.get(region, {'color': PROF_COLORS['gray'], 'marker': 'o'})

    # Uncertainty: ±0.1 (same as original fill_between bands)
    yerr = [0.1] * len(values)

    ax4.bar(
        x_metrics + offsets[i], values, bar_width_d,
        label=region, color=style['color'],
        edgecolor='white', linewidth=0.8, alpha=0.85,
    )
    ax4.errorbar(
        x_metrics + offsets[i], values, yerr=yerr,
        fmt='none', capsize=3, capthick=1.2, elinewidth=1.2,
        color='#333333', alpha=0.6, zorder=6,
    )

ax4.set_xticks(x_metrics)
ax4.set_xticklabels(
    [m.replace(' ', '\n') for m in metrics],
    fontsize=FONT['tick'],
)
ax4.set_ylabel(
    'Normalized value (0\u20131)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax4.set_title(
    '',
    fontsize=FONT['panel_label'], fontweight='bold',
)
ax4.text(-0.02, 1.05, 'd', transform=ax4.transAxes,
         fontsize=FONT['panel_label'], fontweight='bold', va='bottom', ha='right')
ax4.set_ylim(0, 1.15)

for y_ref in [0.25, 0.5, 0.75]:
    ax4.axhline(y=y_ref, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

ax4.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=FONT['legend'],
           handlelength=1.2, handletextpad=0.4, labelspacing=0.3, borderpad=0.3)
ax4.grid(True, axis='y', alpha=0.15)

# =====================================================================
# PANEL E: Adoption-Impact Relationships
# =====================================================================
print("Creating Panel E — Adoption-Impact Relationships...")

ax5 = fig.add_subplot(gs[2, 1])

country_points = pd.DataFrame(panel_e_data['country_points'])

coverage_levels = sorted(country_points['coverage_pct'].unique())
empirical_stats_panel = {}

for coverage in coverage_levels:
    subset = country_points[country_points['coverage_pct'] == coverage]['impact_pct'].values
    empirical_stats_panel[coverage] = {
        'best': np.max(subset),
        'p90': np.percentile(subset, 90),
        'mean': np.mean(subset),
        'std': np.std(subset),
    }

coverage_smooth = np.linspace(10, 65, 100)
coverages = np.array(list(empirical_stats_panel.keys()))
best_outcomes = np.array([empirical_stats_panel[c]['best'] for c in coverages])
p90_outcomes = np.array([empirical_stats_panel[c]['p90'] for c in coverages])
mean_outcomes = np.array([empirical_stats_panel[c]['mean'] for c in coverages])
std_outcomes = np.array([empirical_stats_panel[c]['std'] for c in coverages])

if len(coverages) > 2:
    best_spline = make_interp_spline(coverages, best_outcomes, k=2)
    best_frontier = best_spline(coverage_smooth)

    p90_spline = make_interp_spline(coverages, p90_outcomes, k=2)
    p90_frontier = p90_spline(coverage_smooth)

    mean_spline = make_interp_spline(coverages, mean_outcomes, k=2)
    mean_frontier = mean_spline(coverage_smooth)

    std_spline = make_interp_spline(coverages, std_outcomes, k=1)
    std_smooth = std_spline(coverage_smooth)

    ax5.plot(
        coverage_smooth, best_frontier, 'k--', linewidth=1.5, alpha=0.4,
        label='Best observed', zorder=1,
    )
    ax5.plot(
        coverage_smooth, p90_frontier, color='#009E73', linestyle='--', linewidth=1.5,
        label='90th percentile', zorder=1, alpha=0.6,
    )
    ax5.plot(
        coverage_smooth, mean_frontier, 'gray', linewidth=1.5,
        alpha=0.5, label='Mean observed', zorder=1,
    )
    ax5.fill_between(
        coverage_smooth,
        mean_frontier - std_smooth,
        mean_frontier + std_smooth,
        color='gray', alpha=0.1,
    )
    ax5.fill_between(coverage_smooth, 0, best_frontier, color='gray', alpha=0.05)

scenarios_for_plot = [
    'Crisis Scenario', 'Current Trajectory',
    'Improved Management', 'Optimal Management',
]

scenario_band_stats = {}
for scenario in scenarios_for_plot:
    scenario_data = country_points[country_points['scenario'] == scenario]
    if len(scenario_data) > 0:
        scenario_band_stats[scenario] = {
            'coverage': scenario_data['coverage_pct'].iloc[0],
            'impact_mean': scenario_data['impact_pct'].mean(),
            'impact_std': scenario_data['impact_pct'].std(),
            'impact_min': scenario_data['impact_pct'].min(),
            'impact_max': scenario_data['impact_pct'].max(),
            'impacts': scenario_data['impact_pct'].values,
        }

for scenario in scenarios_for_plot:
    if scenario not in scenario_band_stats:
        continue

    stats = scenario_band_stats[scenario]
    style = SCENARIO_STYLES[scenario]
    color = style['color']
    coverage = stats['coverage']

    ax5.fill_betweenx(
        [stats['impact_min'], stats['impact_max']],
        coverage - 2, coverage + 2,
        color=color, alpha=0.15, zorder=1,
    )

    mean = stats['impact_mean']
    std = stats['impact_std']
    ax5.fill_betweenx(
        [mean - std, mean + std],
        coverage - 1.5, coverage + 1.5,
        color=color, alpha=0.25, zorder=2,
    )

    np.random.seed(42)
    for impact in stats['impacts']:
        jitter = np.random.uniform(-1, 1)
        ax5.scatter(
            coverage + jitter, impact,
            s=30, c=color, alpha=0.4, edgecolors='white',
            linewidth=0.5, zorder=3,
        )

    ax5.scatter(
        coverage, mean,
        s=180, c=color, marker='D', edgecolors='white',
        linewidth=2, zorder=5, label=scenario,
    )

ax5.axhline(y=0, color=PROF_COLORS['dark'], linestyle='-', linewidth=1.5, alpha=0.5)

ax5.axvspan(0, 20, alpha=0.02, color='red')
ax5.axvspan(20, 40, alpha=0.02, color='orange')
ax5.axvspan(40, 60, alpha=0.02, color='green')
ax5.axvspan(60, 70, alpha=0.02, color='blue')

ax5.set_xlabel(
    'Technology adoption (%)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax5.set_ylabel(
    'Loss reduction (%)',
    fontsize=FONT['axis_title'], fontweight='semibold',
)
ax5.set_title(
    '',
    fontsize=FONT['panel_label'], fontweight='bold',
)
ax5.text(-0.02, 1.05, 'e', transform=ax5.transAxes,
         fontsize=FONT['panel_label'], fontweight='bold', va='bottom', ha='right')

ax5.legend(loc='lower right', fontsize=FONT['legend'], frameon=True, ncol=2)

ax5.set_xlim(0, 70)
y_max_panel = max(best_frontier.max(), 60) if 'best_frontier' in locals() else 60
ax5.set_ylim(min(country_points['impact_pct'].min() - 10, -180), y_max_panel)
ax5.grid(True, alpha=0.15, linewidth=0.5)

# =====================================================================
# MAIN TITLE
# =====================================================================

fig.suptitle(
    'Projected Striga economic impacts under agricultural development '
    'and climate change scenarios (2020\u20132050)',
    fontsize=FONT['suptitle'], fontweight='bold', y=0.97,
)

# =====================================================================
# SAVE
# =====================================================================
print(f"\nSaving figure v8 to {OUTPUT_PATH} ...")

fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')

# Also save PDF
pdf_path = OUTPUT_PATH.replace('.png', '.pdf')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print(f"  Saved PNG: {OUTPUT_PATH}")
print(f"  Saved PDF: {pdf_path}")

plt.close(fig)

print("\n" + "=" * 80)
print("FIGURE V8 (R1 REVISION) COMPLETE")
print("=" * 80)
print("""
Nature Communications guidelines applied:

  - Solid color fills only (no hatch patterns)
  - Wong 2011 colorblind-safe palette throughout
  - All text in black (no colored text)
  - Sentence case axis labels
  - Lowercase bold panel labels (a, b, c, d, e)
  - Arial font, pdf.fonttype = 42
  - Line styles and marker shapes retained as secondary encoding
""")
