#!/usr/bin/env python3
"""
Input Data Sensitivity Analysis for Nature Communications R2.3

Addresses reviewer concern: "input data that is itself already modelled...
introduces uncertainty as to the quality of the input data."

Key insight: damage fraction df = IR × severity × dm is INDEPENDENT of
production and price. We pre-compute mean df from the existing 100K MC
results, then perturb production/price with fast raster algebra.
No need to re-run Monte Carlo.

Produces:
- Supplementary Table: perturbation results matrix
- Supplementary Figure: FAOSTAT validation + rank stability
- FAOSTAT cross-validation statistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr, kendalltau
import rasterio
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/francktonle/Downloads/STRIGA_ANALYSIS')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42

# =========================================================================
# DATA LOADING
# =========================================================================

print("=" * 70)
print("INPUT DATA SENSITIVITY ANALYSIS")
print("=" * 70)

# Load baseline MC results
mc_path = 'monte_carlo_publication/monte_carlo_publication/data/mc_results'
mc_data = np.load(f'{mc_path}/mc_full_results_20251206_224123.npz', allow_pickle=True)
baseline_stats = pd.read_csv(f'{mc_path}/country_statistics_20251206_224123.csv', index_col=0)

print(f"\nBaseline: {len(baseline_stats)} countries")
print(f"Total baseline loss: ${baseline_stats['econ_mean'].sum()/1e9:.2f}B")

# Load spatial data
with rasterio.open('preprocessed_data/maize_africa_masked.tif') as src:
    production = src.read(1).astype(float)

with rasterio.open('analysis_outputs/infestation_rate_calibrated.tif') as src:
    ir_calibrated = src.read(1).astype(float)

countries_raster = np.load('analysis_outputs/africa_countries_raster.npy')

# Load boundaries for country name mapping
africa = gpd.read_file('Africa.shp')
if africa.crs != 'EPSG:4326':
    africa = africa.to_crs('EPSG:4326')

# Build country ID → name mapping
country_id_to_name = {}
for idx, row in africa.iterrows():
    country_id_to_name[idx] = row.get('NAME', f'Country_{idx}')

# Load prices
prices_df = pd.read_csv('preprocessed_data/maize_prices_standardized.csv')
price_dict = dict(zip(prices_df['country'], prices_df['price_usd_per_tonne']))

# Load calibration thresholds
thresholds_df = pd.read_csv('monte_carlo_publication/calibration_thresholds.csv')
threshold_dict = dict(zip(thresholds_df['country_lookup'], thresholds_df['computed_threshold']))
print(f"  Loaded calibration thresholds for {len(threshold_dict)} countries")

# Species probability
try:
    p_herm = np.load('analysis_outputs/p_hermonthica_occurrence_based.npy')
except:
    p_herm = np.full_like(ir_calibrated, 0.5)

# =========================================================================
# PRE-COMPUTE BASELINE LOSS RATE PER PIXEL
# =========================================================================

print("\nPre-computing baseline damage fractions...")

# Mean severity parameters (from MC distributions)
MEAN_SEVERITY_HERM = 0.50  # S. hermonthica mean
MEAN_SEVERITY_ASIA = 0.20  # S. asiatica mean

# Mean control parameters
MEAN_RHO = 0.80           # Implementation efficiency
MEAN_ADOPTION = 0.15      # Continental average
MEAN_EFFICACY = 0.515     # Weighted tech efficacy

# Compute mean damage fraction per pixel
mean_severity = p_herm * MEAN_SEVERITY_HERM + (1 - p_herm) * MEAN_SEVERITY_ASIA
mean_dm = 1 - MEAN_RHO * MEAN_ADOPTION * MEAN_EFFICACY  # Residual damage factor

# Apply calibration: only pixels above country-specific threshold
ir_used = np.copy(ir_calibrated)
ir_used[ir_used <= 0] = 0
ir_used[ir_used == -9999] = 0

# Zero out pixels below each country's calibration threshold
for cid in np.unique(countries_raster):
    if cid < 0:
        continue
    cname = country_id_to_name.get(cid, None)
    if cname is None:
        continue
    threshold = threshold_dict.get(cname, 0)
    if threshold > 0:
        mask = countries_raster == cid
        ir_used[mask & (ir_used < threshold)] = 0

print(f"  Pixels with IR > 0 after calibration: {np.sum(ir_used > 0):,}")

mean_df = ir_used * mean_severity * mean_dm
mean_df = np.clip(mean_df, 0, 0.999)

# Loss multiplier: df/(1-df)
loss_multiplier = np.where(mean_df > 0, mean_df / (1 - mean_df), 0)

# Verify baseline
production_valid = np.where(production > 0, production, 0)
baseline_pixel_losses = production_valid * loss_multiplier * 280  # approx mean price

print(f"  Pixels with losses > 0: {np.sum(loss_multiplier > 0):,}")
print(f"  Approximate continental loss: ${np.sum(baseline_pixel_losses)/1e9:.2f}B")

# =========================================================================
# USE ACTUAL MC BASELINE (calibrated $1.04B from 100K iterations)
# =========================================================================

print("\nUsing actual MC baseline losses (calibrated)...")

# The MC country_statistics CSV has the true calibrated baseline
baseline_country_losses = {}
for country_name in baseline_stats.index:
    loss = baseline_stats.loc[country_name, 'econ_mean']
    if loss > 0:
        baseline_country_losses[country_name] = loss

unique_ids = np.unique(countries_raster)

# Sort by loss
baseline_sorted = sorted(baseline_country_losses.items(), key=lambda x: -x[1])
baseline_ranking = {name: rank for rank, (name, _) in enumerate(baseline_sorted)}

print(f"  Countries with losses: {len(baseline_country_losses)}")
print(f"  Total: ${sum(baseline_country_losses.values())/1e9:.2f}B")
print(f"  Top 5: {[f'{n}: ${v/1e6:.0f}M' for n, v in baseline_sorted[:5]]}")

# =========================================================================
# STATISTICAL CAPACITY CLASSIFICATION
# =========================================================================

HIGH_CAPACITY = ['Kenya', 'Nigeria', 'South Africa', 'Ethiopia', 'Tanzania', 'Ghana']
MEDIUM_CAPACITY = ['Uganda', 'Zambia', 'Cameroon', 'Malawi', 'Mozambique', 'Senegal']
# All others: LOW capacity

def get_capacity_range(country):
    if country in HIGH_CAPACITY:
        return 0.10  # ±10%
    elif country in MEDIUM_CAPACITY:
        return 0.20  # ±20%
    else:
        return 0.30  # ±30%

# =========================================================================
# PERTURBATION ANALYSIS
# =========================================================================

N_TRIALS = 1000
np.random.seed(42)

scenarios = {
    'Uniform ±10%': {'prod_range': 0.10, 'price_range': 0.0, 'capacity_weighted': False},
    'Uniform ±20%': {'prod_range': 0.20, 'price_range': 0.0, 'capacity_weighted': False},
    'Uniform ±30%': {'prod_range': 0.30, 'price_range': 0.0, 'capacity_weighted': False},
    'Capacity-weighted': {'prod_range': None, 'price_range': 0.20, 'capacity_weighted': True},
    'Price ±20%': {'prod_range': 0.0, 'price_range': 0.20, 'capacity_weighted': False},
    'Combined (capacity + price)': {'prod_range': None, 'price_range': 0.20, 'capacity_weighted': True},
}

results = {}

for scenario_name, params in scenarios.items():
    print(f"\n--- {scenario_name} ({N_TRIALS} trials) ---")

    trial_totals = []
    trial_rankings = []
    trial_country_losses = []

    for trial in range(N_TRIALS):
        perturbed_losses = {}

        for country_name, base_loss in baseline_country_losses.items():
            # Production perturbation (loss scales linearly with production)
            if params['capacity_weighted']:
                cap_range = get_capacity_range(country_name)
                prod_factor = 1 + np.random.uniform(-cap_range, cap_range)
            elif params['prod_range'] and params['prod_range'] > 0:
                prod_factor = 1 + np.random.uniform(-params['prod_range'], params['prod_range'])
            else:
                prod_factor = 1.0

            # Price perturbation (loss scales linearly with price)
            if params['price_range'] > 0:
                price_factor = 1 + np.random.uniform(-params['price_range'], params['price_range'])
            else:
                price_factor = 1.0

            # Perturbed loss = baseline × production factor × price factor
            perturbed_losses[country_name] = base_loss * prod_factor * price_factor

        total = sum(perturbed_losses.values())
        trial_totals.append(total)

        # Rank
        sorted_countries = sorted(perturbed_losses.items(), key=lambda x: -x[1])
        ranking = {name: rank for rank, (name, _) in enumerate(sorted_countries)}
        trial_rankings.append(ranking)
        trial_country_losses.append(perturbed_losses)

    # Compute metrics
    trial_totals = np.array(trial_totals)

    # Spearman rank correlation (average across trials)
    baseline_ranks = [baseline_ranking[c] for c in baseline_country_losses.keys()]
    spearman_rhos = []
    kendall_taus = []

    for trial_rank in trial_rankings:
        perturbed_ranks = [trial_rank.get(c, 99) for c in baseline_country_losses.keys()]
        rho, _ = spearmanr(baseline_ranks, perturbed_ranks)
        tau, _ = kendalltau(baseline_ranks, perturbed_ranks)
        spearman_rhos.append(rho)
        kendall_taus.append(tau)

    # Risk quintile stability
    baseline_quintiles = {}
    sorted_baseline = sorted(baseline_country_losses.items(), key=lambda x: -x[1])
    n_countries = len(sorted_baseline)
    for i, (name, _) in enumerate(sorted_baseline):
        baseline_quintiles[name] = i * 5 // n_countries  # 0-4 quintile

    quintile_changes = 0
    for trial_losses in trial_country_losses:
        sorted_trial = sorted(trial_losses.items(), key=lambda x: -x[1])
        for i, (name, _) in enumerate(sorted_trial):
            trial_q = i * 5 // n_countries
            if trial_q != baseline_quintiles.get(name, -1):
                quintile_changes += 1

    avg_quintile_changes = quintile_changes / N_TRIALS

    results[scenario_name] = {
        'total_mean': np.mean(trial_totals),
        'total_ci_low': np.percentile(trial_totals, 2.5),
        'total_ci_high': np.percentile(trial_totals, 97.5),
        'total_cv': np.std(trial_totals) / np.mean(trial_totals) * 100,
        'spearman_mean': np.mean(spearman_rhos),
        'spearman_min': np.min(spearman_rhos),
        'kendall_mean': np.mean(kendall_taus),
        'quintile_changes': avg_quintile_changes,
        'n_countries': n_countries,
    }

    r = results[scenario_name]
    print(f"  Total: ${r['total_mean']/1e9:.2f}B "
          f"(95% CI: ${r['total_ci_low']/1e9:.2f}–{r['total_ci_high']/1e9:.2f}B)")
    print(f"  CV: {r['total_cv']:.1f}%")
    print(f"  Spearman ρ: {r['spearman_mean']:.4f} (min: {r['spearman_min']:.4f})")
    print(f"  Kendall τ: {r['kendall_mean']:.4f}")
    print(f"  Avg quintile changes: {r['quintile_changes']:.1f}/{n_countries}")

# =========================================================================
# FAOSTAT CROSS-VALIDATION
# =========================================================================

print("\n" + "=" * 70)
print("FAOSTAT CROSS-VALIDATION")
print("=" * 70)

# MapSPAM national totals (sum pixels by country)
mapspam_totals = {}
for cid in unique_ids:
    if cid < 0:
        continue
    name = country_id_to_name.get(cid, None)
    if name is None:
        continue
    mask = countries_raster == cid
    total = np.sum(production_valid[mask])
    if total > 0:
        mapspam_totals[name] = total

# FAOSTAT 2020 maize production (tonnes) — key countries
# Source: FAOSTAT Production/Crops and livestock products, 2020
faostat_2020 = {
    'Nigeria': 11_547_270,
    'Tanzania': 6_737_528,
    'Ethiopia': 9_643_556,
    'Kenya': 4_070_000,
    'Malawi': 3_777_768,
    'Zambia': 3_387_469,
    'Mali': 3_056_703,
    'Mozambique': 2_100_000,
    'Ghana': 3_054_000,
    'Uganda': 3_500_000,
    'Cameroon': 2_200_000,
    'South Africa': 15_300_000,
    'Burkina Faso': 1_794_067,
    'Benin': 1_462_746,
    'Senegal': 560_553,
    'Togo': 936_571,
    'Niger': 21_043,
    'Zimbabwe': 907_193,
    'Rwanda': 520_000,
    'Burundi': 230_000,
}

# Compare
comparison = []
for country in faostat_2020:
    if country in mapspam_totals:
        fao_val = faostat_2020[country]
        spam_val = mapspam_totals[country]
        pct_diff = (spam_val - fao_val) / fao_val * 100
        comparison.append({
            'country': country,
            'faostat_tonnes': fao_val,
            'mapspam_tonnes': spam_val,
            'pct_difference': pct_diff,
        })

comp_df = pd.DataFrame(comparison)

if len(comp_df) > 0:
    from sklearn.metrics import r2_score, mean_absolute_percentage_error
    r2 = r2_score(comp_df['faostat_tonnes'], comp_df['mapspam_tonnes'])
    mape = mean_absolute_percentage_error(comp_df['faostat_tonnes'], comp_df['mapspam_tonnes']) * 100
    corr = comp_df['faostat_tonnes'].corr(comp_df['mapspam_tonnes'])

    print(f"\n  Countries compared: {len(comp_df)}")
    print(f"  Pearson correlation: {corr:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"\n  Per-country comparison:")
    for _, row in comp_df.sort_values('faostat_tonnes', ascending=False).iterrows():
        print(f"    {row['country']:20s}: FAOSTAT={row['faostat_tonnes']/1e6:.2f}M  "
              f"MapSPAM={row['mapspam_tonnes']/1e6:.2f}M  "
              f"Diff={row['pct_difference']:+.1f}%")

# =========================================================================
# GENERATE SUPPLEMENTARY FIGURE
# =========================================================================

print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: FAOSTAT vs MapSPAM
if len(comp_df) > 0:
    ax1.scatter(comp_df['faostat_tonnes'] / 1e6, comp_df['mapspam_tonnes'] / 1e6,
                s=60, c='#0072B2', alpha=0.7, edgecolors='white', linewidth=0.5)

    # 1:1 line
    max_val = max(comp_df['faostat_tonnes'].max(), comp_df['mapspam_tonnes'].max()) / 1e6
    ax1.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.5, linewidth=1)

    # Label only a few well-spread countries — avoid crowded cluster
    labels_to_show = {
        'South Africa': (-75, -5),
        'Nigeria': (-55, 8),
        'Ethiopia': (-55, -10),
        'Tanzania': (8, -12),
    }
    for _, row in comp_df.iterrows():
        if row['country'] in labels_to_show:
            offset = labels_to_show[row['country']]
            use_arrow = abs(offset[0]) > 30 or abs(offset[1]) > 12
            ax1.annotate(row['country'],
                        (row['faostat_tonnes'] / 1e6, row['mapspam_tonnes'] / 1e6),
                        xytext=offset, textcoords='offset points', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5,
                                       linewidth=0.8) if use_arrow else None)

    ax1.set_xlabel('FAOSTAT production (million tonnes)', fontsize=12)
    ax1.set_ylabel('MapSPAM production (million tonnes)', fontsize=12)
    ax1.set_title(f'a', fontsize=14, fontweight='bold', loc='left')
    ax1.text(0.05, 0.92, f'R² = {r2:.3f}\nMAPE = {mape:.1f}%',
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.set_xlim(0, max_val * 1.1)
    ax1.set_ylim(0, max_val * 1.1)
    ax1.grid(True, alpha=0.2)

# Panel B: Rank stability across perturbation scenarios
scenario_names_plot = list(results.keys())
spearman_means = [results[s]['spearman_mean'] for s in scenario_names_plot]
spearman_mins = [results[s]['spearman_min'] for s in scenario_names_plot]

x_pos = np.arange(len(scenario_names_plot))
bars = ax2.bar(x_pos, spearman_means, width=0.6, color='#009E73', alpha=0.85,
               edgecolor='white', linewidth=0.8)

# Add min markers
ax2.scatter(x_pos, spearman_mins, marker='_', s=200, color='black',
            linewidth=2, zorder=5, label='Minimum ρ')

# Reference lines
ax2.axhline(y=0.95, color='#D55E00', linestyle='--', linewidth=1, alpha=0.6,
            label='ρ = 0.95 threshold')
ax2.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

# Value labels
for i, v in enumerate(spearman_means):
    ax2.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels([s.replace(' ', '\n') for s in scenario_names_plot],
                     fontsize=9)
ax2.set_ylabel("Spearman's ρ (rank stability)", fontsize=12)
ax2.set_title('b', fontsize=14, fontweight='bold', loc='left')
ax2.set_ylim(0.85, 1.02)
ax2.legend(fontsize=10, loc='lower left')
ax2.grid(True, axis='y', alpha=0.2)

plt.tight_layout()

output_dir = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED'
fig.savefig(f'{output_dir}/figure_s_sensitivity.png', dpi=300, bbox_inches='tight',
            facecolor='white')
fig.savefig(f'{output_dir}/figure_s_sensitivity.pdf', bbox_inches='tight',
            facecolor='white')
print(f"\n  Figure saved to {output_dir}/figure_s_sensitivity.png")
plt.close()

# =========================================================================
# GENERATE RESULTS TABLE
# =========================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY TABLE")
print("=" * 70)

table_lines = []
table_lines.append("| Scenario | Total loss mean (95% CI) | CV | Spearman ρ (mean) | Spearman ρ (min) | Kendall τ | Quintile changes |")
table_lines.append("|---|---|---|---|---|---|---|")

# Add baseline row (calibrated MC total)
bl_total = sum(baseline_country_losses.values())
table_lines.append(f"| Baseline (no perturbation) | ${bl_total/1e9:.2f}B | — | 1.000 | 1.000 | 1.000 | 0/{n_countries} |")

for name, r in results.items():
    ci = f"${r['total_ci_low']/1e9:.2f}–{r['total_ci_high']/1e9:.2f}B"
    table_lines.append(
        f"| {name} | ${r['total_mean']/1e9:.2f}B ({ci}) | {r['total_cv']:.1f}% | "
        f"{r['spearman_mean']:.3f} | {r['spearman_min']:.3f} | "
        f"{r['kendall_mean']:.3f} | {r['quintile_changes']:.1f}/{r['n_countries']} |"
    )

print("\n".join(table_lines))

# =========================================================================
# SAVE EVIDENCE REPORT
# =========================================================================

report_path = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/05-ANALYSIS/VALIDATION/r2_3_sensitivity_evidence.md'

with open(report_path, 'w') as f:
    f.write("# R2.3 Input Data Sensitivity Analysis — Evidence Report\n\n")
    f.write(f"**Analysis date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
    f.write(f"**Perturbation trials**: {N_TRIALS}\n\n")

    f.write("## Perturbation Results\n\n")
    f.write("\n".join(table_lines))
    f.write("\n\n")

    f.write("## FAOSTAT Cross-Validation\n\n")
    if len(comp_df) > 0:
        f.write(f"- Countries compared: {len(comp_df)}\n")
        f.write(f"- Pearson correlation: {corr:.4f}\n")
        f.write(f"- R²: {r2:.4f}\n")
        f.write(f"- MAPE: {mape:.1f}%\n\n")

        f.write("| Country | FAOSTAT (Mt) | MapSPAM (Mt) | Difference (%) |\n")
        f.write("|---|---:|---:|---:|\n")
        for _, row in comp_df.sort_values('faostat_tonnes', ascending=False).iterrows():
            f.write(f"| {row['country']} | {row['faostat_tonnes']/1e6:.2f} | "
                    f"{row['mapspam_tonnes']/1e6:.2f} | {row['pct_difference']:+.1f}% |\n")

    f.write("\n## Figures\n\n")
    f.write("- `04-FIGURES/REVISED/figure_s_sensitivity.png` — Panel a: FAOSTAT vs MapSPAM; Panel b: Rank stability\n")

print(f"\n  Report saved to {report_path}")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
