#!/usr/bin/env python3
"""
CALIBRATED Monte Carlo Simulation for Striga Economic Impact
Uses country-specific IR thresholds calibrated to De Groote 2008 field estimates

Key Changes from mc_simulation_updated.py:
1. Loads calibration_thresholds.csv with country-specific thresholds
2. Uses country-specific threshold instead of IR > 0 for infested area
3. Tracks calibration effectiveness (achieved % vs target %)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform
from tqdm import tqdm
import json
import rasterio
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os

print("CALIBRATED MONTE CARLO SIMULATION")
print("="*80)
print("Using De Groote 2008 field-validated prevalence calibration")
print("-"*80)

class CalibratedMonteCarlo:
    """
    Monte Carlo simulation with country-specific IR thresholds
    calibrated to match De Groote 2008 field-validated prevalence estimates
    """

    def __init__(self, n_iterations=1000, output_dir='monte_carlo_publication/data'):
        self.n_iterations = n_iterations
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/mc_results", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        self.load_base_data()
        self.load_calibration_thresholds()
        self.define_uncertainty_distributions()

    def load_base_data(self):
        """Load all required spatial and tabular data"""
        print("\nLoading base data...")

        # Spatial data
        with rasterio.open('../preprocessed_data/maize_africa_masked.tif') as src:
            self.maize_production = src.read(1)
            self.transform = src.transform
            self.shape = src.shape

        with rasterio.open('../preprocessed_data/infestation_rate_hybrid.tif') as src:
            self.base_ir = src.read(1)

        # Other data
        self.africa_boundaries = gpd.read_file('../Africa.shp')
        self.countries_raster = np.load('../analysis_outputs/africa_countries_raster.npy')

        # Price data
        price_candidates = [
            '../maize-price-data.csv',
            '../preprocessed_data/maize_prices_standardized.csv'
        ]
        for path in price_candidates:
            if os.path.exists(path):
                self.maize_prices = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError("No maize price file found.")

        self.control_params_df = pd.read_csv('../analysis_outputs/control_parameters_evidence_based.csv')

        # Species probability
        try:
            self.p_hermonthica = np.load('../analysis_outputs/p_hermonthica_occurrence_based.npy')
        except:
            print("  Warning: Species probability map not found. Using 50-50 split.")
            self.p_hermonthica = np.full_like(self.base_ir, 0.5)

        print(f"  ✓ Loaded {self.shape} pixel grid")
        print(f"  ✓ Loaded {len(self.africa_boundaries)} countries")

        self.calculate_pixel_areas()
        self.calculate_species_masks()

    def calculate_species_masks(self):
        """
        Derive species-range masks from occurrence-based probability surface.
        Thresholds set to retain ~95% of occurrences for each species:
        - Hermonthica mask: p_herm >= 0.512 (5th percentile of herm points)
        - Asiatica mask:    p_herm <= 0.516 (95th percentile of asiatica points)
        """
        # Defaults in case anything fails
        self.herm_mask = np.ones_like(self.p_hermonthica, dtype=bool)
        self.asia_mask = np.ones_like(self.p_hermonthica, dtype=bool)

        try:
            # Load occurrence points
            points = gpd.read_file('../preprocessed_data/all_striga_points.geojson').to_crs(self.africa_boundaries.crs)
            herm = points[points['species'] == 'S. hermonthica']
            asia = points[points['species'] == 'S. asiatica']

            # Sample p_herm at point locations
            def sample(arr, gdf):
                coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
                rows_cols = [~self.transform * (x, y) for x, y in coords]
                rows_cols = [(int(rc[1]), int(rc[0])) for rc in rows_cols]
                vals = []
                for r, c in rows_cols:
                    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                        vals.append(arr[r, c])
                return np.array(vals)

            herm_vals = sample(self.p_hermonthica, herm)
            asia_vals = sample(self.p_hermonthica, asia)

            # Thresholds at 5th percentile (herm) and 95th percentile (asia)
            t_herm = np.percentile(herm_vals, 5) if len(herm_vals) > 0 else 0.5
            t_asia = np.percentile(asia_vals, 95) if len(asia_vals) > 0 else 0.5

            self.herm_mask = self.p_hermonthica >= t_herm
            self.asia_mask = self.p_hermonthica <= t_asia

            print(f"  Species masks computed: t_herm={t_herm:.3f}, t_asia={t_asia:.3f}")
        except Exception as e:
            print(f"  Warning: Species mask computation failed ({e}). Using full masks.")

    def load_calibration_thresholds(self):
        """Load country-specific IR thresholds from calibration file"""
        print("\nLoading calibration thresholds...")

        # Load calibration data
        cal_df = pd.read_csv('calibration_thresholds.csv')

        # Create lookup dictionaries
        self.calibration_thresholds = {}
        self.calibration_targets = {}
        self.calibration_achieved = {}

        for _, row in cal_df.iterrows():
            country = row['country']
            self.calibration_thresholds[country] = row['computed_threshold']
            self.calibration_targets[country] = row['de_groote_target_pct']
            self.calibration_achieved[country] = row['achieved_pct']

        print(f"  ✓ Loaded thresholds for {len(self.calibration_thresholds)} countries")

        # Print key countries
        key_countries = ['Kenya', 'Nigeria', 'Tanzania', 'Malawi', 'Ethiopia']
        print("\n  Key calibration thresholds:")
        for c in key_countries:
            if c in self.calibration_thresholds:
                print(f"    {c}: threshold={self.calibration_thresholds[c]:.4f}, "
                      f"target={self.calibration_targets[c]*100:.0f}%, "
                      f"achieved={self.calibration_achieved[c]*100:.1f}%")

    def calculate_pixel_areas(self):
        """Pre-calculate pixel areas"""
        print("  Calculating pixel areas...")

        lat_min = self.transform[5] + self.transform[4] * self.shape[0]
        lat_max = self.transform[5]
        latitudes = np.linspace(lat_max, lat_min, self.shape[0])

        R = 6371  # Earth radius in km
        delta_lon = abs(self.transform[0])
        delta_lat = abs(self.transform[4])

        delta_lon_rad = np.radians(delta_lon)
        delta_lat_rad = np.radians(delta_lat)

        pixel_areas_km2 = np.zeros(self.shape[0])
        for i, lat in enumerate(latitudes):
            lat_rad = np.radians(lat)
            pixel_areas_km2[i] = R * R * delta_lon_rad * delta_lat_rad * np.cos(lat_rad)

        self.pixel_areas_ha = np.repeat(pixel_areas_km2[:, np.newaxis], self.shape[1], axis=1) * 100

    def define_uncertainty_distributions(self):
        """Define distributions for uncertainty analysis"""
        print("\nDefining uncertainty distributions...")

        self.param_distributions = {
            'implementation_efficiency': {
                'dist': 'uniform',
                'params': {'min': 0.7, 'max': 0.9},
            }
        }

        self.control_distributions = {
            'advanced_technology': {'alpha': 8, 'beta': 4},
            'traditional_methods': {'alpha': 4, 'beta': 7},
            'technology_mix': {'alpha': 3, 'beta': 3},
        }

        self.severity_distributions = {
            'hermonthica': {'alpha': 8, 'beta': 8},
            'asiatica': {'alpha': 4, 'beta': 16}
        }

    def calculate_scenario_loss(self, params, iteration_num):
        """Calculate loss with CALIBRATED thresholds"""

        ir_used = self.base_ir

        # Sample control parameters
        tech_pi = beta.rvs(
            self.control_distributions['technology_mix']['alpha'],
            self.control_distributions['technology_mix']['beta']
        )
        e_high = beta.rvs(
            self.control_distributions['advanced_technology']['alpha'],
            self.control_distributions['advanced_technology']['beta']
        )
        e_low = beta.rvs(
            self.control_distributions['traditional_methods']['alpha'],
            self.control_distributions['traditional_methods']['beta']
        )
        efficacy = tech_pi * e_high + (1 - tech_pi) * e_low

        rho = params.get('implementation_efficiency', 0.8)

        total_loss = 0
        country_losses = []
        country_losses_dict = {}
        country_losses_physical = {}

        components = {
            'iteration': iteration_num,
            'control_effectiveness': efficacy,
            'technology_mix': tech_pi,
            'advanced_tech_performance': e_high,
            'traditional_method_performance': e_low,
            'implementation_efficiency': rho
        }

        total_affected_ha = 0

        for idx, row in self.africa_boundaries.iterrows():
            country_name = row['NAME']
            country_mask = self.countries_raster == idx

            # Sample country coverage
            country_params = self.control_params_df[
                self.control_params_df['country'] == country_name
            ]
            if len(country_params) > 0:
                coverage = beta.rvs(
                    country_params.iloc[0]['alpha'],
                    country_params.iloc[0]['beta']
                )
            else:
                coverage = 0.25

            dm = 1 - rho * coverage * efficacy

            # Get price
            price_row = self.maize_prices[self.maize_prices['country'] == country_name]
            if len(price_row) > 0:
                price = price_row.iloc[0]['price_usd_per_tonne']
            else:
                price = self.maize_prices['price_usd_per_tonne'].mean()

            country_pixels = country_mask & (self.maize_production > 0)
            # Apply species masks uniformly (plausible Striga area)
            species_mask = (self.herm_mask | self.asia_mask)
            country_pixels = country_pixels & species_mask

            if np.any(country_pixels):
                # CALIBRATED: Get country-specific threshold
                threshold = self.calibration_thresholds.get(country_name, 0.25)

                # Sample severity only for pixels above threshold
                severity_sampled = self.sample_severity_pixels_calibrated(
                    country_pixels, ir_used, threshold
                )

                # Physical loss using relative-yield formulation
                damage_fraction = (
                    ir_used[country_pixels] *
                    severity_sampled[country_pixels] * dm
                )
                damage_fraction = np.clip(damage_fraction, 0, 0.999999)
                pixel_physical = (
                    self.maize_production[country_pixels] *
                    damage_fraction /
                    (1 - damage_fraction)
                )

                pixel_economic = pixel_physical * price
                country_total_physical = np.sum(pixel_physical)
                country_total = np.sum(pixel_economic)

                # CALIBRATED: Track affected area using threshold
                infested_pixels = country_pixels & (ir_used > threshold)
                country_affected_ha = np.sum(self.pixel_areas_ha[infested_pixels])
                total_affected_ha += country_affected_ha

                # Calculate actual % affected for validation
                total_country_maize_ha = np.sum(self.pixel_areas_ha[country_pixels])
                pct_affected = country_affected_ha / total_country_maize_ha if total_country_maize_ha > 0 else 0

                total_loss += country_total
                country_losses_dict[country_name] = country_total
                country_losses_physical[country_name] = country_total_physical

                country_losses.append({
                    'country': country_name,
                    'loss': country_total,
                    'physical_loss_tonnes': country_total_physical,
                    'farmer_adoption_rate': coverage,
                    'damage_multiplier': dm,
                    'affected_ha': country_affected_ha,
                    'pct_affected': pct_affected,
                    'threshold': threshold
                })

                # Store severity statistics
                infested_in_country = country_pixels & (ir_used > threshold)
                if np.any(infested_in_country):
                    components[f'damage_intensity_{country_name}'] = np.mean(
                        severity_sampled[infested_in_country]
                    )
                components[f'farmer_adoption_{country_name}'] = coverage
                components[f'pct_affected_{country_name}'] = pct_affected

        components['total_affected_ha'] = total_affected_ha

        return total_loss, country_losses, components, country_losses_dict, country_losses_physical

    def sample_severity_pixels_calibrated(self, country_pixels, ir, threshold):
        """
        Sample severity only where IR > threshold (not IR > 0)
        and within species-range masks derived from probability surface.
        """
        severity_full = np.zeros_like(country_pixels, dtype=float)

        # CALIBRATED: Only sample where country_pixels AND IR > threshold
        infested_mask = country_pixels & (ir > threshold)

        if np.any(infested_mask):
            p_herm = self.p_hermonthica[infested_mask]
            n_infested = np.sum(infested_mask)

            is_herm = np.random.random(n_infested) < p_herm

            severity_vector = np.zeros(n_infested)

            if np.any(is_herm):
                severity_vector[is_herm] = beta.rvs(
                    self.severity_distributions['hermonthica']['alpha'],
                    self.severity_distributions['hermonthica']['beta'],
                    size=np.sum(is_herm)
                )

            if np.any(~is_herm):
                severity_vector[~is_herm] = beta.rvs(
                    self.severity_distributions['asiatica']['alpha'],
                    self.severity_distributions['asiatica']['beta'],
                    size=np.sum(~is_herm)
                )

            severity_full[infested_mask] = severity_vector

        return severity_full

    def run_simulation(self):
        """Run calibrated Monte Carlo simulation"""
        print(f"\nRunning {self.n_iterations} CALIBRATED Monte Carlo iterations...")
        print("Using De Groote 2008 field-validated thresholds")

        total_losses = np.zeros(self.n_iterations)
        param_samples = []
        country_results = []
        all_components = []
        country_losses_matrix = []
        country_physical_matrix = []

        for i in tqdm(range(self.n_iterations)):
            params = {}
            for param_name, config in self.param_distributions.items():
                if config['dist'] == 'uniform':
                    value = uniform.rvs(
                        config['params']['min'],
                        config['params']['max'] - config['params']['min']
                    )
                params[param_name] = value

            param_samples.append(params)

            loss, country_losses, components, country_losses_dict, country_losses_physical = self.calculate_scenario_loss(params, i)
            total_losses[i] = loss
            country_losses_matrix.append(country_losses_dict)
            country_physical_matrix.append(country_losses_physical)

            all_components.append({**params, **components, 'total_loss': loss})

            if i % 100 == 0:
                country_results.append({
                    'iteration': i,
                    'countries': country_losses
                })

        components_df = pd.DataFrame(all_components)
        country_losses_df = pd.DataFrame(country_losses_matrix).fillna(0)
        country_physical_df = pd.DataFrame(country_physical_matrix).fillna(0)

        results = {
            'total_losses': total_losses,
            'param_samples': pd.DataFrame(param_samples),
            'components_df': components_df,
            'country_losses_df': country_losses_df,
            'country_physical_df': country_physical_df,
            'statistics': {
                'mean': np.mean(total_losses),
                'std': np.std(total_losses),
                'cv': np.std(total_losses) / np.mean(total_losses),
                'ci_95': np.percentile(total_losses, [2.5, 97.5]),
                'ci_90': np.percentile(total_losses, [5, 95]),
                'median': np.median(total_losses),
                'min': np.min(total_losses),
                'max': np.max(total_losses)
            }
        }

        if len(all_components) > 0:
            affected_areas = [c['total_affected_ha'] for c in all_components]
            results['statistics']['mean_affected_ha'] = np.mean(affected_areas)
            results['statistics']['std_affected_ha'] = np.std(affected_areas)

        self.save_comprehensive_results(results, country_results)

        return results

    def save_comprehensive_results(self, results, country_results):
        """Save all results - EXACT MATCH to uncalibrated version structure"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save primary results - SAME naming as original
        np.savez(f"{self.output_dir}/mc_results/mc_full_results_{timestamp}.npz",
                total_losses=results['total_losses'],
                country_losses_matrix=results['country_losses_df'].values,
                country_physical_matrix=results['country_physical_df'].values,
                country_names=results['country_losses_df'].columns.values)

        # Save country statistics - EXACT column structure as original
        country_stats = {}
        for country in results['country_losses_df'].columns:
            econ = results['country_losses_df'][country]
            phys = results['country_physical_df'][country] if country in results['country_physical_df'] else None

            # EXACT same columns as original mc_simulation_updated.py
            stats = {
                'econ_mean': np.mean(econ),
                'econ_median': np.median(econ),
                'econ_std': np.std(econ),
                'econ_ci_95_low': np.percentile(econ, 2.5),
                'econ_ci_95_high': np.percentile(econ, 97.5),
                'econ_ci_90_low': np.percentile(econ, 5),
                'econ_ci_90_high': np.percentile(econ, 95),
            }
            if phys is not None:
                stats.update({
                    'phys_mean_tonnes': np.mean(phys),
                    'phys_median_tonnes': np.median(phys),
                    'phys_std_tonnes': np.std(phys),
                    'phys_ci_95_low_tonnes': np.percentile(phys, 2.5),
                    'phys_ci_95_high_tonnes': np.percentile(phys, 97.5),
                    'phys_ci_90_low_tonnes': np.percentile(phys, 5),
                    'phys_ci_90_high_tonnes': np.percentile(phys, 95),
                })
            country_stats[country] = stats

        # SAME naming as original
        pd.DataFrame(country_stats).T.to_csv(
            f"{self.output_dir}/mc_results/country_statistics_{timestamp}.csv"
        )

        # Save parameter samples - SAME as original
        results['param_samples'].to_csv(
            f"{self.output_dir}/mc_results/parameter_samples_{timestamp}.csv",
            index=False
        )

        # Save components for GSA - SAME as original
        results['components_df'].to_csv(
            f"{self.output_dir}/mc_results/components_all_{timestamp}.csv",
            index=False
        )

        # Save confidence intervals - SAME structure as original
        ci_data = {
            'ci_95': results['statistics']['ci_95'].tolist(),
            'ci_90': results['statistics']['ci_90'].tolist(),
            'mean': results['statistics']['mean'],
            'median': results['statistics']['median'],
            'std': results['statistics']['std'],
            'cv': results['statistics']['cv']
        }

        with open(f"{self.output_dir}/mc_results/confidence_intervals_{timestamp}.json", 'w') as f:
            json.dump(ci_data, f, indent=2)

        # ADDITIONAL: Save calibration metadata separately
        calibration_meta = {
            'method': 'De Groote 2008 field-validated thresholds',
            'thresholds': {k: float(v) for k, v in self.calibration_thresholds.items()},
            'targets': {k: float(v) for k, v in self.calibration_targets.items()},
            'timestamp': timestamp
        }
        with open(f"{self.output_dir}/mc_results/calibration_metadata_{timestamp}.json", 'w') as f:
            json.dump(calibration_meta, f, indent=2)

        print(f"\n✓ Results saved with timestamp: {timestamp}")
        print(f"  (Output structure matches original mc_simulation_updated.py)")

        return timestamp

    def print_summary(self, results):
        """Print summary with calibration details"""
        print("\n" + "="*80)
        print("CALIBRATED RESULTS:")
        print("-"*60)
        print(f"Mean loss: ${results['statistics']['mean']/1e9:.2f} billion")
        print(f"Median loss: ${results['statistics']['median']/1e9:.2f} billion")
        print(f"95% CI: [${results['statistics']['ci_95'][0]/1e9:.2f}, "
              f"${results['statistics']['ci_95'][1]/1e9:.2f}] billion")
        print(f"CV: {results['statistics']['cv']*100:.1f}%")

        if 'mean_affected_ha' in results['statistics']:
            print(f"\nAffected area: {results['statistics']['mean_affected_ha']/1e6:.2f} ± "
                  f"{results['statistics']['std_affected_ha']/1e6:.2f} million ha")

        # Print key country comparisons
        print("\n" + "-"*60)
        print("KEY COUNTRY RESULTS (CALIBRATED):")
        print("-"*60)

        key_countries = ['Kenya', 'Nigeria', 'Tanzania', 'Malawi', 'Ethiopia', 'Benin']
        print(f"{'Country':<15} {'Econ Loss':>12} {'Phys Loss':>12} {'Threshold':>10} {'Target %':>10}")
        print("-"*60)

        for country in key_countries:
            if country in results['country_losses_df'].columns:
                econ = np.mean(results['country_losses_df'][country])
                phys = np.mean(results['country_physical_df'][country]) if country in results['country_physical_df'] else 0
                threshold = self.calibration_thresholds.get(country, 0.25)
                target = self.calibration_targets.get(country, 0) * 100

                print(f"{country:<15} ${econ/1e6:>10.1f}M {phys/1e3:>10.1f}kt {threshold:>10.4f} {target:>9.0f}%")


# Run simulation
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CALIBRATED MONTE CARLO SIMULATION")
    print("Using De Groote 2008 field-validated prevalence estimates")
    print("="*80)

    # Allow iteration override via environment variable (default 1000)
    n_iters = int(os.environ.get("MC_ITERATIONS", "1000"))
    mc = CalibratedMonteCarlo(n_iterations=n_iters)
    results = mc.run_simulation()

    mc.print_summary(results)

    print("\n✓ CALIBRATED analysis complete")
    print("✓ Results anchored to De Groote 2008 field validation")
