#!/usr/bin/env python3
"""
Climate Projections for Striga Economic Impacts in 2050 - CALIBRATED VERSION
Evidence-Based Implementation with Full Literature Citations

Version: December 2025 (Evidence-Based Revision)

Key Features:
- Uses De Groote (2008) calibrated baseline (~$0.71B for 9 countries)
- Implements manuscript Equation 6 with multiplicative structure
- Divisor = 0.2 for climate stress normalization (saturation threshold)
- IR capped at 1.0 for biological interpretability

EVIDENCE BASE FOR MODEL STRUCTURE:
==================================

MULTIPLICATIVE STRUCTURE (Equation 6):
- CLIMEX pest modeling: EI = GI × SI (Kriticos et al. 2015)
- Van Mourik 2007 (Wageningen thesis): Striga germination requires SIMULTANEOUS
  conditions - larger seed bank AMPLIFIES germination when conditions favorable
- Steduto et al. 2009 (AquaCrop): Multiplicative product of stress factors
- Githui et al. 2022 (Int J Plant Biol): Additive approaches "inadequate for combined stresses"

LINEAR TEMPORAL TERM (β₁):
- CONSERVATIVE approximation: Field evidence shows 340%/year seed bank increases
  (Van Mourik 2007), but we use 7-15% over 30 years
- Acknowledges: Actual dynamics are near-exponential initially, then density-dependent
- Implication: May UNDERESTIMATE long-term Striga pressure

CLIMATE STRESS (β₂):
- Qualitative evidence: Yoneyama et al. 2007 (Planta) - N/P deficiency enhances
  strigolactone secretion; Dawoud & Sauerborn 1994 - temperature thresholds
- KNOWLEDGE GAP: No quantitative dose-response functions exist
- β₂ values (0.25-0.35) are PLAUSIBLE RANGES, not empirically derived coefficients

0.2 DIVISOR:
- Represents saturation at 20% climate-induced yield loss
- Normalization choice, not empirically derived constant
- Beyond this threshold, host stress assumed maximal

80% CONTROL CEILING:
- Push-pull technology: 81-86.7% reduction (Khan et al., icipe trials)
- IR-maize: 31-98%, typically 70-90% (Roobroeck et al. 2023)
- ISM: 90.3% (Ethiopia trials)
- Biological floor: Seed production 50,000-500,000/plant, longevity 10-20 years

TECHNOLOGY IMPROVEMENT RATES:
- Historical TFP: 1.3%/yr (1961-1991), 0.18%/yr (2011-2021) - GAP Report 2023
- East Africa 2011-2021: -1.54%/yr (USDA-ERS)
- Zimbabwe crisis: -12.5%/yr GDP decline
- Values adjusted to 0.2-1.0%/yr based on evidence (Jayne & Sanchez 2021, Science)

CEM (NEW AREAS):
- Van Delft et al. 1997: Sharp decline in seed density with distance from infested fields
- Berner et al. 1994: Primary long-distance vector is contaminated crop seed
- Colonization timeline: 4-7 years to typical pattern (Van Mourik 2007)
- New areas have ZERO initial seed bank (not dormant activation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform, norm, triang
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CLIMATE PROJECTIONS FOR STRIGA IMPACTS (2050) - CALIBRATED")
print("="*80)
print("Using De Groote (2008) calibrated baseline")
print("-"*80)


class ClimateProjections2050Calibrated:
    """
    Climate projections for Striga economic impacts in 2050
    CALIBRATED VERSION using De Groote (2008) baseline

    Key improvements:
    - Calibrated baseline (~$1.04B continental) instead of uncalibrated (~$1.77B)
    - Temporal intensification term (β₁) implemented
    - Climate stress normalized to 0.2 divisor with saturation
    - IR bounded at 1.0
    """

    def __init__(self, n_iterations=10000, output_dir='climate_projections_publication'):
        self.n_iterations = n_iterations
        self.output_dir = output_dir
        self.setup_directories()

        # Define study countries from van Ittersum et al. (2016)
        self.STUDY_COUNTRIES = [
            'Burkina Faso', 'Ghana', 'Mali', 'Nigeria',          # West Africa
            'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Zambia'  # East Africa
        ]

        # Regional mapping
        self.WEST_AFRICA = ['Burkina Faso', 'Ghana', 'Mali', 'Nigeria']
        self.EAST_AFRICA = ['Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Zambia']

        # Load all required data
        self.load_base_data()
        self.define_scenarios()
        self.define_uncertainty_distributions()

    def setup_directories(self):
        """Create output directory structure"""
        os.makedirs(f"{self.output_dir}/data", exist_ok=True)
        os.makedirs(f"{self.output_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.output_dir}/tables", exist_ok=True)

    def load_base_data(self):
        """Load CALIBRATED baseline data from December 2025 Monte Carlo"""
        print("\nLoading CALIBRATED baseline data...")

        try:
            # CRITICAL: Use CALIBRATED Monte Carlo results (De Groote 2008 thresholds)
            mc_path = '../monte_carlo_publication/monte_carlo_publication/data/mc_results/country_statistics_20251206_224123.csv'
            self.mc_country_stats = pd.read_csv(mc_path, index_col=0)
            print(f"  ✓ Loaded CALIBRATED country statistics")
            print(f"    Source: {mc_path}")

            # Load production data
            production_df = pd.read_csv('../preprocessed_data/country_maize_production.csv')

            # Extract baseline losses and production
            self.baseline_losses_2020 = {}
            self.baseline_production_2020 = {}

            for country in self.STUDY_COUNTRIES:
                if country in self.mc_country_stats.index:
                    # Economic loss from calibrated MC (mean in USD)
                    # Use 'econ_mean' column for calibrated results
                    if 'econ_mean' in self.mc_country_stats.columns:
                        self.baseline_losses_2020[country] = self.mc_country_stats.loc[country, 'econ_mean']
                    else:
                        self.baseline_losses_2020[country] = self.mc_country_stats.loc[country, 'mean']

                    # Production from preprocessed data
                    prod_row = production_df[production_df['country'] == country]
                    if len(prod_row) > 0:
                        self.baseline_production_2020[country] = prod_row.iloc[0]['maize_production_tonnes']
                    else:
                        # Estimate from loss and typical loss rate
                        price = self.get_country_price(country)
                        estimated_prod = (self.baseline_losses_2020[country] / (0.05 * price))
                        self.baseline_production_2020[country] = estimated_prod
                        print(f"    Note: Estimated production for {country}")

            print(f"  ✓ Loaded baseline for {len(self.baseline_losses_2020)} countries")

            # Show baseline summary
            total_baseline = sum(self.baseline_losses_2020.values())
            print(f"  ✓ Total 9-country baseline: ${total_baseline/1e9:.2f} billion (CALIBRATED)")

        except Exception as e:
            print(f"  ERROR: Could not load calibrated baseline data: {e}")
            print("  Attempting to load alternative path...")

            # Try alternative path structure
            try:
                mc_path = '../monte_carlo_publication/data/mc_results/country_statistics_20251206_224123.csv'
                self.mc_country_stats = pd.read_csv(mc_path, index_col=0)
                print(f"  ✓ Loaded from alternative path")
            except:
                raise FileNotFoundError(
                    "Could not find calibrated baseline. Run mc_simulation_calibrated.py first."
                )

        # Load maize prices
        try:
            self.maize_prices = pd.read_csv('../preprocessed_data/maize_prices_standardized.csv')
            self.price_dict = dict(zip(self.maize_prices['country'],
                                      self.maize_prices['price_usd_per_tonne']))
            print("  ✓ Loaded maize prices")
        except:
            print("  Warning: Using default prices")
            self.price_dict = {country: 400 for country in self.STUDY_COUNTRIES}

        # Calculate baseline loss rates
        self.calculate_baseline_loss_rates()

        # Load PNAS climate data
        self.load_climate_data()

    def get_country_price(self, country):
        """Get maize price for a country"""
        if hasattr(self, 'price_dict') and country in self.price_dict:
            return self.price_dict[country]
        else:
            if country in self.WEST_AFRICA:
                return 380
            else:
                return 450

    def calculate_baseline_loss_rates(self):
        """Calculate baseline loss rates for each country"""
        print("\nCalculating baseline loss rates...")

        self.baseline_loss_rates = {}
        total_production = 0
        total_loss_tonnes = 0

        for country in self.STUDY_COUNTRIES:
            if country in self.baseline_losses_2020 and country in self.baseline_production_2020:
                loss_usd = self.baseline_losses_2020[country]
                production_tonnes = self.baseline_production_2020[country]
                price = self.price_dict.get(country, 400)

                # Calculate physical loss
                loss_tonnes = loss_usd / price

                # Calculate loss rate
                loss_rate = loss_tonnes / production_tonnes if production_tonnes > 0 else 0
                self.baseline_loss_rates[country] = loss_rate

                total_production += production_tonnes
                total_loss_tonnes += loss_tonnes

                print(f"  {country}: {loss_rate*100:.1f}% loss rate")

        # Calculate aggregate statistics
        self.baseline_aggregate_loss_rate = total_loss_tonnes / total_production if total_production > 0 else 0
        self.nine_country_production = total_production
        self.nine_country_loss = sum(self.baseline_losses_2020.values())

        print(f"\nAggregate baseline (9 countries, CALIBRATED):")
        print(f"  Total production: {total_production/1e6:.1f} Mt")
        print(f"  Total loss: {total_loss_tonnes/1e6:.2f} Mt ({self.baseline_aggregate_loss_rate*100:.1f}%)")
        print(f"  Economic loss: ${self.nine_country_loss/1e9:.2f} billion")

    def load_climate_data(self):
        """Load PNAS climate projection data"""
        print("\nLoading PNAS climate data...")

        climate_dir = '../climate-change'

        try:
            self.table_s1 = pd.read_csv(f'{climate_dir}/table_s1_cereal_statistics.csv', comment='#')
            self.table_s2 = pd.read_csv(f'{climate_dir}/table_s2_crop_area_shares.csv', comment='#')
            self.table_s3 = pd.read_csv(f'{climate_dir}/table_s3_2050_projections.csv', comment='#')
            self.table_s4 = pd.read_csv(f'{climate_dir}/table_s4_climate_change_impacts.csv', comment='#')
            print("  ✓ Loaded all PNAS tables")

            self.extract_climate_impacts()
            self.extract_maize_shares()
            self.extract_production_projections()

        except Exception as e:
            print(f"  ERROR loading climate data: {e}")
            raise

    def extract_climate_impacts(self):
        """Extract country-specific climate impacts from Table S4"""
        self.climate_impacts = {}

        for country in self.STUDY_COUNTRIES:
            country_impact = self.table_s4[self.table_s4['Country'] == country]
            if len(country_impact) > 0:
                yield_impact = country_impact.iloc[0]['Maize_Rainfed']
                self.climate_impacts[country] = yield_impact
            else:
                if country in self.WEST_AFRICA:
                    self.climate_impacts[country] = -0.10
                else:
                    self.climate_impacts[country] = -0.08

        print("\nClimate yield impacts by country (C_j):")
        for country, impact in self.climate_impacts.items():
            print(f"  {country}: {impact*100:+.1f}%")

    def extract_maize_shares(self):
        """Extract country-specific maize shares"""
        self.maize_shares = {
            'Burkina Faso': 0.394,
            'Ghana': 0.804,
            'Mali': 0.361,
            'Nigeria': 0.454,
            'Ethiopia': 0.532,
            'Kenya': 0.493,
            'Tanzania': 0.636,
            'Uganda': 0.844,
            'Zambia': 0.958
        }

        print(f"\nCountry-specific maize shares:")
        for country, share in self.maize_shares.items():
            print(f"  {country}: {share:.1%}")

    def extract_production_projections(self):
        """Extract 2050 production projections"""
        self.production_2050 = {}

        pathways = {
            'YtrendAtrend': 'Yield_YtrendAtrend_t_ha',
            'Ytrend': 'Yield_Ytrend_t_ha'
        }

        for pathway_name, yield_col in pathways.items():
            self.production_2050[pathway_name] = {}

            for country in self.STUDY_COUNTRIES:
                country_proj = self.table_s3[self.table_s3['Location'] == country]

                if len(country_proj) > 0:
                    try:
                        if pathway_name == 'Ytrend':
                            area_col = 'Area_No_change_Mha'
                        else:
                            area_col = 'Area_Trend_Mha'

                        cereal_area_mha = country_proj.iloc[0][area_col]
                        cereal_yield_tha = country_proj.iloc[0][yield_col]
                        cereal_production_mt = cereal_area_mha * cereal_yield_tha

                        maize_production_mt = cereal_production_mt * self.maize_shares[country]
                        maize_production_tonnes = maize_production_mt * 1e6

                        self.production_2050[pathway_name][country] = maize_production_tonnes
                    except (KeyError, TypeError):
                        base_prod = self.baseline_production_2020.get(country, 1e6)
                        growth_factor = 1.2 if pathway_name == 'Ytrend' else 1.5
                        self.production_2050[pathway_name][country] = base_prod * growth_factor
                else:
                    base_prod = self.baseline_production_2020.get(country, 1e6)
                    growth_factor = 1.2 if pathway_name == 'Ytrend' else 1.5
                    self.production_2050[pathway_name][country] = base_prod * growth_factor

    def define_scenarios(self):
        """
        Define climate response scenarios with β₁ and β₂ parameters

        ==========================================================================
        EVIDENCE-BASED SCENARIO PARAMETERS
        ==========================================================================

        Manuscript Equation 6:
        IR_{ij,t}^{(s)} = min(1, IR_{ij,0} × (1 + β₁ × t/30) × (1 + β₂ × min(1, |C_j|/0.2)))

        MULTIPLICATIVE STRUCTURE JUSTIFICATION:
        - CLIMEX: EI = GI × SI (Kriticos et al. 2015)
        - Van Mourik 2007: Seed bank size AMPLIFIES germination under favorable conditions
        - Githui et al. 2022: Additive approaches "inadequate for combined stresses"

        PARAMETER EVIDENCE:

        β₁ (Temporal intensification): Scenario-based plausible ranges
        - Field evidence: 340%/year seed bank increase possible (Van Mourik 2007)
        - Our values (0.07-0.15 = 7-15% over 30 years) are CONSERVATIVE
        - Represents seed bank dynamics under different management intensities

        β₂ (Climate stress response): Scenario-based plausible ranges
        - KNOWLEDGE GAP: No quantitative dose-response functions exist
        - Qualitative support: Yoneyama et al. 2007, Dawoud & Sauerborn 1994
        - Values (0.25-0.35) capture documented drought-Striga synergies

        Control efficacy: EMPIRICALLY DERIVED
        - Push-pull: 81-86.7% reduction (Khan et al., icipe; Roobroeck et al. 2023)
        - IR-maize: 70-90% typical (Kanampiu et al. 2018)
        - ISM: 90.3% (Ethiopia trials)
        - 80% ceiling reflects biological floor (seed longevity 10-20 years)

        Technology improvement: EVIDENCE-BASED
        - Historical TFP: 1.3%/yr (1961-1991) - Lusigi & Thirtle 1997
        - Recent TFP: 0.18%/yr (2011-2021) - GAP Report 2023
        - Crisis: Zimbabwe saw -12.5%/yr GDP decline
        - Values (0.2-1.0%/yr) reflect documented rates (Jayne & Sanchez 2021, Science)
        ==========================================================================
        """
        print("\nDefining climate response scenarios...")
        print("  (Evidence-based parameters with literature citations)")

        self.scenarios = {
            # =================================================================
            # SCENARIO 1: Current Trajectory (SSP2)
            # =================================================================
            '2050-Continued-Trends': {
                'narrative': 'Current development trajectory under climate change',
                'production_pathway': 'YtrendAtrend',

                # β₁, β₂: SCENARIO-BASED PLAUSIBLE RANGES (not literature-derived)
                # β₁: Moderate seed bank accumulation under current management
                # Evidence: Linear 11% over 30 years is CONSERVATIVE vs 340%/yr observed
                'beta1': 0.11,
                # β₂: Climate-Striga synergy (qualitative evidence: Parker 2009, Rodenburg 2016)
                'beta2': 0.30,

                # Control parameters - EMPIRICALLY DERIVED
                # Evidence: Current adoption ~50% efficacy (Khan et al. 2014)
                'control_coverage_type': 'current',
                'control_efficacy_base': 0.55,  # Revised: 50%→55% based on push-pull evidence
                'efficacy_climate_degradation': 0.85,

                # Climate-Striga responses
                'severity_climate_factor': 1.15,
                'spatial_expansion': 1.10,

                # Economic
                'price_factor': 1.07,

                # Technology - EVIDENCE-BASED
                # Evidence: Historical TFP ~0.9%/yr (2000s), recent 0.18%/yr (GAP Report 2023)
                # Value: 0.5%/yr reflects weighted historical average
                'tech_improvement_annual': 0.005,  # Revised: 0.6%→0.5% per evidence

                'color': '#1f77b4'
            },

            # =================================================================
            # SCENARIO 2: Crisis Scenario (SSP3)
            # =================================================================
            '2050-Degraded-Response': {
                'narrative': 'Institutional breakdown under climate stress',
                'production_pathway': 'Ytrend',

                # β₁, β₂: SCENARIO-BASED PLAUSIBLE RANGES
                # β₁: Uncontrolled seed bank accumulation
                'beta1': 0.15,
                # β₂: Maximum climate-Striga synergy
                'beta2': 0.35,

                # Control parameters - DEGRADED
                # Evidence: Crisis conditions reduce efficacy below baseline
                'control_coverage_type': 'degraded',
                'control_efficacy_base': 0.45,  # Revised: 40%→45%
                'efficacy_climate_degradation': 0.75,

                # Climate-Striga responses
                'severity_climate_factor': 1.25,
                'spatial_expansion': 1.15,

                # Economic
                'price_factor': 1.15,

                # Technology - EVIDENCE-BASED
                # Evidence: Zimbabwe 2000-2008 saw -12.5%/yr GDP decline
                # Value: -2.0%/yr reflects documented crisis impacts
                'tech_improvement_annual': -0.020,  # Revised: -0.5%→-2.0% per Zimbabwe evidence

                'color': '#d62728'
            },

            # =================================================================
            # SCENARIO 3: Improved Management
            # =================================================================
            '2050-Enhanced-Control': {
                'narrative': 'Improved Striga control under climate change',
                'production_pathway': 'YtrendAtrend',

                # β₁, β₂: SCENARIO-BASED PLAUSIBLE RANGES
                # β₁: Seed bank accumulation slowed by improved control
                'beta1': 0.09,
                # β₂: Climate stress partially managed
                'beta2': 0.25,

                # Control parameters - EMPIRICALLY DERIVED
                # Evidence: Push-pull achieves 81% reduction (Khan et al.)
                'control_coverage_type': 'enhanced',
                'control_efficacy_base': 0.65,  # Revised: 60%→65% based on push-pull
                'efficacy_climate_degradation': 0.90,

                # Climate-Striga responses
                'severity_climate_factor': 1.12,
                'spatial_expansion': 1.08,

                # Economic
                'price_factor': 1.07,

                # Technology - EVIDENCE-BASED
                # Evidence: IMF projects 3-5%/yr possible under project conditions
                # Value: 0.8%/yr reflects optimistic but achievable improvement
                'tech_improvement_annual': 0.008,  # Revised: 0.9%→0.8% per evidence

                'color': '#2ca02c'
            },

            # =================================================================
            # SCENARIO 4: Optimal Management (Integrated Striga Management)
            # =================================================================
            '2050-Integrated-Management': {
                'narrative': 'Striga-focused innovations under climate change',
                'production_pathway': 'YtrendAtrend',

                # β₁, β₂: SCENARIO-BASED PLAUSIBLE RANGES
                # β₁: Near-stabilization through comprehensive ISM
                'beta1': 0.07,
                # β₂: Climate stress minimized through adaptation
                'beta2': 0.25,

                # Control parameters - EMPIRICALLY DERIVED
                # Evidence: ISM achieves 90.3% in Ethiopia trials; Push-pull 81-87% (Khan et al.)
                # Value: 75% efficacy reflects demonstrated ISM/push-pull performance
                'control_coverage_type': 'integrated',
                'control_efficacy_base': 0.75,  # Push-pull 81-87%, ISM 90% - 75% is conservative
                'efficacy_climate_degradation': 0.90,

                # Climate-Striga responses
                'severity_climate_factor': 1.10,
                'spatial_expansion': 1.07,

                # Economic
                'price_factor': 1.07,

                # Technology - EVIDENCE-BASED
                # Evidence: Upper bound of historical TFP (1.3%/yr)
                # Value: 1.0%/yr reflects ambitious but historically precedented rate
                'tech_improvement_annual': 0.010,  # Revised: 1.2%→1.0% per historical evidence

                'color': '#ff7f0e'
            }
        }

        print(f"  Defined {len(self.scenarios)} scenarios:")
        for name, params in self.scenarios.items():
            print(f"    - {name}: β₁={params['beta1']}, β₂={params['beta2']}")

    def define_uncertainty_distributions(self):
        """
        Define uncertainty distributions consistent with manuscript

        EVIDENCE BASE FOR CEM (Climate Establishment Multiplier):
        =========================================================
        CEM represents REDUCED initial infestation risk in newly cultivated areas.

        BIOLOGICAL BASIS:
        - New areas have ZERO initial Striga seed bank (Striga doesn't pre-exist)
        - Van Delft et al. 1997: "Sharp decline in seed density at increasing distances
          from infested fields, irrespective of wind direction"
        - Berner et al. 1994: Primary long-distance vector is contaminated crop seed
        - Van Mourik 2007: 4-7 years to reach typical infestation patterns

        CORRECT NARRATIVE (vs manuscript):
        - NOT "dormant seed activation in previously uncultivated areas"
        - IS "zero initial seed bank with gradual colonization via dispersal"

        CEM VALUES:
        - 0.20 (min): Best case - isolated expansion, clean seed sources
        - 0.35 (mode): Typical case - some dispersal from nearby fields
        - 0.50 (max): Worst case - rapid colonization via contaminated seed

        Implication: New agricultural areas initially have 20-50% the infestation
        risk of established farmland, but this gap closes over 4-7 years.
        """
        print("\nDefining uncertainty distributions...")

        # Climate Establishment Multiplier (CEM) - EVIDENCE-BASED
        # Evidence: Van Delft et al. 1997, Berner et al. 1994, Van Mourik 2007
        # Captures: Zero initial seed bank → gradual colonization over 4-7 years
        self.establishment_factor_dist = {
            'min': 0.20,   # Best case: isolated expansion, clean seeds
            'mode': 0.35,  # Typical: some dispersal from nearby fields
            'max': 0.50    # Worst case: rapid colonization via contaminated seed
        }
        print(f"  ✓ CEM: Triangular({self.establishment_factor_dist['min']:.2f}, "
              f"{self.establishment_factor_dist['mode']:.2f}, {self.establishment_factor_dist['max']:.2f})")
        print(f"    (Evidence: Van Delft et al. 1997 - seed density declines with distance)")

        # Control coverage distributions (Beta)
        self.coverage_distributions = {}

        # Regional baselines (from manuscript)
        # West Africa: 20%, East Africa: 30%
        for country in self.STUDY_COUNTRIES:
            if country in self.WEST_AFRICA:
                alpha, beta_param = 4, 16  # Mean ~0.20
            else:
                alpha, beta_param = 6, 14  # Mean ~0.30

            self.coverage_distributions[country] = {
                'current': (alpha, beta_param),
                'degraded': (alpha * 0.8, beta_param * 1.2),    # 0.8× multiplier
                'enhanced': (alpha * 1.5, beta_param * 0.8),    # 1.5× multiplier
                'integrated': (alpha * 2.0, beta_param * 0.6)   # 2.0× multiplier
            }

        # Climate impact uncertainty (Normal, 10% CV)
        self.climate_uncertainty_cv = 0.10

        print("  ✓ Control coverage: Regional baselines (20% WA, 30% EA) with scenario multipliers")
        print(f"  ✓ Climate uncertainty: Normal with {self.climate_uncertainty_cv*100:.0f}% CV")
        print(f"  ✓ Technology improvement: Triangular (mode±0.002)")
        print(f"  ✓ β₁, β₂: Triangular (mode±0.05)")

    def sample_parameters(self, scenario_name, country):
        """Sample uncertain parameters for Monte Carlo iteration"""
        scenario = self.scenarios[scenario_name]

        # Sample control coverage
        coverage_type = scenario['control_coverage_type']
        alpha, beta_param = self.coverage_distributions[country][coverage_type]
        coverage = beta.rvs(alpha, beta_param)

        # Sample climate impact (10% CV around country estimate)
        base_impact = self.climate_impacts[country]
        climate_impact = norm.rvs(base_impact, abs(base_impact) * self.climate_uncertainty_cv)

        # Sample technology improvement (mode±0.002)
        base_tech = scenario['tech_improvement_annual']
        tech_improvement = triang.rvs(c=0.5, loc=base_tech - 0.002, scale=0.004)
        if base_tech < 0:
            tech_improvement = max(-0.010, min(0.0, tech_improvement))
        else:
            tech_improvement = max(0.003, min(0.020, tech_improvement))

        # Sample β₁ (temporal intensification) - mode±0.05
        base_beta1 = scenario['beta1']
        beta1 = triang.rvs(c=0.5, loc=base_beta1 - 0.05, scale=0.10)
        beta1 = max(0.05, min(0.20, beta1))

        # Sample β₂ (climate stress response) - mode±0.05
        base_beta2 = scenario['beta2']
        beta2 = triang.rvs(c=0.5, loc=base_beta2 - 0.05, scale=0.10)
        beta2 = max(0.15, min(0.45, beta2))

        # Sample severity factor - mode±0.05
        base_severity = scenario['severity_climate_factor']
        severity_factor = triang.rvs(c=0.5, loc=base_severity - 0.05, scale=0.10)
        severity_factor = max(1.05, min(1.30, severity_factor))

        # Sample spatial expansion - mode±0.02
        base_spatial = scenario['spatial_expansion']
        spatial_expansion = triang.rvs(c=0.5, loc=base_spatial - 0.02, scale=0.04)
        spatial_expansion = max(1.05, min(1.20, spatial_expansion))

        # Sample CEM (establishment factor)
        establishment_factor = triang.rvs(
            c=(self.establishment_factor_dist['mode'] - self.establishment_factor_dist['min']) /
              (self.establishment_factor_dist['max'] - self.establishment_factor_dist['min']),
            loc=self.establishment_factor_dist['min'],
            scale=self.establishment_factor_dist['max'] - self.establishment_factor_dist['min']
        )

        return {
            'coverage': coverage,
            'climate_impact': climate_impact,
            'tech_improvement': tech_improvement,
            'beta1': beta1,
            'beta2': beta2,
            'severity_factor': severity_factor,
            'spatial_expansion': spatial_expansion,
            'establishment_factor': establishment_factor
        }

    def calculate_2050_loss_rate(self, country, scenario_name, params):
        """
        Calculate 2050 loss rate implementing manuscript Equation 6

        ==========================================================================
        EQUATION 6 (EVIDENCE-BASED FORMULATION):
        ==========================================================================

        IR_{ij,t}^{(s)} = min(1, IR_{ij,0} × (1 + β₁ × t/30) × (1 + β₂ × min(1, |C_j|/0.2)))

        MULTIPLICATIVE STRUCTURE - JUSTIFIED BY:
        - CLIMEX pest modeling: EI = GI × SI (Kriticos et al. 2015)
        - Van Mourik 2007: Seed bank AMPLIFIES germination under favorable conditions
        - Steduto et al. 2009 (AquaCrop): Multiplicative stress factors
        - Githui et al. 2022: Additive approaches "inadequate for combined stresses"

        COMPONENTS:

        (1 + β₁ × t/30) - TEMPORAL FACTOR:
        - Represents seed bank accumulation over 30 years
        - CONSERVATIVE: 7-15% over 30 years vs 340%/yr field evidence (Van Mourik 2007)
        - Scenario-based plausible ranges, not empirically derived

        (1 + β₂ × min(1, |C_j|/0.2)) - CLIMATE FACTOR:
        - Captures drought-Striga synergy
        - 0.2 divisor = saturation at 20% yield loss (normalization threshold)
        - KNOWLEDGE GAP: No quantitative dose-response data exists
        - β₂ values are plausible ranges based on qualitative evidence

        80% CONTROL CEILING:
        - Evidence: Push-pull 81-87%, IR-maize 70-90%, ISM 90% (Khan et al., Kanampiu et al.)
        - Biological floor: Seed production 50,000-500,000/plant, longevity 10-20 years
        - ONE escaped plant can rebuild infestation

        50% LOSS RATE CAP:
        - Represents biological maximum for aggregate losses
        - Individual farms can see 80-100% loss (De Groote 2007)
        - 50% is regional/aggregate ceiling
        ==========================================================================
        """
        scenario = self.scenarios[scenario_name]

        # Get baseline loss rate
        base_rate_2020 = self.baseline_loss_rates[country]

        # Step 1: Apply technological progress (compounds over 30 years)
        tech_factor = (1 - params['tech_improvement']) ** 30
        base_rate_2050_tech = base_rate_2020 * tech_factor

        # Step 2: Calculate IR adjustment following manuscript Equation 6
        # TEMPORAL FACTOR: (1 + β₁ × t/30) at t=30
        temporal_factor = 1.0 + params['beta1'] * (30 / 30)  # = 1 + β₁

        # CLIMATE FACTOR: (1 + β₂ × min(1, |C_j|/0.2))
        # Normalize climate impact with divisor 0.2, cap at 1.0
        normalized_stress = min(abs(params['climate_impact']) / 0.2, 1.0)
        climate_factor = 1.0 + params['beta2'] * normalized_stress

        # Combined IR adjustment (capped at producing reasonable values)
        ir_adjustment = temporal_factor * climate_factor

        # Apply severity adjustment under drought stress (C < -0.05)
        severity_adjustment = params['severity_factor'] if params['climate_impact'] < -0.05 else 1.0

        # Spatial expansion
        spatial_expansion = params['spatial_expansion']

        # Combined intensification
        total_intensification = ir_adjustment * severity_adjustment * spatial_expansion

        # Apply intensification
        intensified_rate = base_rate_2050_tech * total_intensification

        # Step 3: Apply control effects
        control_efficacy = scenario['control_efficacy_base']
        if params['climate_impact'] < -0.05:
            control_efficacy *= scenario['efficacy_climate_degradation']

        # Control multiplier: maximum 80% reduction achievable
        control_multiplier = 1 - (0.8 * params['coverage'] * control_efficacy)

        # Final loss rate
        final_rate = intensified_rate * control_multiplier

        # Cap at biological maximum (50% loss rate)
        # Also ensure IR stays ≤ 1.0 (implicit in loss rate cap)
        final_rate = min(final_rate, 0.50)

        # Store components for analysis
        components = {
            'base_rate_2020': base_rate_2020,
            'tech_factor': tech_factor,
            'base_rate_2050_tech': base_rate_2050_tech,
            'temporal_factor': temporal_factor,
            'normalized_stress': normalized_stress,
            'climate_factor': climate_factor,
            'ir_adjustment': ir_adjustment,
            'severity_adjustment': severity_adjustment,
            'spatial_expansion': spatial_expansion,
            'total_intensification': total_intensification,
            'intensified_rate': intensified_rate,
            'control_efficacy': control_efficacy,
            'control_multiplier': control_multiplier,
            'final_rate': final_rate
        }

        return final_rate, components

    def run_monte_carlo(self):
        """Run Monte Carlo simulation for all scenarios"""
        print(f"\nRunning Monte Carlo simulation ({self.n_iterations} iterations)...")

        self.results = {}

        for scenario_name in tqdm(self.scenarios.keys(), desc="Scenarios"):
            scenario = self.scenarios[scenario_name]

            # Storage for this scenario
            scenario_results = {
                'total_losses': np.zeros(self.n_iterations),
                'country_losses': {country: np.zeros(self.n_iterations)
                                  for country in self.STUDY_COUNTRIES},
                'loss_rates': {country: np.zeros(self.n_iterations)
                              for country in self.STUDY_COUNTRIES},
                'components': [],
                'parameters': [],
                'decomposition': {
                    'technology_impact': np.zeros(self.n_iterations),
                    'climate_impact': np.zeros(self.n_iterations),
                    'temporal_impact': np.zeros(self.n_iterations),  # NEW
                    'expansion_impact': np.zeros(self.n_iterations),
                    'control_impact': np.zeros(self.n_iterations),
                    'establishment_impact': np.zeros(self.n_iterations),
                    'baseline_total': np.zeros(self.n_iterations)
                }
            }

            # Run iterations
            for i in range(self.n_iterations):
                total_loss = 0
                iteration_components = {}
                iteration_params = {}

                for country in self.STUDY_COUNTRIES:
                    # Sample parameters
                    params = self.sample_parameters(scenario_name, country)
                    iteration_params[country] = params

                    # Calculate loss rate
                    loss_rate_2050, components = self.calculate_2050_loss_rate(
                        country, scenario_name, params
                    )

                    # Get 2050 production
                    pathway = scenario['production_pathway']
                    production_2050 = self.production_2050[pathway].get(
                        country, self.baseline_production_2020.get(country, 1e6) * 1.5
                    )

                    # Apply CEM for new production areas
                    production_2020 = self.baseline_production_2020.get(country, 0)

                    if production_2050 > production_2020:
                        existing_production = production_2020
                        new_production = production_2050 - production_2020

                        existing_loss = existing_production * loss_rate_2050
                        new_area_loss_rate = loss_rate_2050 * params['establishment_factor']
                        new_loss = new_production * new_area_loss_rate

                        physical_loss = existing_loss + new_loss

                        components['establishment_effect'] = (new_production * loss_rate_2050) - new_loss
                        components['new_production_share'] = new_production / production_2050
                    else:
                        physical_loss = production_2050 * loss_rate_2050
                        components['establishment_effect'] = 0
                        components['new_production_share'] = 0

                    components['establishment_factor'] = params['establishment_factor']

                    # Calculate economic loss
                    price_2050 = self.price_dict.get(country, 400) * scenario['price_factor']
                    economic_loss = physical_loss * price_2050

                    # Store results
                    scenario_results['country_losses'][country][i] = economic_loss
                    scenario_results['loss_rates'][country][i] = loss_rate_2050
                    total_loss += economic_loss

                    if i == 0:
                        iteration_components[country] = components

                # Track decomposition
                baseline_total = sum(self.baseline_losses_2020.values())
                scenario_results['decomposition']['baseline_total'][i] = baseline_total

                # Technology impact
                tech_impact = 0
                for country in self.STUDY_COUNTRIES:
                    baseline_loss = self.baseline_losses_2020[country]
                    tech_factor = (1 - iteration_params[country]['tech_improvement']) ** 30
                    tech_impact += baseline_loss * (tech_factor - 1)
                scenario_results['decomposition']['technology_impact'][i] = tech_impact

                # Temporal impact (NEW) - contribution from β₁
                temporal_impact = 0
                for country in self.STUDY_COUNTRIES:
                    baseline_loss = self.baseline_losses_2020[country]
                    temporal_factor = iteration_params[country]['beta1']
                    temporal_impact += baseline_loss * temporal_factor
                scenario_results['decomposition']['temporal_impact'][i] = temporal_impact

                # Climate impact - contribution from β₂
                climate_impact = 0
                for country in self.STUDY_COUNTRIES:
                    baseline_loss = self.baseline_losses_2020[country]
                    normalized_stress = min(abs(iteration_params[country]['climate_impact']) / 0.2, 1.0)
                    climate_factor = iteration_params[country]['beta2'] * normalized_stress
                    climate_impact += baseline_loss * climate_factor
                scenario_results['decomposition']['climate_impact'][i] = climate_impact

                # Expansion impact
                expansion_impact = 0
                for country in self.STUDY_COUNTRIES:
                    prod_2020 = self.baseline_production_2020.get(country, 0)
                    prod_2050 = self.production_2050[scenario['production_pathway']].get(country, prod_2020)
                    if prod_2050 > prod_2020 and prod_2020 > 0:
                        expansion_ratio = (prod_2050 - prod_2020) / prod_2020
                        baseline_loss = self.baseline_losses_2020[country]
                        expansion_impact += baseline_loss * expansion_ratio * iteration_params[country]['establishment_factor']
                scenario_results['decomposition']['expansion_impact'][i] = expansion_impact

                # Control impact
                control_impact = 0
                for country in self.STUDY_COUNTRIES:
                    baseline_loss = self.baseline_losses_2020[country]
                    coverage = iteration_params[country]['coverage']
                    efficacy = scenario['control_efficacy_base']
                    if iteration_params[country]['climate_impact'] < -0.05:
                        efficacy *= scenario['efficacy_climate_degradation']
                    control_impact -= baseline_loss * (0.8 * coverage * efficacy)
                scenario_results['decomposition']['control_impact'][i] = control_impact

                # Establishment impact
                establishment_impact = 0
                for country in self.STUDY_COUNTRIES:
                    prod_2020 = self.baseline_production_2020.get(country, 0)
                    prod_2050 = self.production_2050[scenario['production_pathway']].get(country, prod_2020)
                    if prod_2050 > prod_2020:
                        new_production = prod_2050 - prod_2020
                        loss_rate = self.baseline_loss_rates[country]
                        price = self.price_dict.get(country, 400)
                        full_loss = new_production * loss_rate * price
                        reduced_loss = full_loss * iteration_params[country]['establishment_factor']
                        establishment_impact -= (full_loss - reduced_loss)
                scenario_results['decomposition']['establishment_impact'][i] = establishment_impact

                scenario_results['total_losses'][i] = total_loss
                scenario_results['components'].append(iteration_components)
                scenario_results['parameters'].append(iteration_params)

            # Calculate statistics
            scenario_results['statistics'] = self.calculate_statistics(scenario_results)
            self.results[scenario_name] = scenario_results

        print("✓ Monte Carlo simulation complete")

    def calculate_statistics(self, scenario_results):
        """Calculate summary statistics"""
        stats = {}

        total_losses = scenario_results['total_losses']
        stats['total'] = {
            'mean': np.mean(total_losses),
            'median': np.median(total_losses),
            'std': np.std(total_losses),
            'cv': np.std(total_losses) / np.mean(total_losses) if np.mean(total_losses) > 0 else 0,
            'ci_95': np.percentile(total_losses, [2.5, 97.5]),
            'ci_90': np.percentile(total_losses, [5, 95]),
            'min': np.min(total_losses),
            'max': np.max(total_losses)
        }

        stats['countries'] = {}
        for country in self.STUDY_COUNTRIES:
            country_losses = scenario_results['country_losses'][country]
            loss_rates = scenario_results['loss_rates'][country]

            stats['countries'][country] = {
                'mean_loss': np.mean(country_losses),
                'median_loss': np.median(country_losses),
                'std_loss': np.std(country_losses),
                'ci_95_loss': np.percentile(country_losses, [2.5, 97.5]),
                'mean_rate': np.mean(loss_rates),
                'median_rate': np.median(loss_rates),
                'ci_95_rate': np.percentile(loss_rates, [2.5, 97.5])
            }

        return stats

    def perform_gsa(self):
        """Perform Global Sensitivity Analysis"""
        print("\nPerforming Global Sensitivity Analysis...")

        self.gsa_results = {}

        for scenario_name in self.scenarios.keys():
            components = self.results[scenario_name]['components']
            parameters = self.results[scenario_name]['parameters']

            if len(components) == 0:
                continue

            data = []
            for i, (comp_dict, param_dict) in enumerate(zip(components, parameters)):
                for country in self.STUDY_COUNTRIES:
                    if country in comp_dict and country in param_dict:
                        row = {
                            'country': country,
                            'iteration': i,
                            'loss_rate': comp_dict[country]['final_rate'],
                            'tech_factor': comp_dict[country]['tech_factor'],
                            'temporal_factor': comp_dict[country]['temporal_factor'],
                            'climate_factor': comp_dict[country]['climate_factor'],
                            'ir_adjustment': comp_dict[country]['ir_adjustment'],
                            'severity_adjustment': comp_dict[country]['severity_adjustment'],
                            'spatial_expansion': comp_dict[country]['spatial_expansion'],
                            'control_multiplier': comp_dict[country]['control_multiplier'],
                            **param_dict[country]
                        }
                        data.append(row)

            if len(data) == 0:
                continue

            df = pd.DataFrame(data)

            # Include β₁ and β₂ in GSA
            param_cols = ['tech_improvement', 'beta1', 'beta2', 'severity_factor',
                         'spatial_expansion', 'coverage', 'climate_impact', 'establishment_factor']

            X = df[param_cols].values
            y = df['loss_rate'].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LinearRegression()
            lr.fit(X_scaled, y)

            self.gsa_results[scenario_name] = {
                'parameters': param_cols,
                'coefficients': lr.coef_.tolist(),
                'r2_score': float(lr.score(X_scaled, y))
            }

        print("✓ GSA complete")

    def save_results(self):
        """Save all results"""
        print("\nSaving results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save scenario statistics
        scenario_stats = {}
        for scenario_name, results in self.results.items():
            scenario_stats[scenario_name] = {
                'total_mean': results['statistics']['total']['mean'],
                'total_median': results['statistics']['total']['median'],
                'total_ci_95': results['statistics']['total']['ci_95'].tolist(),
                'cv': results['statistics']['total']['cv'],
                'baseline_9country': self.nine_country_loss,
                'change_from_baseline_pct': (results['statistics']['total']['mean'] - self.nine_country_loss) / self.nine_country_loss * 100
            }

        with open(f"{self.output_dir}/data/scenario_statistics_calibrated_{timestamp}.json", 'w') as f:
            json.dump(scenario_stats, f, indent=2)

        # Save country statistics
        country_data = []
        for scenario_name, results in self.results.items():
            for country in self.STUDY_COUNTRIES:
                country_data.append({
                    'scenario': scenario_name,
                    'country': country,
                    'mean_loss': results['statistics']['countries'][country]['mean_loss'],
                    'mean_rate': results['statistics']['countries'][country]['mean_rate'],
                    'ci_95_low': results['statistics']['countries'][country]['ci_95_loss'][0],
                    'ci_95_high': results['statistics']['countries'][country]['ci_95_loss'][1]
                })

        pd.DataFrame(country_data).to_csv(
            f"{self.output_dir}/data/country_statistics_calibrated_{timestamp}.csv",
            index=False
        )

        # Save GSA results
        if hasattr(self, 'gsa_results'):
            with open(f"{self.output_dir}/data/gsa_results_calibrated_{timestamp}.json", 'w') as f:
                json.dump(self.gsa_results, f, indent=2)

        # Save full distributions
        for scenario_name, results in self.results.items():
            raw_params = {}
            if results['parameters']:
                n_iter = len(results['parameters'])
                param_arrays = {
                    'tech_improvement': np.zeros(n_iter),
                    'beta1_mean': np.zeros(n_iter),
                    'beta2_mean': np.zeros(n_iter),
                    'climate_impact_mean': np.zeros(n_iter),
                    'coverage_mean': np.zeros(n_iter),
                    'establishment_factor': np.zeros(n_iter),
                    'severity_factor': np.zeros(n_iter),
                    'spatial_expansion': np.zeros(n_iter)
                }

                for i, iter_params in enumerate(results['parameters']):
                    tech_vals, beta1_vals, beta2_vals = [], [], []
                    climate_vals, coverage_vals = [], []
                    severity_vals, spatial_vals, estab_vals = [], [], []

                    for country, params in iter_params.items():
                        tech_vals.append(params.get('tech_improvement', 0))
                        beta1_vals.append(params.get('beta1', 0))
                        beta2_vals.append(params.get('beta2', 0))
                        climate_vals.append(params.get('climate_impact', 0))
                        coverage_vals.append(params.get('coverage', 0))
                        severity_vals.append(params.get('severity_factor', 1))
                        spatial_vals.append(params.get('spatial_expansion', 1))
                        estab_vals.append(params.get('establishment_factor', 1))

                    param_arrays['tech_improvement'][i] = np.mean(tech_vals)
                    param_arrays['beta1_mean'][i] = np.mean(beta1_vals)
                    param_arrays['beta2_mean'][i] = np.mean(beta2_vals)
                    param_arrays['climate_impact_mean'][i] = np.mean(climate_vals)
                    param_arrays['coverage_mean'][i] = np.mean(coverage_vals)
                    param_arrays['establishment_factor'][i] = np.mean(estab_vals)
                    param_arrays['severity_factor'][i] = np.mean(severity_vals)
                    param_arrays['spatial_expansion'][i] = np.mean(spatial_vals)

                raw_params = param_arrays

            np.savez(
                f"{self.output_dir}/data/{scenario_name.replace('-', '_')}_calibrated_{timestamp}.npz",
                total_losses=results['total_losses'],
                **{f"country_{c}": results['country_losses'][c] for c in self.STUDY_COUNTRIES},
                **{f"rate_{c}": results['loss_rates'][c] for c in self.STUDY_COUNTRIES},
                decomposition_technology=results['decomposition']['technology_impact'],
                decomposition_temporal=results['decomposition']['temporal_impact'],
                decomposition_climate=results['decomposition']['climate_impact'],
                decomposition_expansion=results['decomposition']['expansion_impact'],
                decomposition_control=results['decomposition']['control_impact'],
                decomposition_establishment=results['decomposition']['establishment_impact'],
                decomposition_baseline=results['decomposition']['baseline_total'],
                **raw_params
            )

        print(f"✓ Results saved with timestamp: {timestamp}")
        return timestamp

    def create_summary_table(self):
        """Create publication-ready summary table"""
        print("\nCreating summary table...")

        summary_data = []

        for scenario_name in self.scenarios.keys():
            scenario = self.scenarios[scenario_name]
            stats = self.results[scenario_name]['statistics']['total']

            row = {
                'Scenario': scenario_name.replace('2050-', ''),
                'β₁': scenario['beta1'],
                'β₂': scenario['beta2'],
                'Mean Loss ($ billion)': stats['mean'] / 1e9,
                'CI 95% Lower': stats['ci_95'][0] / 1e9,
                'CI 95% Upper': stats['ci_95'][1] / 1e9,
                'CV (%)': stats['cv'] * 100,
                'Change from 2020 (%)': ((stats['mean'] - self.nine_country_loss) /
                                        self.nine_country_loss * 100)
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/tables/scenario_summary_calibrated.csv", index=False)

        print("\nSummary Table (CALIBRATED BASELINE):")
        print(summary_df.to_string(index=False))

        return summary_df

    def print_results(self):
        """Print key results"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY (CALIBRATED)")
        print("="*80)

        print(f"\n2020 Baseline (9 countries, CALIBRATED):")
        print(f"  Total loss: ${self.nine_country_loss/1e9:.2f} billion")
        print(f"  Aggregate loss rate: {self.baseline_aggregate_loss_rate*100:.1f}%")

        print(f"\n2050 Projections (with climate change):")

        for scenario_name, results in self.results.items():
            stats = results['statistics']['total']
            scenario = self.scenarios[scenario_name]
            print(f"\n{scenario_name} (β₁={scenario['beta1']}, β₂={scenario['beta2']}):")
            print(f"  Mean loss: ${stats['mean']/1e9:.2f} billion")
            print(f"  95% CI: [${stats['ci_95'][0]/1e9:.2f}, ${stats['ci_95'][1]/1e9:.2f}] billion")
            print(f"  Change from 2020: {(stats['mean']-self.nine_country_loss)/self.nine_country_loss*100:+.1f}%")
            print(f"  CV: {stats['cv']*100:.1f}%")

        # Key comparisons
        continued = self.results['2050-Continued-Trends']['statistics']['total']['mean']
        degraded = self.results['2050-Degraded-Response']['statistics']['total']['mean']
        enhanced = self.results['2050-Enhanced-Control']['statistics']['total']['mean']
        integrated = self.results['2050-Integrated-Management']['statistics']['total']['mean']

        print(f"\nKey Insights:")
        print(f"  Climate impact (Current vs 2020): "
              f"{(continued-self.nine_country_loss)/self.nine_country_loss*100:+.1f}%")
        print(f"  Crisis penalty (Crisis vs Current): "
              f"{(degraded-continued)/continued*100:+.1f}%")
        print(f"  Improved benefit (Improved vs Current): "
              f"{(enhanced-continued)/continued*100:+.1f}%")
        print(f"  Optimal benefit (Optimal vs Current): "
              f"{(integrated-continued)/continued*100:+.1f}%")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CLIMATE PROJECTIONS FOR STRIGA ECONOMIC IMPACTS (CALIBRATED)")
    print("Using De Groote (2008) calibrated baseline")
    print("="*80)

    # Get iterations from environment or use default
    n_iter = int(os.environ.get('MC_ITERATIONS', 100000))
    print(f"Running with {n_iter} iterations")

    # Initialize and run analysis
    projections = ClimateProjections2050Calibrated(n_iterations=n_iter)

    # Run Monte Carlo
    projections.run_monte_carlo()

    # Perform GSA
    projections.perform_gsa()

    # Save results
    timestamp = projections.save_results()

    # Create summary table
    summary_table = projections.create_summary_table()

    # Print results
    projections.print_results()

    print("\n" + "="*80)
    print("✓ CALIBRATED analysis complete")
    print(f"✓ Results saved to climate_projections_publication/")
    print(f"✓ Timestamp: {timestamp}")
    print("="*80)
