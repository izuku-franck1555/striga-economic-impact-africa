#!/usr/bin/env python3
"""
Global Sensitivity Analysis for Climate Projections
Adapted from Monte Carlo GSA methodology for multi-scenario analysis
Focuses on key decomposition drivers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
import os
from datetime import datetime

print("CLIMATE PROJECTIONS GLOBAL SENSITIVITY ANALYSIS")
print("="*80)
print("Multi-scenario uncertainty and driver analysis")
print("-"*80)

class ClimateGSA:
    """
    Global Sensitivity Analysis for Climate Projections
    Uses decomposition components from Monte Carlo runs
    """
    
    def __init__(self, timestamp='20250814_232707'):
        self.timestamp = timestamp
        self.data_dir = 'climate_projections_publication/data'
        self.output_dir = 'climate_projections_publication/outputs'
        
        # Create output directories
        os.makedirs(f'{self.output_dir}/gsa', exist_ok=True)
        os.makedirs(f'{self.output_dir}/figures', exist_ok=True)
        
        # Define scenarios
        self.scenarios = [
            '2050_Continued_Trends',
            '2050_Degraded_Response', 
            '2050_Enhanced_Control',
            '2050_Integrated_Management'
        ]
        
        # Scenario display names
        self.scenario_names = {
            '2050_Continued_Trends': 'Continued Trends',
            '2050_Degraded_Response': 'Degraded Response',
            '2050_Enhanced_Control': 'Enhanced Control', 
            '2050_Integrated_Management': 'Integrated Management'
        }
        
        # Load all scenario data
        self.load_scenario_data()
        
    def load_scenario_data(self):
        """Load NPZ files for all scenarios"""
        print("\nLoading scenario data...")
        
        self.scenario_data = {}
        
        for scenario in self.scenarios:
            file_path = f'{self.data_dir}/{scenario}_{self.timestamp}.npz'
            
            if os.path.exists(file_path):
                data = np.load(file_path)
                self.scenario_data[scenario] = data
                print(f"  ✓ Loaded {scenario}")
                
                # Check available keys
                if scenario == self.scenarios[0]:  # Print for first scenario only
                    print(f"    Available components: {[k for k in data.keys() if 'decomposition' in k]}")
            else:
                print(f"  ✗ File not found: {file_path}")
                
        if len(self.scenario_data) == 0:
            raise FileNotFoundError(f"No scenario data found for timestamp {self.timestamp}")
            
    def perform_decomposition_gsa(self):
        """
        Perform GSA using decomposition components
        These represent the key drivers of change from 2020 to 2050
        """
        print("\nPerforming decomposition-based GSA...")
        
        self.gsa_results = {}
        
        # Define decomposition components and their display names
        # Note: Expansion and Establishment are combined as they're perfectly correlated
        components = {
            'decomposition_technology': 'Technology Progress',
            'decomposition_climate': 'Climate Impact',
            'decomposition_expansion': 'Area Expansion Effects',  # Combined with establishment
            'decomposition_control': 'Control Measures'
        }
        
        for scenario_name, data in self.scenario_data.items():
            print(f"\n  Analyzing {self.scenario_names[scenario_name]}...")
            
            # Get total losses and decomposition components
            total_losses = data['total_losses']
            n_iterations = len(total_losses)
            
            # Build matrix of decomposition components
            # Combine expansion and establishment since they're perfectly correlated
            X = np.zeros((n_iterations, len(components)))
            component_names = []
            
            for i, (comp_key, comp_name) in enumerate(components.items()):
                if comp_key == 'decomposition_expansion':
                    # Combine expansion and establishment effects
                    if 'decomposition_expansion' in data and 'decomposition_establishment' in data:
                        X[:, i] = data['decomposition_expansion'] + data['decomposition_establishment']
                        component_names.append(comp_name)
                    else:
                        print(f"    Warning: expansion/establishment not found")
                        X[:, i] = 0
                        component_names.append(comp_name)
                elif comp_key in data:
                    X[:, i] = data[comp_key]
                    component_names.append(comp_name)
                else:
                    print(f"    Warning: {comp_key} not found")
                    X[:, i] = 0
                    component_names.append(comp_name)
            
            # Calculate net change from baseline
            baseline = data['decomposition_baseline'][0] if 'decomposition_baseline' in data else 1.175e9
            net_change = total_losses - baseline
            
            # Method 1: Variance decomposition (proportion of variance explained)
            # This shows how much each component contributes to total variance
            variance_contrib = {}
            total_variance = np.var(net_change)
            
            for i, name in enumerate(component_names):
                component_variance = np.var(X[:, i])
                # Calculate correlation with net change
                correlation = np.corrcoef(X[:, i], net_change)[0, 1]
                # Variance contribution = correlation^2 * component_variance / total_variance
                variance_contrib[name] = abs(correlation) * np.std(X[:, i]) / np.std(net_change)
            
            # Normalize to percentages
            total_contrib = sum(variance_contrib.values())
            for name in variance_contrib:
                variance_contrib[name] = (variance_contrib[name] / total_contrib) * 100
            
            # Method 2: Standardized regression coefficients
            # Standardize predictors and response
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(net_change.reshape(-1, 1)).ravel()
            
            # Linear regression
            lr = LinearRegression()
            lr.fit(X_scaled, y_scaled)
            
            # Standardized coefficients (absolute values for importance)
            std_coefs = np.abs(lr.coef_)
            std_coefs_norm = (std_coefs / std_coefs.sum()) * 100
            
            # Method 3: First-order Sobol indices approximation
            # Using correlation ratios
            sobol_indices = {}
            for i, name in enumerate(component_names):
                # Sort by this component
                sorted_idx = np.argsort(X[:, i])
                # Bin into groups
                n_bins = 20
                bin_size = n_iterations // n_bins
                
                conditional_means = []
                for b in range(n_bins):
                    start_idx = b * bin_size
                    end_idx = (b + 1) * bin_size if b < n_bins - 1 else n_iterations
                    bin_indices = sorted_idx[start_idx:end_idx]
                    conditional_means.append(np.mean(net_change[bin_indices]))
                
                # Variance of conditional means
                var_conditional = np.var(conditional_means)
                sobol_indices[name] = (var_conditional / total_variance) * 100
            
            # Normalize Sobol indices
            total_sobol = sum(sobol_indices.values())
            if total_sobol > 0:
                for name in sobol_indices:
                    sobol_indices[name] = (sobol_indices[name] / total_sobol) * 100
            
            # Store results
            self.gsa_results[scenario_name] = {
                'component_names': component_names,
                'variance_contribution': variance_contrib,
                'standardized_coefficients': dict(zip(component_names, std_coefs_norm)),
                'sobol_indices': sobol_indices,
                'r2_score': lr.score(X_scaled, y_scaled),
                'total_variance': total_variance,
                'mean_change': np.mean(net_change),
                'baseline': baseline
            }
            
            # Print summary
            print(f"    R² score: {lr.score(X_scaled, y_scaled):.3f}")
            print(f"    Top drivers (by standardized coefficients):")
            sorted_drivers = sorted(zip(component_names, std_coefs_norm), 
                                  key=lambda x: x[1], reverse=True)
            for driver, importance in sorted_drivers[:3]:
                print(f"      {driver}: {importance:.1f}%")
                
    def perform_scenario_comparison_gsa(self):
        """
        GSA comparing across scenarios to identify which factors
        drive differences between scenarios
        """
        print("\nPerforming cross-scenario GSA...")
        
        # Collect all scenario outcomes
        all_losses = []
        scenario_indicators = []
        
        for i, (scenario_name, data) in enumerate(self.scenario_data.items()):
            losses = data['total_losses']
            all_losses.extend(losses)
            # Create indicator variables for scenarios
            scenario_indicators.extend([i] * len(losses))
        
        all_losses = np.array(all_losses)
        scenario_indicators = np.array(scenario_indicators)
        
        # Calculate variance explained by scenario choice
        total_var = np.var(all_losses)
        scenario_means = [np.mean(all_losses[scenario_indicators == i]) 
                         for i in range(len(self.scenarios))]
        between_scenario_var = np.var(scenario_means) * len(all_losses) / len(self.scenarios)
        
        scenario_var_explained = (between_scenario_var / total_var) * 100
        
        print(f"  Variance explained by scenario choice: {scenario_var_explained:.1f}%")
        print(f"  Variance within scenarios: {100 - scenario_var_explained:.1f}%")
        
        self.cross_scenario_results = {
            'total_variance': total_var,
            'scenario_variance_explained': scenario_var_explained,
            'within_scenario_variance': 100 - scenario_var_explained,
            'scenario_means': dict(zip(self.scenario_names.values(), scenario_means))
        }
        
    def aggregate_gsa_results(self):
        """
        Aggregate GSA results across scenarios to identify
        universal vs scenario-specific drivers
        """
        print("\nAggregating results across scenarios...")
        
        # Collect importance scores for each component across scenarios
        component_importance = {}
        
        for scenario_name, results in self.gsa_results.items():
            for comp_name, importance in results['standardized_coefficients'].items():
                if comp_name not in component_importance:
                    component_importance[comp_name] = []
                component_importance[comp_name].append(importance)
        
        # Calculate mean and std of importance across scenarios
        self.aggregate_importance = {}
        for comp_name, importances in component_importance.items():
            self.aggregate_importance[comp_name] = {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'min': np.min(importances),
                'max': np.max(importances),
                'range': np.max(importances) - np.min(importances)
            }
        
        # Identify consistent vs variable drivers
        print("\n  Component importance consistency:")
        sorted_comps = sorted(self.aggregate_importance.items(), 
                            key=lambda x: x[1]['mean'], reverse=True)
        
        for comp_name, stats in sorted_comps:
            consistency = "Consistent" if stats['std'] < 5 else "Variable"
            print(f"    {comp_name:25s}: {stats['mean']:5.1f}% ± {stats['std']:4.1f}% ({consistency})")
            
    def save_results(self):
        """Save GSA results to files"""
        print("\nSaving GSA results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_for_save = {}
        for scenario_name, results in self.gsa_results.items():
            results_for_save[scenario_name] = {
                'component_names': results['component_names'],
                'variance_contribution': results['variance_contribution'],
                'standardized_coefficients': results['standardized_coefficients'],
                'r2_score': float(results['r2_score']),
                'mean_change': float(results['mean_change']),
                'baseline': float(results['baseline'])
            }
        
        # Add cross-scenario results
        if hasattr(self, 'cross_scenario_results'):
            results_for_save['cross_scenario'] = self.cross_scenario_results
        
        # Add aggregate results
        if hasattr(self, 'aggregate_importance'):
            results_for_save['aggregate_importance'] = self.aggregate_importance
        
        # Save to JSON
        output_file = f'{self.output_dir}/gsa/climate_gsa_results_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results_for_save, f, indent=2, default=float)
        
        print(f"  ✓ Saved results to {output_file}")
        
        # Also save a summary CSV for easy viewing
        summary_data = []
        for scenario in self.scenarios:
            if scenario in self.gsa_results:
                results = self.gsa_results[scenario]
                for comp_name in results['component_names']:
                    summary_data.append({
                        'Scenario': self.scenario_names[scenario],
                        'Component': comp_name,
                        'Importance (%)': results['standardized_coefficients'][comp_name],
                        'Variance Contribution (%)': results['variance_contribution'].get(comp_name, 0)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f'{self.output_dir}/gsa/climate_gsa_summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"  ✓ Saved summary to {summary_file}")
        
        return timestamp

def main():
    """Main execution"""
    print("\nInitializing Climate GSA Analysis...")
    
    # Create GSA analyzer
    gsa = ClimateGSA(timestamp='20250814_232707')
    
    # Perform decomposition-based GSA
    gsa.perform_decomposition_gsa()
    
    # Perform cross-scenario comparison
    gsa.perform_scenario_comparison_gsa()
    
    # Aggregate results
    gsa.aggregate_gsa_results()
    
    # Save results
    timestamp = gsa.save_results()
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Print key insights
    print("\n1. Most Important Drivers (averaged across scenarios):")
    sorted_drivers = sorted(gsa.aggregate_importance.items(), 
                          key=lambda x: x[1]['mean'], reverse=True)
    for driver, stats in sorted_drivers:
        print(f"   {driver:25s}: {stats['mean']:5.1f}%")
    
    print(f"\n2. Scenario Choice Impact:")
    print(f"   Explains {gsa.cross_scenario_results['scenario_variance_explained']:.1f}% of total variance")
    print(f"   Within-scenario uncertainty: {gsa.cross_scenario_results['within_scenario_variance']:.1f}%")
    
    print("\n3. Most Variable Driver Across Scenarios:")
    most_variable = max(gsa.aggregate_importance.items(), 
                       key=lambda x: x[1]['range'])
    print(f"   {most_variable[0]}: ranges from {most_variable[1]['min']:.1f}% to {most_variable[1]['max']:.1f}%")
    
    print("\n" + "="*80)
    print("✓ GSA analysis complete")
    print(f"✓ Results saved with timestamp: {timestamp}")
    print("✓ Ready for visualization")
    print("="*80)
    
    return gsa

if __name__ == "__main__":
    gsa = main()