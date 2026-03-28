# Spatial Assessment of *Striga* Economic Impact on African Maize Production

This repository contains the code for reproducing the analyses and figures presented in:

> **Spatial targeting and expansion prevention offer effective pathways to protect African maize from parasitic weed threats**
>
> *Manuscript under review*

## Overview

A computational framework for continental-scale assessment of parasitic weed (*Striga hermonthica* and *S. asiatica*) economic impacts on maize production across sub-Saharan Africa. The framework integrates agroecological similarity mapping, Monte Carlo simulation, future scenario modeling, and priority zone identification.

## Repository Structure

```
├── 01-current-impact/
│   └── mc_simulation_calibrated.py          # Monte Carlo economic loss estimation
├── 02-future-projections/
│   ├── climate_projections_2050_calibrated.py  # Scenario-based projections to 2050
│   └── perform_climate_gsa.py               # Global sensitivity analysis
├── 03-sensitivity-analysis/
│   └── input_sensitivity_analysis.py        # Input data perturbation analysis
├── 04-figures/
│   ├── create_figure2_panels_ab.py          # Figure 2a,b (suitability + infestation rate)
│   ├── create_figure2_panels_cd.py          # Figure 2c,d (economic loss + priority)
│   ├── create_figure2_combined.py           # Figure 2 combined
│   ├── create_figure3.py                    # Figure 3 (economic impact analysis)
│   ├── create_figure4.py                    # Figure 4 (future projections)
│   ├── create_figure_s1.py                  # Figure S1 (occurrence data + maize production)
│   └── create_figure_s2.py                  # Figure S2 (spatial resolution demonstration)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/izuku-franck1555/striga-economic-impact-africa.git
cd striga-economic-impact-africa
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies

## Data Sources

The analysis relies on the following publicly available datasets:

| Dataset | Source | Access |
|---------|--------|--------|
| *Striga* occurrence records (n=2,171) | CIMMYT/IITA | [Harvard Dataverse](https://doi.org/10.7910/DVN/YLDMHB) |
| Maize production (SPAM 2020 V2r0) | IFPRI | [Harvard Dataverse](https://doi.org/10.7910/DVN/SWPENT) |
| Maize prices (2020–2022 averages) | FAO | [FAOSTAT](https://www.fao.org/faostat/) |
| Bioclimatic variables (CHELSA v2.1) | Karger et al. | [chelsa-climate.org](https://chelsa-climate.org/) |
| Soil properties | iSDAsoil | [isda-africa.com](https://isda-africa.com/isdasoil) |
| Road network (GRIP) | Meijer et al. | [globio.info](https://www.globio.info/download-grip-dataset) |
| Production projections | van Ittersum et al. (2025) | [Paper](https://doi.org/10.1073/pnas.2423669122) |

## Execution Order

1. **`01-current-impact/mc_simulation_calibrated.py`** — Baseline economic loss estimation using 100,000 Monte Carlo iterations with Latin Hypercube Sampling
2. **`02-future-projections/climate_projections_2050_calibrated.py`** — Projections to 2050 under four agricultural development scenarios
3. **`02-future-projections/perform_climate_gsa.py`** — Variance decomposition using standardized regression coefficients
4. **`03-sensitivity-analysis/input_sensitivity_analysis.py`** — Perturbation analysis on production and price inputs
5. **`04-figures/`** — Figure generation (requires outputs from steps 1–4)

## Key Parameters

- Spatial resolution: 10 km
- Monte Carlo iterations: 100,000
- Calibration: De Groote et al. (2008) field-validated prevalence thresholds
- Projection period: 2020–2050
- Study scope: 49 sub-Saharan African countries

## Citation

If you use this code, please cite:

```
[Citation will be updated upon publication]
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
