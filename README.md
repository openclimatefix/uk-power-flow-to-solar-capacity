# UK Solar Power & Embedded Solar Capacity Estimation

A repository for forecasting primary transformer power using XGBoost and subsequently estimating embedded capacity.

## Major Functionality

* [cite_start]**Time-Series Forecasting**: Trains an `XGBoost` model to predict half-hourly power demand at primary transformer sites based on historical power flow and ERA5 weather data (total cloud cover, solar radiation, temperature)[cite: 47, 287].
* **Embedded Capacity Estimation**: Utilises the trained model as an effective 'digital twin' to hence estimate the unmetered solar generation capacity within the network. [cite_start]Two distinct methodologies are considered for this further stage: "Dummy Weather" and "Historical Analogy"[cite: 48].

## Visualisations

The following graphs from the project's preliminary report illustrate key data and findings.

**(Figure 2) Map of Selected Transformers**
[cite_start]A map displaying the initial 20 transformer sites selected for the analysis, chosen to ensure geographical diversity as a basis across the UKPN network[cite: 151].
![Map of Selected Transformers](./docs/images/map_figure_2.png)

**(Figure 3) Power Reduction vs. Solar Irradiance**
[cite_start]A time-series plot for the Kingsbury site displaying the inverse relationship between net power demand and solar irradiance (`SSRD`) - demonstrating the impact of PV generation[cite: 204, 205].
![Power vs Solar Irradiance](./docs/images/power_vs_solar_figure_3.png)

**(Figure 6) Model Forecast Performance**
[cite_start]Comparing the actual power flow against the XGBoost model's predicted power on the test set for the Marshalswick site - demonstrating high forecast accuracy[cite: 287, 299].
![Model Forecast Performance](./docs/images/performance_figure_6.png)

**(Figure A1) Top 25 Feature Importances**
[cite_start]F-scores of the top 25 most influential features within the trained XGBoost model[cite: 418, 478].
![Feature Importances](./docs/images/feature_importance_figure_A1.png)

## Technical Architecture

* **Source / Main Pipeline**: The project is structured as a series of distinct, scriptable stages: data loading/unzipping, preprocessing, feature engineering, model training, and scenario analysis.
* **Configuration**: All parameters, paths, and hyperparameters are centrally located in the `config.yaml` file.
* **Core Libraries**: Not limited to: `pandas`, `xgboost`, `scikit-learn`, and `xarray` / `pyproj`.
* [cite_start]**External Data**: Integrates UKPN power data, ERA5 meteorological data, and Passiv installed PV capacity data[cite: 119, 138, 143].

## Data Cleaning & Processing Stages

* [cite_start]**Source Data**: The primary dataset is the "UKPN Primary Transformer - Historic Half Hourly" power flow data, covering the period from January 1, 2021, to December 31, 2024[cite: 119, 122].
* **Data Quality Issues Addressed**:
    * [cite_start]**Negative Power Values**: Absolute value of power readings is utilised to correct for measurement conventions that hence result in negative values[cite: 128, 134].
    * [cite_start]**Duplicate Entries**: Duplicate readings for single timestamp are resolved via retaining the first instance[cite: 125, 133].
    * [cite_start]**Temporal Harmonisation**: The native half-hourly power data is resampled to a uniform hourly frequency by taking the mean to align with the hourly ERA5 weather data[cite: 130, 132].

## Installation

Clone repository and install necessary dependencies in editable mode with the `dev` extras.

```bash
git clone <repository-url>
cd uk-power-flow-to-solar-capacity
pip install -e "[dev]"