# UK Power Flow to Embedded Solar Capacity

- Inferring behind-the-meter PV capacity
- Producing operational solar power forecasts.
- For approximately 600 UK primary substation locations.

---

## Problem Formulation - Embedded Capacity Estimation

Domestic level solar panels feed directly into the local network without being metered at the substation. The more solar capacity exists behind the meter, the lower the observed demand reading during daylight hours. This hence creates a fundamental observability problem without directly seeing how much of any apparent demand reduction is driven by solar generation versus genuine changes in consumption.

Without knowing how much solar is installed at each of the primary substations across the UKPN network, operators cannot accurately forecast future flows, plan for constraint management, or assess headroom available for flexibility services.

The aim of this project is to uncover hidden solar capacity figures purely from historical substation power readings and co-located weather data.

The main inference objective is hence to recover installed domestic capacity from historical residual signal - without direct metering.

---

## Data

| Source | Description | Resolution |
|---|---|---|
| UKPN | Half-hourly active power per primary substation | 30 min |
| ERA5 | Reanalysis weather variables (SSRD, T2M, U10, TP, etc.) | Hourly interpolated |
| PVLive (Sheffield Solar) | GSP-level aggregated PV generation actuals | 30 min |

**Coverage**: Approximately 600 UKPN primary substations across the South East and East of England.

**Feature engineering** including but not limited to: solar zenith/azimuth proxies (`solar_noon_proximity`, `sunrise_sunset_proximity`, `is_civil_twilight`), lagged irradiance (`ssrd_w_m2_lag_2h`, `ssrd_w_m2_lag_6h`), cyclical time encodings (`hour_sin`, `hour_cos`, `month_sin`, `dayofweek_cos`), and cloud/weather quality indices.

[UKPN Open Data Portal](https://ukpowernetworks.opendatasoft.com/pages/home/)

[Copernicus ERA5 Reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)

---

## Model Architecture

### Temporal Fusion Transformer (TFT)

The primary forecasting model utilised is a modified [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) (Lim et al., 2021), implemented via `pytorch-forecasting`.

**Architectural modifications:**

- LSTM encoder and decoder cells replaced with GRU units, reducing parameter count while retaining sequential inductive bias and improving both training stability and speed.

- An optional attention-head diversity regularisation term is added to the training loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SMAPE}} + \lambda \cdot \Omega_{\text{div}}$$

where:
- $\mathcal{L}_{\text{total}}$ — total training loss.
- $\mathcal{L}_{\text{SMAPE}}$ — symmetric mean absolute percentage error between predicted and actual power values.
- $\lambda$ — regularisation strength, controlling how much the diversity term influences training.
- $\Omega_{\text{div}} = \frac{1}{\binom{H}{2}} \sum_{i < j} \langle \mathbf{a}_i, \mathbf{a}_j \rangle$ — mean pairwise cosine similarity across all $H$ attention heads; penalising this encourages each head to specialise on different temporal patterns rather than learning redundant representations.

**Scheduler**: Cosine annealing with $\eta_{\min} = 0.01 \cdot \eta_0$, where $\eta_0$ is the initial learning rate and $\eta_{\min}$ is the minimum it decays to. Switchable to `ReduceLROnPlateau` via config.

**Target normalisation**: Per-group softplus normalisation via `GroupNormalizer`, applied independently per substation.

<img src="docs/images/feature_importance_top25.png" width="60%"/>

*Top 25 encoder variable importance weights. Interaction feature `temp_x_hour_cos` and seasonal encoding `month_sin` dominate, alongside 2-hour lagged irradiance and sunrise/sunset proximity.*

<img src="docs/images/global_model_attention_profile.png" width="60%"/>

*Aggregated temporal attention weights across the encoder window. Peaks correspond to identical timestamp lookbacks on previous days.*

<img src="docs/images/kemp_town_final_forecast.png" width="60%"/>

*Continuous one-week forecast for randomly selected primary substation - June 2025. The TFT captures daily demand cycles closely, with minor overestimation on high-irradiance afternoons.*

---

## Capacity Estimation: On-Manifold Method (cVAE)

As opposed to a brute force method, constructing synthetic weather scenarios from scratch which can in turn produce physically implausible conditions, the capacity estimation model first learns what real weather actually looks like at each site, then generates scenarios that are guaranteed to stay within that observed range.

**Stage 1 — Learning the weather distribution.** A cVAE is trained on years of historical weather data, learning a compact representation of typical conditions per location, month, and hour. Training minimises:

$$\mathcal{L} = \text{reconstruction error} + \text{latent regularisation}$$

where the reconstruction error penalises the network for generating weather vectors that differ from real observations, and the latent regularisation keeps the learned space smooth and continuous so that new samples remain physically plausible.

**Stage 2 — Generating extreme scenarios.** The model generates $k$ plausible weather vectors for a given site and time. Each is scored by how 'solar-'favourable' it is:

$$s = \sum_j w_j x_j$$

where $x_j$ is the value of weather feature $j$ (e.g. irradiance, cloud cover) and $w_j$ is a configured weight reflecting how strongly that feature drives solar generation (positive for irradiance, negative for cloud cover). The bottom 20% of scores become the *low-solar* pool; the top 20% become the *high-solar* pool.

**Stage 3 — Estimating capacity.** For each of $N$ draws, the TFT is run under both a low and high solar scenario. The difference in predicted power is the estimated embedded solar contribution:

$$\Delta_n = \max(0,\ \hat{P}^{+}_n - \hat{P}^{-}_n)$$

where $\hat{P}^{+}_n$ is the predicted power under high solar conditions and $\hat{P}^{-}_n$ under low solar conditions for draw $n$. 

The $\max(0, \cdot)$ ensures only positive deltas contribute — i.e. cases whereby more sun genuinely reduces net demand.

Embedded capacity is hence the mean delta across all draws:

$$\hat{C} = \frac{1}{N} \sum_{n=1}^{N} \Delta_n$$

with P95 across the $N$ draws reported as an uncertainty bound.

---

## Validation

Three complementary strategies are employed, consistent with the absence of ground truth:

1. **Year-on-year**: Capacity estimates should grow per year as solar deployment increases. Decreases flag potential estimation failures.
2. **Synthetic injection**: Known capacity solar generation is subtracted from observed net load and scaled at varied factors; the capacity estimation method should recover the scaled value as fundamental difference.
3. **GSP aggregation**: Substation level estimates are aggregated to GSP boundaries and compared against PVLive actuals.

<img src="docs/images/v2_p90_k0.1.png" width="60%"/>

*Validation Method 2 - per location scaling factor correspondance (k=0.1) using P90 estimated embedded capacities.*

---

## Repository Structure
```
uk-power-flow-to-solar-capacity/
├── configs/
│   ├── tft/
│   ├── nhits/
│   ├── boosting/
│   └── capacity_estimation/
├── src/
│   ├── tft/                        # TFT pipeline (train, infer, tune, data, model, utils)
│   ├── nhits/                      # N-HiTS pipeline
│   ├── boosting/                   # XGBoost pipeline
│   ├── capacity_estimation/
│   │   ├── brute_force/            # Grid search capacity estimator
│   │   ├── on_manifold/            # cVAE pipeline (sampler, run, data_utils, model_utils)
│   │   └── historical_analogue.py  # Estimator relying on historical periods
│   └── process/                    # Data preprocessing and merge pipeline
├── tests/
│   ├── tft/
│   ├── nhits/
│   ├── boosting/
│   └── process/
├── pyproject.toml
└── README.md
```

---

## Installation

Requires Python 3.12.

GPU recommended for TFT and N-HiTS training.

```bash
git clone https://github.com/openclimatefix/uk-power-flow-to-solar-capacity.git
cd uk-power-flow-to-solar-capacity
uv sync
```

Before running, replace all `PLACEHOLDER_*` tokens in the relevant config files under `configs/` with local paths.