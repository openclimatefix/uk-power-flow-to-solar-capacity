# UK Power Flow to Embedded Solar Capacity

Inferring behind-the-meter PV capacity and producing operational solar power forecasts for approximately 600 UK primary substation locations.

## Problem Formulation

Domestic level solar panels feed electricity directly into the local network without being metered at the substation. The more solar capacity exists behind the meter, the lower the observed demand reading during daylight hours. This creates a fundamental observability problem without directly seeing how much of any apparent demand reduction is driven by solar generation versus genuine changes in consumption.

Without knowing how much solar is installed at each of the primary substations across the UKPN network, operators cannot accurately forecast future flows, plan for constraint management, or assess headroom available for flexibility services.

The aim of this project is to uncover hidden solar capacity figures purely from historical substation power readings and co-located weather data.

The main inference objective is hence:

**Capacity estimation**: Recover installed domestic capacity from historical residual signal, without direct metering.

---

## Data

| Source | Description | Resolution |
|---|---|---|
| UKPN | Half-hourly active power per primary substation | 30 min |
| ERA5 | Reanalysis weather variables (SSRD, T2M, U10, TP, etc.) | Hourly interpolated |
| PVLive (Sheffield Solar) | GSP-level aggregated PV generation actuals | 30 min |

**Coverage**: Approximately 600 UKPN primary substations across the South East and East of England.

**Feature engineering** includes: solar zenith/azimuth proxies (`solar_noon_proximity`, `sunrise_sunset_proximity`, `is_civil_twilight`), lagged irradiance (`ssrd_w_m2_lag_2h`, `ssrd_w_m2_lag_6h`), cyclical time encodings (`hour_sin`, `hour_cos`, `month_sin`, `dayofweek_cos`), and cloud/weather quality indices.

## Model Architectures

### Temporal Fusion Transformer (TFT)

The primary forecasting model utilised is a modified [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) (Lim et al., 2021), implemented via `pytorch-forecasting`.

**Architectural modifications:**

- LSTM encoder and decoder cells replaced with **GRU** units, reducing parameter count while retaining sequential inductive bias and improving both training stability and speed.

- An optional **attention-head diversity regularisation** term is added to the training loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SMAPE}} + \lambda \cdot \Omega_{\text{div}}$$

where $\Omega_{\text{div}} = \frac{1}{\binom{H}{2}} \sum_{i < j} \langle \mathbf{a}_i, \mathbf{a}_j \rangle$ penalises mean pairwise cosine similarity across the $H$ attention heads, encouraging specialisation across temporal scales.

**Scheduler**: Cosine annealing with $\eta_{\min} = 0.01 \cdot \eta_0$, switchable to `ReduceLROnPlateau` via config.

**Target normalisation**: Per-group softplus normalisation via `GroupNormalizer`.

![Encoder Variable Importance](docs/images/feature_importance_top25.png)
*Top 25 encoder variable importance weights. The interaction feature `temp_x_hour_cos` and seasonal encoding `month_sin` dominate, alongside 2-hour lagged irradiance and sunrise/sunset proximity.*

![Global Temporal Attention Profile](docs/images/global_model_attention_profile.png)
*Aggregated temporal attention weights across the encoder window. Peaks correspond to same-hour lookbacks on previous days, confirming the model exploits strong diurnal periodicity.*

![Kemp Town Forecast](docs/images/kemp_town_final_forecast.png)
*Continuous one-week forecast for Kemp Town substation (June 2025). The model captures daily demand cycles closely, with minor amplitude overestimation on high-irradiance afternoons consistent with residual BTM solar suppression.*

---

## Capacity Estimation: On-Manifold Method (cVAE)

Rather than constructing synthetic weather scenarios from scratch — which can produce physically implausible conditions — the model first learns what real weather actually looks like at each site, then generates scenarios that are guaranteed to stay within that observed range.

**Stage 1 — Learning the weather distribution.** A neural network (cVAE) is trained on years of historical ERA5 weather, learning a compact representation of typical conditions per site, month, and hour. Training minimises:

$$\mathcal{L} = \text{reconstruction error} + \text{latent regularisation}$$

**Stage 2 — Generating extreme scenarios.** The model generates $k$ plausible weather vectors for a given site and time. Each is scored by how solar-favourable it is:

$$s = \sum_j w_j x_j$$

The bottom 20% become the *low-solar* pool; the top 20% become the *high-solar* pool.

**Stage 3 — Estimating capacity.** For each of $N$ draws, the TFT is run under both a low- and high-solar scenario. The difference in predicted power is the estimated solar contribution:

$$\Delta_n = \max(0,\ \hat{P}^{+}_n - \hat{P}^{-}_n)$$

Capacity is the mean delta across all draws:

$$\hat{C} = \frac{1}{N} \sum_{n=1}^{N} \Delta_n$$