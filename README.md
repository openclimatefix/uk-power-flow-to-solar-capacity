# UK Power Flow to Embedded Solar Capacity

Inferring behind-the-meter PV capacity and producing operational solar power forecasts for approximately 600 UK primary substation locations.

## Problem Formulation

Let $P_t^{\text{obs}}$ denote net active power demand observed at a primary substation at half-hourly timestep $t$.

Under the decomposition:

$$P_t^{\text{obs}} = P_t^{\text{demand}} - P_t^{\text{solar}}$$

where $P_t^{\text{solar}} = \eta \cdot C \cdot \text{GHI}_t \cdot f(\theta_t)$ encodes irradiance $\text{GHI}_t$, installed capacity $C$, system efficiency $\eta$, and an angular response term $f(\theta_t)$.

The main inference objective is:

**Capacity estimation**: Recover $\hat{C}$ from historical residual signal, without direct metering.

---

## Data

| Source | Description | Resolution |
|---|---|---|
| UKPN | Half-hourly active power per primary substation | 30 min |
| ERA5 | Reanalysis weather variables (SSRD, T2M, U10, TP, etc.) | Hourly interpolated |
| PVLive (Sheffield Solar) | GSP-level aggregated PV generation actuals | 30 min |

**Coverage**: Approximately 600 UKPN primary substations across the South East and East of England..

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
