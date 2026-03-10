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

---

## Capacity Estimation: On-Manifold Method (cVAE)

The on-manifold method frames BTM capacity estimation as a **counterfactual intervention problem** under a learned weather distribution. Rather than injecting synthetic solar based on parametric irradiance models — which risk physically implausible feature combinations — this approach constrains all scenario generation to the *learned data manifold* of observed ERA5 weather.

**Stage 1 — Manifold Learning.** A conditional variational autoencoder (cVAE; Kingma & Welling, 2014; Sohn et al., 2015) is trained on historical weather feature vectors $\mathbf{x} \in \mathbb{R}^d$, conditioned on site identity $\ell$, calendar month $m$, and hour of day $h$:

$$q_\phi(\mathbf{z} \mid \mathbf{x}, \ell, m, h) = \mathcal{N}(\boldsymbol{\mu}_\phi, \text{diag}(\boldsymbol{\sigma}^2_\phi))$$

The model is optimised via the conditional ELBO:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}\left[\log p_\theta(\mathbf{x} \mid \mathbf{z}, \ell, m, h)\right] - \beta \cdot D_{\text{KL}}\left(q_\phi \| \mathcal{N}(\mathbf{0}, \mathbf{I})\right)$$

where the reconstruction term uses MSE and $\beta = 1$ (standard VAE). Location, month, and hour are embedded via learned lookup tables and concatenated to form the conditioning vector before both encoder and decoder.

**Stage 2 — Counterfactual Scenario Sampling.** At inference time, $k$ weather vectors are sampled from the prior $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and decoded through the conditioned generator. Samples are partitioned into *high-solar* and *low-solar* pools by a weighted scoring function:

$$s(\mathbf{x}) = \sum_j w_j x_j$$

where weights $w_j$ are configured per feature (e.g. positive for `ssrd_w_m2`, negative for `cloud_variability`). Pool membership is determined by the 20th and 80th percentile thresholds of $s(\mathbf{x})$ across the $k$ draws.

**Stage 3 — Capacity Attribution via TFT Delta.** For each of $N$ Monte Carlo draws, a low-solar scenario $\mathbf{v}^{-}$ and high-solar scenario $\mathbf{v}^{+}$ are injected into the observed site time series during daylight hours. The TFT is run under both conditions, and a linear calibration $\hat{y} \leftarrow a\hat{y} + b$ (fitted on the validation split via ordinary least-squares) corrects for systematic model bias before computing the solar impact delta:

$$\Delta_n = \max\left(0,\ \hat{P}^{+}_n - \hat{P}^{-}_n\right)$$

The estimated installed capacity is then:

$$\hat{C} = \frac{1}{N} \sum_{n=1}^{N} \Delta_n$$

with the P95 reported as an uncertainty bound. The sign of $\Delta$ is adjusted by an orientation flag inferred from the Pearson correlation between observed power and irradiance during summer midday windows, handling both generation-positive and net-load-positive site conventions.