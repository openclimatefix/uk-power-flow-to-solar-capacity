# UK Power Flow to Embedded Solar Capacity

Inferring behind-the-meter PV capacity and producing operational solar power forecasts for approximately 600 UK primary substation locations.

[![CI](https://github.com/openclimatefix/uk-power-flow-to-solar-capacity/actions/workflows/ci.yml/badge.svg)](https://github.com/openclimatefix/uk-power-flow-to-solar-capacity/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Motivation

The rapid, largely unobserved deployment of distributed solar PV across the UK distribution network has created a significant observability gap for distribution network operators (DNOs). Behind-the-meter (BTM) solar generation reduces net load at primary substations without being directly metered, producing characteristic daytime suppression signatures in half-hourly active power measurements. Without reliable estimates of embedded BTM capacity and accurate short-horizon generation forecasts, DNOs are unable to distinguish solar-driven load suppression from genuine demand reduction, compromising flexibility dispatch, network constraint management, and long-term investment planning.

This repository addresses that gap by combining statistical capacity inference with deep learning-based generation forecasting across the full fleet of 600+ UKPN primary substations.

---

## Problem Formulation

Let $P_t^{\text{obs}}$ denote the net active power observed at a primary substation at half-hourly timestep $t$. Under the decomposition:

$$P_t^{\text{obs}} = P_t^{\text{demand}} - P_t^{\text{solar}}$$

where $P_t^{\text{solar}} = \eta \cdot C \cdot \text{GHI}_t \cdot f(\theta_t)$ encodes irradiance $\text{GHI}_t$, installed capacity $C$, system efficiency $\eta$, and an angular response term $f(\theta_t)$.

The dual inference objectives are:

1. **Capacity estimation**: Recover $\hat{C}$ from the historical residual signal, without direct metering.
2. **Generation forecasting**: Predict $\hat{P}_{t+h}^{\text{solar}}$ for horizon $h \in \{1, \ldots, 12\}$ half-hour steps (6-hour nowcast).

---

## Data

| Source | Description | Resolution |
|---|---|---|
| UKPN SCADA | Half-hourly active power per primary substation | 30 min |
| ERA5 (ECMWF) | Reanalysis weather variables (SSRD, T2M, U10, TP, etc.) | Hourly → interpolated |
| PVLive (Sheffield Solar) | GSP-level aggregated PV generation actuals | 30 min |
| OS / Satellite imagery | Manual PV panel counts for validation method 4 | — |

**Coverage**: ~600 UKPN primary substations across the South East, East of England, and East Midlands licence areas.

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
