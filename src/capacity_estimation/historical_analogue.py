"""Year-on-year solar capacity estimation via historical weather analogues."""

from __future__ import annotations

import gc
import logging
import warnings

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def ensure_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp to tz-naive UTC index.

    Args:
        df: Input DataFrame with a timestamp column.

    Returns:
        Copy of df with normalised timestamp and time_idx columns.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    origin = df["timestamp"].min()
    df["time_idx"] = ((df["timestamp"] - origin).dt.total_seconds() // 1800).astype(int)
    return df


def estimate_capacity_for_window(
    eval_window: pd.DataFrame,
    df_full: pd.DataFrame,
    model: TemporalFusionTransformer,
    features: list[str],
    analogue_years: list[int],
    cfg: DictConfig,
) -> float | None:
    """Estimate solar capacity for single encoder + horizon window.

    Builds high and low irradiance analogue scenarios by sampling from
    historical windows matching, then computes mean daytime delta
    between the two model predictions.

    Args:
        eval_window: Encoder + horizon slice for one site.
        df_full: Full multi-site DataFrame used as the analogue library.
        model: Loaded TemporalFusionTransformer.
        features: Weather feature columns to overwrite per scenario.
        analogue_years: Years to draw analogues from.
        cfg: Full Hydra config.

    Returns:
        Estimated capacity in MW.
    """
    params = model.dataset_parameters
    max_enc = params.get("max_encoder_length", 336)
    max_pred = params.get("max_prediction_length", 96)

    if len(eval_window) < (max_enc + max_pred):
        return None

    window = eval_window.iloc[: max_enc + max_pred].copy()

    # Restrict analogue library to the configured historical years
    lib = df_full[df_full["timestamp"].dt.year.isin(analogue_years)]
    if lib.empty:
        return None

    # Partition library into extreme irradiance pools using configured percentiles
    h_pool = lib[lib["ssrd_w_m2"] >= lib["ssrd_w_m2"].quantile(cfg.analogue_percentile_high / 100)]
    l_pool = lib[lib["ssrd_w_m2"] <= lib["ssrd_w_m2"].quantile(cfg.analogue_percentile_low / 100)]

    def _build_scenario(base: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
        # Replace weather features with analogue samples
        scen = base.copy()
        for idx in scen.index:
            h = scen.loc[idx, "timestamp"].hour
            match = library[library["timestamp"].dt.hour == h]
            sample = match.sample(1, random_state=42) if not match.empty else library.sample(1)
            scen.loc[idx, features] = sample[features].to_numpy()
        return scen

    def _predict_mw(df_in: pd.DataFrame) -> np.ndarray:
        ds = TimeSeriesDataSet.from_parameters(
            params, df_in, predict=True, stop_randomization=True, allow_missing_timesteps=True
        )
        dl = ds.to_dataloader(train=False, batch_size=1)
        with torch.no_grad():
            return model.predict(dl, mode="prediction").detach().cpu().numpy().flatten()

    y_high = _predict_mw(_build_scenario(window, h_pool))
    y_low = _predict_mw(_build_scenario(window, l_pool))

    # Restrict delta to configured solar peak hours
    peak_start, peak_end = cfg.peak_hours
    p_times = pd.to_datetime(window.iloc[max_enc : max_enc + max_pred]["timestamp"].to_numpy())
    mask = (p_times.hour >= peak_start) & (p_times.hour <= peak_end)

    if mask.sum() == 0:
        return None

    # Capacity is mean daytime reduction in output under high vs low irradiance
    return max(0.0, float((y_low[mask] - y_high[mask]).mean()))


def estimate_site_capacity_year(
    site_df: pd.DataFrame,
    df_full: pd.DataFrame,
    model: TemporalFusionTransformer,
    features: list[str],
    year: int,
    analogue_years: list[int],
    cfg: DictConfig,
) -> dict[str, float | int] | None:
    """Estimate capacity for one site in one year via sliding window averaging.

    Args:
        site: Site identifier string.
        site_df: Single-site DataFrame.
        df_full: Full multi-site DataFrame for analogue library.
        model: Loaded TemporalFusionTransformer.
        features: Weather feature columns to overwrite per scenario.
        year: Target year to estimate.
        analogue_years: Years to draw analogues from.
        cfg: Full Hydra config.

    Returns:
        Dict with keys mean, p90, n_samples, or None if insufficient windows.
    """
    max_enc = model.dataset_parameters.get("max_encoder_length", 336)
    max_pred = model.dataset_parameters.get("max_prediction_length", 96)
    window_size = max_enc + max_pred

    site_year = (
        site_df[site_df["timestamp"].dt.year == year]
        .sort_values("time_idx")
        .reset_index(drop=True)
    )

    if len(site_year) < cfg.min_samples_per_site:
        return None

    estimates = []
    for start_idx in range(0, len(site_year) - window_size, cfg.step_size):
        window = site_year.iloc[start_idx : start_idx + window_size]
        capacity = estimate_capacity_for_window(
            window, df_full, model, features, analogue_years, cfg
        )
        if capacity is not None:
            estimates.append(capacity)

    if len(estimates) < cfg.min_windows_for_estimate:
        return None

    arr = np.array(estimates)
    return {
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "n_samples": len(arr)
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/capacity_estimation",
    config_name="historical_analogue"
)
def main(cfg: DictConfig) -> None:
    """Main function.

    Args:
        cfg: Config injected automatically.
    """
    logger.info("Loading model from %s", cfg.paths.checkpoint)
    model = TemporalFusionTransformer.load_from_checkpoint(cfg.paths.checkpoint)
    trained_sites = list(model.dataset_parameters["categorical_encoders"]["location"].classes_)

    logger.info("Loading data from %s", cfg.paths.data)
    df = pd.read_parquet(cfg.paths.data, filters=[("location", "in", trained_sites)])
    df = ensure_time_idx(df)

    actual_features = [f for f in cfg.features if f in df.columns]
    years = list(cfg.years_to_analyze)
    logger.info("Analysing %d sites across years %s", len(trained_sites), years)

    results = []

    for i, site in enumerate(sorted(trained_sites), 1):
        logger.info("[%d/%d] %s", i, len(trained_sites), site)
        site_df = df[df["location"] == site].sort_values("time_idx")
        site_results: dict[str, object] = {"location": site}

        for year in years:
            # Use previous two years as analogues; fall back to configured fallback years
            analogue_years = (
                [year - 2, year - 1] if year > years[0] else list(cfg.analogue_fallback_years)
            )
            estimate = estimate_site_capacity_year(
                site, site_df, df, model, actual_features, year, analogue_years, cfg
            )

            if estimate:
                site_results[f"capacity_{year}_mean"] = round(estimate["mean"], 4)
                site_results[f"capacity_{year}_p90"] = round(estimate["p90"], 4)
                site_results[f"n_samples_{year}"] = estimate["n_samples"]
                logger.info("  %d: %.3f MW (n=%d)", year, estimate["mean"], estimate["n_samples"])
            else:
                site_results[f"capacity_{year}_mean"] = None
                site_results[f"capacity_{year}_p90"] = None
                site_results[f"n_samples_{year}"] = 0
                logger.info("  %d: No estimate", year)

        # Year-on-year growth between consecutive years
        for j in range(len(years) - 1):
            year1, year2 = years[j], years[j + 1]
            cap1 = site_results.get(f"capacity_{year1}_mean")
            cap2 = site_results.get(f"capacity_{year2}_mean")

            if cap1 and cap2 and cap1 > 0:
                growth_abs = float(cap2) - float(cap1)
                site_results[f"growth_{year1}_{year2}_mw"] = round(growth_abs, 4)
                site_results[f"growth_{year1}_{year2}_pct"] = round(
                    (growth_abs / float(cap1)) * 100, 2
                )
            else:
                site_results[f"growth_{year1}_{year2}_mw"] = None
                site_results[f"growth_{year1}_{year2}_pct"] = None

        results.append(site_results)

        if i % cfg.sites_per_gc == 0:
            gc.collect()
            torch.cuda.empty_cache()

    df_results = pd.DataFrame(results)
    df_results.to_csv(cfg.paths.output_csv, index=False)
    logger.info("Results saved to %s", cfg.paths.output_csv)

    # All location summary
    for j in range(len(years) - 1):
        year1, year2 = years[j], years[j + 1]
        growth_col = f"growth_{year1}_{year2}_mw"
        growth_pct_col = f"growth_{year1}_{year2}_pct"
        valid = df_results[df_results[growth_col].notna()]

        if len(valid) > 0:
            n_positive = (valid[growth_col] > 0).sum()
            logger.info(
                "%d → %d | sites=%d | avg=%.3f MW (%.1f%%) | median=%.1f%% | positive=%d/%d",
                year1, year2,
                len(valid),
                valid[growth_col].mean(),
                valid[growth_pct_col].mean(),
                valid[growth_pct_col].median(),
                n_positive,
                len(valid),
            )

    year1, year2 = years[-2], years[-1]
    growth_col = f"growth_{year1}_{year2}_mw"
    if growth_col in df_results.columns:
        top_cols = [
            "location",
            f"capacity_{year1}_mean",
            f"capacity_{year2}_mean",
            growth_col,
            f"growth_{year1}_{year2}_pct",
        ]
        logger.info(
            "Top 10 growing sites (%d→%d):\n%s",
            year1, year2,
            df_results.nlargest(10, growth_col)[top_cols].to_string(index=False),
        )


if __name__ == "__main__":
    main()
