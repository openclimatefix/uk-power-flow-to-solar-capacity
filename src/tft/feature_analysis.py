from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.tft.utils import (
    configure_logging,
    get_logger,
    load_yaml,
    ensure_keys,
    coerce_to_dict,
    save_json,
)
from src.tft.plotting import plot_tft_interpretation


CONFIG_PATH = Path("config_tft_initial.yaml")
config = load_yaml(CONFIG_PATH)


log_cfg = config.get("logging", {})
level = str(log_cfg.get("level", "INFO"))
fmt = str(log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
configure_logging(level=level, fmt=fmt)
logger = get_logger("feature_analysis")


if log_cfg.get("file", False):
    paths_cfg = config.get("paths", {})
    log_file = Path(paths_cfg.get("log_file", "tft_training.log"))
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger().addHandler(fh)


ensure_keys(config, ["paths", "output_path"])
ensure_keys(config, ["model", "time_idx"])
logger.info("Configuration loaded successfully")


def analyze_weather_features(model: TemporalFusionTransformer, test_data_df: pd.DataFrame) -> dict[str, Any]:
    training_params = model.dataset_parameters
    test_dataset = TimeSeriesDataSet.from_parameters(
        training_params,
        test_data_df,
        predict=True,
        stop_randomization=True,
    )
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    reduction = config.get("analysis", {}).get("interpretation", {}).get("reduction", "sum")
    raw_predictions = model.predict(
        test_dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs={"accelerator": "cpu"},
    )
    raw_output = getattr(raw_predictions, "output", raw_predictions)
    raw_output = coerce_to_dict(raw_output)
    interpretation = model.interpret_output(raw_output, reduction=reduction)
    encoder_importance = interpretation["encoder_variables"]
    decoder_importance = interpretation["decoder_variables"]
    attention = interpretation["attention"]
    encoder_vars = list(model.encoder_variables)
    decoder_vars = list(model.decoder_variables)
    all_vars = encoder_vars + decoder_vars
    all_importance = np.concatenate([np.asarray(encoder_importance), np.asarray(decoder_importance)])
    weather_categories = {
        "Solar": [
            "ssrd_w_m2",
            "ssr_w_m2",
            "str_w_m2",
            "slhf_w_m2",
            "sshf_w_m2",
            "ishf_w_m2",
            "ssrd_w_m2_lag_1h",
            "ssrd_w_m2_lag_24h",
            "ssrd_w_m2_roll_mean_3h",
            "irradiance_x_clear_sky",
            "solar_elevation_angle",
            "solar_azimuth_angle",
            "solar_zenith_angle",
            "solar_intensity_factor",
        ],
        "Temperature": [
            "t2m_c",
            "d2m_c",
            "skt_c",
            "t2m_c_lag_1h",
            "t2m_c_lag_24h",
            "t2m_c_roll_mean_3h",
            "temp_x_hour_sin",
            "temp_x_hour_cos",
        ],
        "Wind": [
            "u10",
            "v10",
            "wind_speed",
            "wind_speed_lag_1h",
            "wind_speed_lag_24h",
            "wind_speed_roll_mean_3h",
            "wind_dir_sin",
            "wind_dir_cos",
        ],
        "Atmospheric": ["tcc", "blh", "msl_hpa", "sp_hpa", "tp_mm", "weather_index"],
    }
    category_importance: dict[str, float] = {}
    for category, features in weather_categories.items():
        total = 0.0
        for feature in features:
            if feature in all_vars:
                idx = all_vars.index(feature)
                total += float(all_importance[idx])
        category_importance[category] = total
    feature_importance = sorted([(f, float(i)) for f, i in zip(all_vars, all_importance)], key=lambda x: x[1], reverse=True)
    category_ranking = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    encoder_len = int(model.hparams.max_encoder_length)
    att = np.asarray(attention)
    if att.ndim == 3:
        attention_mean = att.mean(axis=(0, 1))
    elif att.ndim == 2:
        attention_mean = att.mean(axis=0)
    elif att.ndim == 1:
        attention_mean = att
    else:
        attention_mean = att.mean(axis=tuple(range(att.ndim - 1)))
    if attention_mean.size >= encoder_len:
        hist_attention = float(attention_mean[:encoder_len].mean())
        future_attention = float(attention_mean[encoder_len:].mean()) if attention_mean.size > encoder_len else 0.0
    else:
        hist_attention = float(attention_mean.mean())
        future_attention = 0.0
    logger.info("Feature importance analysis results:")
    for i, (feature, imp) in enumerate(feature_importance[:10], start=1):
        logger.info(f"{i:2d}. {feature:<35s} {imp:>8.4f}")
    total_category_importance = sum(category_importance.values()) or 1.0
    for i, (cat, imp) in enumerate(category_ranking, start=1):
        pct = 100.0 * imp / total_category_importance
        logger.info(f"{i}. {cat:<12s} {imp:>8.4f} ({pct:>5.1f}%)")
    total_attention = hist_attention + future_attention or 1.0
    hist_pct = 100.0 * hist_attention / total_attention
    future_pct = 100.0 * future_attention / total_attention
    logger.info(
        f"Temporal attention - Historical: {hist_attention:.4f} ({hist_pct:.1f}%), "
        f"Future: {future_attention:.4f} ({future_pct:.1f}%)"
    )
    results: dict[str, Any] = {
        "top_25_features": [{"feature": f, "importance": float(i)} for f, i in feature_importance[:25]],
        "category_importance": [{"category": c, "importance": float(i)} for c, i in category_ranking],
        "temporal_attention": {
            "historical_avg": hist_attention,
            "future_avg": future_attention,
            "historical_percentage": hist_pct,
            "future_percentage": future_pct,
        },
        "model_info": {
            "encoder_length": encoder_len,
            "total_encoder_vars": len(encoder_vars),
            "total_decoder_vars": len(decoder_vars),
            "total_features": len(all_vars),
        },
        "raw": {
            "all_vars": all_vars,
            "all_importance": [float(x) for x in all_importance],
            "attention_mean": [float(x) for x in attention_mean],
        },
    }
    return results


def main() -> None:
    analysis_cfg = config.get("analysis", {})
    data_path = Path(analysis_cfg.get("data_path", config["paths"]["output_path"]))
    ckpt_default = Path.home() / "production_tft_model.ckpt"
    checkpoint = Path(config.get("paths", {}).get("checkpoint", ckpt_default))
    rng_seed = int(config.get("data_processing", {}).get("random_state", 42))
    n_locations = int(config.get("num_locations", 5))
    val_split = float(config.get("val_split", 0.8))
    time_idx_col = config["model"]["time_idx"]
    logger.info("Starting TFT feature importance analysis")
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
        logger.info(f"Model loaded from {checkpoint}")
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {checkpoint}")
        return
    logger.info(f"Loading data from {data_path}")
    df_full = pd.read_parquet(data_path)
    rng = np.random.default_rng(rng_seed)
    all_locations = np.unique(df_full["location"])
    n_sites = min(n_locations, len(all_locations))
    selected = rng.choice(all_locations, size=n_sites, replace=False)
    df = df_full[df_full["location"].isin(selected)].copy()
    logger.info(f"Filtered data to {n_sites} sites: {list(selected)}")
    max_time_idx = int(df[time_idx_col].max())
    test_df = df[df[time_idx_col] > int(max_time_idx * val_split)]
    logger.info(f"Test data prepared: {len(test_df)} rows")
    results = analyze_weather_features(model, test_df)
    logger.info("Analysis completed successfully")
    save_cfg = analysis_cfg.get("save_plots", {})
    out_dir = Path(save_cfg.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "feature_analysis_results.json"
    trimmed = {k: v for k, v in results.items() if k != "raw"}
    save_json(trimmed, out_json)
    logger.info(f"Detailed results saved to {out_json}")
    plot_tft_interpretation(results, config)
    logger.info("Summary:")
    logger.info(f"Most important feature: {results['top_25_features'][0]['feature']}")
    logger.info(f"Most important category: {results['category_importance'][0]['category']}")
    logger.info(f"Model focuses {results['temporal_attention']['historical_percentage']:.1f}% on historical data")


if __name__ == "__main__":
    main()
