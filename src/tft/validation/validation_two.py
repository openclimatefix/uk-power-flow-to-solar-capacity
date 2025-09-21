import os
import json
import logging
import warnings
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml") as f:
    config = yaml.safe_load(f)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tft_scenarios")

TRAINED_SITE_FALLBACK = [
    "belchamp_grid_11kv",
    "george_hill_primary_11kv",
    "manor_way_primary_11kv",
    "st_stephens_primary_11kv",
    "swaffham_grid_11kv",
]

V2_VARIANT_COLS = ["v2", "v2_1p0", "v2_0p5", "v2_0p1"]

OCF_COLORS = {
    "primary_orange": "#FF4901",
    "secondary_orange": "#FF8F73",
    "dark_gray": "#292B2B",
    "very_dark_gray": "#0C0D0D",
    "white": "#FFFFFF",
    "off_white": "#FFFBF5",
    "light_beige": "#F0ECE8",
    "medium_beige": "#D9D0CA",
}

OCF_PALETTE = [
    OCF_COLORS["primary_orange"],
    OCF_COLORS["secondary_orange"],
    OCF_COLORS["dark_gray"],
    OCF_COLORS["very_dark_gray"],
]


def setup_ocf_plot_style():
    plt.style.use("default")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=OCF_PALETTE)
    plt.rcParams["figure.facecolor"] = OCF_COLORS["off_white"]
    plt.rcParams["axes.facecolor"] = OCF_COLORS["white"]
    plt.rcParams["text.color"] = OCF_COLORS["very_dark_gray"]
    plt.rcParams["axes.labelcolor"] = OCF_COLORS["dark_gray"]
    plt.rcParams["xtick.color"] = OCF_COLORS["dark_gray"]
    plt.rcParams["ytick.color"] = OCF_COLORS["dark_gray"]
    plt.rcParams["grid.color"] = OCF_COLORS["medium_beige"]
    plt.rcParams["grid.alpha"] = 0.6


def ensure_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location", "timestamp"]).copy()
    if "time_idx" not in df.columns or df["time_idx"].isna().any():
        df["time_idx"] = df.groupby("location")["timestamp"].transform(lambda s: pd.factorize(s.values)[0].astype(np.int64))
    return df


def _infer_pv_orientation(df: pd.DataFrame) -> tuple[int, float]:
    d = df.loc[
        (df["timestamp"].dt.month.isin([6, 7])) & (df["timestamp"].dt.hour.between(11, 14)),
        ["passive_pv_generation_mw", "active_power_mw"],
    ].dropna()
    if len(d) < 50:
        d = df[["passive_pv_generation_mw", "active_power_mw"]].dropna()
    corr = d["passive_pv_generation_mw"].corr(d["active_power_mw"]) if len(d) else np.nan
    if np.isnan(corr):
        corr = -0.5
    sign_flag = -1 if corr < 0 else 1
    return sign_flag, float(corr)


class SingleSiteScenarioAnalyzer:
    def __init__(self, model_path: str, metadata_path: str, data_path: str, target_location: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.target_location = target_location
        self.model = None
        self.metadata = None
        self.training_dataset: TimeSeriesDataSet | None = None
        self.encoder_length = 168
        self.model_trained_target: str | None = None
        self.trained_locations: List[str] = []
        self.proxy_location: str | None = None
        self.ds_params_used: Dict[str, List[str]] = {}

    def choose_proxy_location(self, target_name: str, trained_locs: List[str]) -> str:
        if not trained_locs:
            return target_name
        h = int(hashlib.sha256(target_name.encode("utf-8")).hexdigest(), 16)
        return trained_locs[h % len(trained_locs)]

    def get_model_target_name(self) -> str:
        if self.model_trained_target:
            return self.model_trained_target
        tgt = None
        try:
            if hasattr(self.model, "hparams"):
                hp = self.model.hparams
                if isinstance(hp, dict):
                    tgt = hp.get("target", None)
                else:
                    tgt = getattr(hp, "target", None)
            if not tgt and hasattr(self.model, "dataset_parameters"):
                tgt = self.model.dataset_parameters.get("target")
            if not tgt:
                tgt = "active_power_mw"
        except Exception:
            tgt = "active_power_mw"
        self.model_trained_target = tgt
        return tgt

    def load_and_prepare_data(self) -> pd.DataFrame | None:
        logger.info("Loading data from: %s", self.data_path)
        df_full = pd.read_parquet(self.data_path)
        logger.info("Filtering for target location: %s", self.target_location)
        df_site = df_full[df_full["location"] == self.target_location].copy()
        if df_site.empty:
            logger.error("Location '%s' not found.", self.target_location)
            return None
        df_site["timestamp"] = pd.to_datetime(df_site["timestamp"], utc=True)
        df_site = ensure_time_idx(df_site)
        pv_check = df_site[
            (df_site["timestamp"].dt.year == 2024)
            & (df_site["timestamp"].dt.month.isin([6, 7]))
            & (df_site["timestamp"].dt.hour.between(11, 14))
            & (df_site["passive_pv_generation_mw"] > 0)
        ]
        if pv_check.empty:
            logger.error("No PV>0 found for %s in 2024-06/07 11–14 UTC.", self.target_location)
            return None
        logger.info("Data prepared for %s with %s records.", self.target_location, f"{len(df_site):,}")
        return df_site

    def find_high_pv_periods(self, df: pd.DataFrame, n_periods: int = 200) -> pd.DataFrame:
        logger.info("Finding top %d PEAK SOLAR periods...", n_periods)
        df_summer = df[(df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([6, 7]))].copy()
        df_peak_hours = df_summer[(df_summer["timestamp"].dt.hour >= 11) & (df_summer["timestamp"].dt.hour <= 14)].copy()
        if "tcc" in df_peak_hours.columns:
            df_peak_hours = df_peak_hours.sort_values("tcc")
        high_pv_periods = df_peak_hours.nlargest(n_periods, "passive_pv_generation_mw").sort_values("timestamp")
        logger.info("Identified %d PEAK SOLAR periods for analysis.", len(high_pv_periods))
        return high_pv_periods

    def load_model_and_setup(self):
        logger.info("Loading TFT model and metadata...")
        self.metadata = torch.load(self.metadata_path, map_location="cpu")
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path, map_location="cpu")
        self.model.eval()
        logger.info("Model and metadata loaded successfully.")
        logger.info("Model trained target: '%s'", self.get_model_target_name())

    def prepare_data_splits(self, df: pd.DataFrame):
        val_cutoff_date = pd.Timestamp("2024-08-01", tz="UTC")
        val_cutoff = df[df["timestamp"] < val_cutoff_date]["time_idx"].max()
        logger.info("Data splits defined with validation cutoff at time_idx: %s", val_cutoff)
        return df, val_cutoff

    def create_training_dataset_reference(self, df: pd.DataFrame, val_cutoff: int):
        ds_params = None
        if hasattr(self.model, "dataset_parameters"):
            ds_params = dict(self.model.dataset_parameters)
        elif isinstance(self.metadata, dict) and "dataset_parameters" in self.metadata:
            ds_params = dict(self.metadata["dataset_parameters"])

        train_df = df[df["time_idx"] <= val_cutoff].copy()
        trained_locs: list = []
        if isinstance(ds_params, dict):
            try:
                cat_encs = ds_params.get("categorical_encoders", {}) or {}
                loc_enc = cat_encs.get("location", None)
                classes = None
                if loc_enc is not None:
                    classes = getattr(loc_enc, "classes_", None)
                if classes is None and hasattr(loc_enc, "encoder"):
                    classes = getattr(loc_enc.encoder, "classes_", None)
                if classes is None and hasattr(loc_enc, "__dict__"):
                    classes = loc_enc.__dict__.get("classes_", None)
                if classes is not None:
                    trained_locs = list(classes)
            except Exception as e:
                logger.warning("Could not extract trained locations from ds_params: %s", e)
        if not trained_locs:
            trained_locs = TRAINED_SITE_FALLBACK[:]
            logger.info("Using fallback trained sites: %s", trained_locs)
        self.trained_locations = trained_locs
        if self.target_location not in self.trained_locations:
            self.proxy_location = self.choose_proxy_location(self.target_location, self.trained_locations)
            logger.info("Target '%s' not in training set. Using proxy '%s'.", self.target_location, self.proxy_location)
            train_df.loc[:, "location"] = self.proxy_location
        else:
            self.proxy_location = self.target_location
            logger.info("Target '%s' is a trained location (no proxy needed).", self.target_location)

        if ds_params:
            logger.info("Reconstructing dataset from model's saved dataset parameters.")
            ds_params = dict(ds_params)
            ds_params["target"] = self.get_model_target_name()
            self.training_dataset = TimeSeriesDataSet(train_df, **ds_params)
        else:
            logger.warning("Falling back to config-based dataset construction.")
            model_config = {**config["model"], "target": self.get_model_target_name()}
            self.training_dataset = TimeSeriesDataSet(
                train_df,
                time_idx=model_config["time_idx"],
                target=model_config["target"],
                group_ids=model_config["group_ids"],
                max_encoder_length=model_config["max_encoder_length"],
                max_prediction_length=model_config["max_prediction_length"],
                static_categoricals=model_config["static_categoricals"],
                static_reals=model_config["static_reals"],
                time_varying_known_reals=model_config["time_varying_known_reals"],
                time_varying_unknown_reals=model_config["time_varying_unknown_reals"],
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
        self._init_used_feature_lists(ds_params)
        logger.info("Trained locations: %s", self.trained_locations)
        logger.info("Proxy in use: %s", self.proxy_location)

    def _init_used_feature_lists(self, ds_params: Dict[str, Any] | None):
        used: Dict[str, List[str]] = {}
        src = ds_params if isinstance(ds_params, dict) else {}
        for k in [
            "static_categoricals",
            "static_reals",
            "time_varying_known_reals",
            "time_varying_unknown_reals",
            "time_varying_known_categoricals",
            "time_varying_unknown_categoricals",
        ]:
            vals = src.get(k, [])
            used[k] = list(vals) if isinstance(vals, (list, tuple)) else []
        if not any(used.values()):
            mcfg = config.get("model", {})
            for k in [
                "static_categoricals",
                "static_reals",
                "time_varying_known_reals",
                "time_varying_unknown_reals",
                "time_varying_known_categoricals",
                "time_varying_unknown_categoricals",
            ]:
                vals = mcfg.get(k, [])
                used[k] = list(vals) if isinstance(vals, (list, tuple)) else []
        td = self.training_dataset
        if td is not None:
            for attr in [
                "static_categoricals",
                "static_reals",
                "time_varying_known_reals",
                "time_varying_unknown_reals",
                "time_varying_known_categoricals",
                "time_varying_unknown_categoricals",
            ]:
                vals = getattr(td, attr, None)
                if isinstance(vals, (list, tuple)):
                    used.setdefault(attr, [])
                    for v in vals:
                        if isinstance(v, str) and v not in used[attr]:
                            used[attr].append(v)
            for attr in ["reals", "categoricals"]:
                rc = getattr(td, attr, None)
                if isinstance(rc, dict):
                    for side in ("encoder", "decoder"):
                        vals = rc.get(side, [])
                        if isinstance(vals, (list, tuple)):
                            bucket = "time_varying_unknown_reals" if attr == "reals" else "time_varying_unknown_categoricals"
                            used.setdefault(bucket, [])
                            for v in vals:
                                if isinstance(v, str) and v not in used[bucket]:
                                    used[bucket].append(v)
        self.ds_params_used = used

    def _collect_used_feature_names(self) -> set:
        used = set()
        if isinstance(self.ds_params_used, dict):
            for k in self.ds_params_used:
                vals = self.ds_params_used.get(k, [])
                if isinstance(vals, (list, tuple)):
                    used.update([v for v in vals if isinstance(v, str)])
        return used

    def _filter_modifications_to_used_features(self, modifications: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(modifications, dict):
            return {}
        used_names = self._collect_used_feature_names()
        if not used_names:
            logger.warning("Could not determine used feature names; applying all modifications as-is.")
            return dict(modifications)
        mods_kept = {k: v for k, v in modifications.items() if k in used_names}
        mods_dropped = sorted(set(modifications.keys()) - set(mods_kept.keys()))
        if mods_dropped:
            logger.warning("Scenario features NOT used by model (ignored): %s", mods_dropped)
        return mods_kept

    def _fit_pv_proxy(self, df: pd.DataFrame) -> dict:
        candidates = [
            "ssrd_w_m2_roll_mean_3h",
            "ssrd_w_m2",
            "ssr_w_m2",
            "irradiance_x_clear_sky",
            "solar_intensity_factor",
        ]
        m = (df["timestamp"].dt.month.isin([6, 7])) & (df["timestamp"].dt.hour.between(11, 14))
        best = {}
        for name in candidates:
            if name not in df.columns:
                continue
            d = df.loc[m, [name, "passive_pv_generation_mw"]].dropna()
            if len(d) < 200:
                d = df[[name, "passive_pv_generation_mw"]].dropna()
            if d.empty or d[name].std() == 0:
                continue
            x = d[name].astype(float).values
            y = d["passive_pv_generation_mw"].astype(float).values
            X = np.vstack([x, np.ones_like(x)]).T
            alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
            r = float(np.corrcoef(x, y)[0, 1])
            if not best or abs(r) > abs(best.get("corr", 0.0)):
                best = {"name": name, "alpha": float(alpha), "beta": float(beta), "corr": r}
        return best

    def _estimate_k_from_history(self, df: pd.DataFrame, variant_col: str) -> float:
        if variant_col not in df.columns:
            return np.nan
        d = df.loc[
            (df["timestamp"].dt.month.isin([6, 7])) & (df["timestamp"].dt.hour.between(11, 14)),
            ["active_power_mw", variant_col, "passive_pv_generation_mw"],
        ].dropna()
        if d.empty:
            return np.nan
        diff = d["active_power_mw"] - d[variant_col]
        pv = d["passive_pv_generation_mw"].replace(0, np.nan)
        k = (diff / pv).replace([np.inf, -np.inf], np.nan).dropna()
        return float(k.median()) if not k.empty else np.nan

    def _pv_raw_from_scenario(
        self,
        df: pd.DataFrame,
        timestamps_ref: List[pd.Timestamp],
        filtered_scenarios: Dict[str, Dict],
        proxy: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        proxy_name, a, b = proxy["name"], proxy["alpha"], proxy["beta"]
        n = len(timestamps_ref)
        v_min = filtered_scenarios.get("MinSolar", {}).get(proxy_name, np.nan)
        v_max = filtered_scenarios.get("MaxSolar", {}).get(proxy_name, np.nan)
        if np.isfinite(v_min) and np.isfinite(v_max):
            pv_raw_min = np.full(n, a * float(v_min) + b, dtype=float)
            pv_raw_max = np.full(n, a * float(v_max) + b, dtype=float)
            return pv_raw_min, pv_raw_max
        df_idx = df.set_index("time_idx")
        t_idx = df_idx.loc[df_idx["timestamp"].isin(timestamps_ref)].index.values
        pv_raw_min, pv_raw_max = [], []
        for t in t_idx:
            enc = df_idx.loc[(df_idx.index >= t - self.encoder_length) & (df_idx.index < t)]
            if proxy_name in enc.columns and not enc.empty:
                x = float(enc.iloc[-1][proxy_name])
                pv_est = a * x + b
            else:
                pv_est = np.nan
            pv_raw_min.append(pv_est)
            pv_raw_max.append(pv_est)
        return np.array(pv_raw_min, dtype=float), np.array(pv_raw_max, dtype=float)

    def _build_v2_validation(
        self,
        df: pd.DataFrame,
        timestamps_ref: List[pd.Timestamp],
        filtered_scenarios: Dict[str, Dict],
        delta_nonneg: np.ndarray,
    ) -> dict:
        variants_present = [c for c in V2_VARIANT_COLS if c in df.columns]
        if not variants_present:
            return {"available": False, "reason": "no_v2_columns_found"}
        proxy = self._fit_pv_proxy(df)
        if not proxy:
            return {"available": False, "reason": "no_suitable_proxy"}
        pv_raw_min, pv_raw_max = self._pv_raw_from_scenario(df, timestamps_ref, filtered_scenarios, proxy)
        validations = {}
        for var in variants_present:
            k_hat = self._estimate_k_from_history(df, var)
            if np.isfinite(k_hat) and k_hat > 0:
                validations[var] = {
                    "variant": var,
                    "k_hat_hist": float(k_hat),
                    "mean_expected_delta_min": float(np.nanmean(k_hat * pv_raw_min)),
                    "mean_expected_delta_max": float(np.nanmean(k_hat * pv_raw_max)),
                }
        ordered_keys = [k for k in ["v2_0p1", "v2_0p5", "v2_1p0", "v2"] if k in validations]
        order_vals = [(k, validations[k]["mean_expected_delta_max"]) for k in ordered_keys]
        is_mono = all(order_vals[i][1] <= order_vals[i + 1][1] for i in range(len(order_vals) - 1)) if len(order_vals) >= 2 else True
        ts_ref = pd.to_datetime(pd.Series(timestamps_ref), utc=True)
        df_sel = df.loc[df["timestamp"].isin(ts_ref)].copy().set_index("timestamp").sort_index()
        df_sel = df_sel.reindex(ts_ref)
        variant_delta_seqs = {}
        for var in variants_present:
            if var not in df_sel.columns:
                continue
            seq = (df_sel["active_power_mw"] - df_sel[var]).astype(float)
            variant_delta_seqs[var] = seq.values.tolist()

        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3:
                return np.nan
            a, b = a[m], b[m]
            if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
                return np.nan
            return float(np.corrcoef(a, b)[0, 1])

        corr_delta_pvmax = _safe_corr(delta_nonneg, pv_raw_max)

        return {
            "available": bool(variant_delta_seqs),
            "proxy": proxy,
            "variants_evaluated": list(variant_delta_seqs.keys()),
            "validations": validations,
            "pv_raw_means": {
                "min": float(np.nanmean(pv_raw_min)),
                "max": float(np.nanmean(pv_raw_max)),
            },
            "pv_raw_min_seq": pv_raw_min.tolist(),
            "pv_raw_max_seq": pv_raw_max.tolist(),
            "variant_delta_seqs": variant_delta_seqs,
            "monotonicity_on_max": {"order": order_vals, "is_monotonic_increasing": bool(is_mono)},
            "corr_deltaV1_vs_pvraw_max": corr_delta_pvmax,
        }

    def define_enhanced_scenarios(self) -> Dict[str, Dict]:
        return {
            "MinSolar": {
                "tcc": 0.90,
                "solar_intensity_factor": 0.08,
                "ssr_w_m2": 120.0,
                "ssrd_w_m2": 100.0,
                "ssrd_w_m2_lag_1h": 100.0,
                "ssrd_w_m2_lag_24h": 100.0,
                "ssrd_w_m2_roll_mean_3h": 120.0,
                "irradiance_x_clear_sky": 0.15,
            },
            "MaxSolar": {
                "tcc": 0.10,
                "solar_intensity_factor": 0.92,
                "ssr_w_m2": 750.0,
                "ssrd_w_m2": 700.0,
                "ssrd_w_m2_lag_1h": 700.0,
                "ssrd_w_m2_lag_24h": 700.0,
                "ssrd_w_m2_roll_mean_3h": 720.0,
                "irradiance_x_clear_sky": 0.95,
            },
        }

    def _build_quantile_scenarios(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        used = self._collect_used_feature_names()
        candidate_names = [
            "ssrd_w_m2",
            "ssrd_w_m2_roll_mean_3h",
            "ssr_w_m2",
            "irradiance_x_clear_sky",
            "solar_intensity_factor",
            "tcc",
            "solar_elevation_angle",
            "solar_zenith_angle",
            "ghi",
            "dhi",
            "dni",
            "cs_ghi",
        ]
        cols = [c for c in candidate_names if (c in df.columns) and (c in used)]
        if not cols:
            logger.warning("No usable solar/cloud features found in df ∩ used_features; scenarios will be empty.")
            return {"MinSolar": {}, "MaxSolar": {}}
        mask = (df["timestamp"].dt.month.isin([6, 7])) & (df["timestamp"].dt.hour.between(11, 14))
        d = df.loc[mask, cols].dropna()
        if len(d) < 200:
            d = df[cols].dropna()

        def _make(lo_q: float, hi_q: float):
            q = d.quantile([lo_q, hi_q])
            lo = q.loc[lo_q].to_dict()
            hi = q.loc[hi_q].to_dict()
            good = []
            min_mods, max_mods = {}, {}
            for c in cols:
                v_lo, v_hi = float(lo.get(c, np.nan)), float(hi.get(c, np.nan))
                if not (np.isfinite(v_lo) and np.isfinite(v_hi) and (v_hi > v_lo)):
                    continue
                if c == "solar_zenith_angle":
                    min_mods[c], max_mods[c] = v_hi, v_lo
                elif c == "solar_elevation_angle":
                    min_mods[c], max_mods[c] = v_lo, v_hi
                else:
                    min_mods[c], max_mods[c] = v_lo, v_hi
                good.append(c)
            return good, min_mods, max_mods

        good, min_mods, max_mods = _make(0.05, 0.95)
        if not good:
            good, min_mods, max_mods = _make(0.10, 0.90)
        if not good:
            lo = d.min(numeric_only=True).to_dict()
            hi = d.max(numeric_only=True).to_dict()
            for c in cols:
                v_lo, v_hi = float(lo.get(c, np.nan)), float(hi.get(c, np.nan))
                if not (np.isfinite(v_lo) and np.isfinite(v_hi) and (v_hi > v_lo)):
                    continue
                if c == "solar_zenith_angle":
                    min_mods[c], max_mods[c] = v_hi, v_lo
                elif c == "solar_elevation_angle":
                    min_mods[c], max_mods[c] = v_lo, v_hi
                else:
                    min_mods[c], max_mods[c] = v_lo, v_hi
                good.append(c)
        if not good:
            logger.warning("All candidate solar/cloud features had zero spread; scenarios will be empty.")
            return {"MinSolar": {}, "MaxSolar": {}}
        logger.info("Quantile scenarios using features: %s", good)
        return {"MinSolar": min_mods, "MaxSolar": max_mods}

    def prepare_single_prediction_input(self, encoder_data: pd.DataFrame, target_time_idx: int, modifications: Dict[str, Any]) -> Dict:
        temp_data = encoder_data.copy()
        target_row = temp_data.iloc[-1:].copy()
        target_row["time_idx"] = target_time_idx
        for feature, value in modifications.items():
            if feature in target_row.columns:
                target_row.loc[:, feature] = value
        temp_data = pd.concat([temp_data, target_row], ignore_index=True)
        temp_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, temp_data, predict=True, stop_randomization=True)
        temp_dataloader = temp_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        for batch in temp_dataloader:
            return batch
        return None

    def make_single_prediction(self, input_batch) -> float:
        with torch.no_grad():
            x, y = input_batch
            pred = self.model.forward(x)
            return float(pred["prediction"].cpu().numpy().flatten()[0])

    def run_analysis(self, df: pd.DataFrame, high_pv_periods: pd.DataFrame, val_cutoff: int) -> Dict[str, Any]:
        trained_target = self.get_model_target_name()
        self.create_training_dataset_reference(df, val_cutoff)
        idx_ts = (
            df.loc[df["time_idx"].isin(high_pv_periods["time_idx"].unique()), ["time_idx", "timestamp"]]
            .drop_duplicates()
            .sort_values("timestamp")
        )
        test_time_indices = idx_ts["time_idx"].tolist()
        scenarios = self.define_enhanced_scenarios()
        filtered_scenarios = {name: self._filter_modifications_to_used_features(mods) for name, mods in scenarios.items()}
        if (not filtered_scenarios.get("MinSolar")) and (not filtered_scenarios.get("MaxSolar")):
            logger.warning("Hardcoded scenarios contained no used features; switching to quantile-based scenarios.")
            filtered_scenarios = self._build_quantile_scenarios(df)
        if filtered_scenarios.get("MinSolar") == filtered_scenarios.get("MaxSolar"):
            logger.warning("MinSolar and MaxSolar are identical; rebuilding scenarios from quantiles.")
            filtered_scenarios = self._build_quantile_scenarios(df)
        if (not filtered_scenarios.get("MinSolar")) and (not filtered_scenarios.get("MaxSolar")):
            logger.error("No usable solar/cloud features available for this checkpoint; Δ cannot be computed.")
            return {}
        logger.info(
            "Scenario feature counts: Min=%d, Max=%d",
            len(filtered_scenarios.get("MinSolar", {})),
            len(filtered_scenarios.get("MaxSolar", {})),
        )
        timestamps_ref: List[pd.Timestamp] = []
        preds_min: List[float] = []
        preds_max: List[float] = []
        for scenario_name in ("MinSolar", "MaxSolar"):
            mods = filtered_scenarios.get(scenario_name, {})
            logger.info("Scenario: %s (applying %d features)", scenario_name, len(mods))
            cur_preds: List[float] = []
            for time_idx in test_time_indices:
                enc = df[(df["time_idx"] >= time_idx - self.encoder_length) & (df["time_idx"] < time_idx)].copy()
                if len(enc) < self.encoder_length:
                    continue
                if self.proxy_location:
                    enc.loc[:, "location"] = self.proxy_location
                for f, v in mods.items():
                    if f in enc.columns:
                        enc.loc[:, f] = v
                if scenario_name == "MinSolar":
                    row_ts = df.loc[df["time_idx"] == time_idx, "timestamp"]
                    if len(row_ts) > 0:
                        timestamps_ref.append(row_ts.iloc[0])
                batch = self.prepare_single_prediction_input(enc, time_idx, mods)
                if batch is not None:
                    cur_preds.append(self.make_single_prediction(batch))
            if scenario_name == "MinSolar":
                preds_min = cur_preds
            else:
                preds_max = cur_preds
        L = min(len(preds_min), len(preds_max), len(timestamps_ref))
        preds_min = np.array(preds_min[:L])
        preds_max = np.array(preds_max[:L])
        timestamps_ref = np.array(timestamps_ref[:L])
        if L == 0:
            logger.error("No scenario predictions produced.")
            return {}
        order = np.argsort(timestamps_ref)
        preds_min = preds_min[order]
        preds_max = preds_max[order]
        timestamps_ref = timestamps_ref[order]
        sep = float(np.median(np.abs(preds_max - preds_min)))
        logger.info("Scenario separation (median |Max-Min|) = %.6f", sep)
        sign_flag, corr = _infer_pv_orientation(df)
        if sign_flag < 0:
            delta_raw = preds_min - preds_max
            orientation = "MinSolar - MaxSolar"
        else:
            delta_raw = preds_max - preds_min
            orientation = "MaxSolar - MinSolar"
        delta = np.maximum(0.0, delta_raw)
        stats = {
            "mean": float(np.mean(delta)),
            "max": float(np.max(delta)),
            "min": float(np.min(delta)),
            "std": float(np.std(delta)),
        }
        v2_validation = self._build_v2_validation(
            df=df,
            timestamps_ref=list(timestamps_ref),
            filtered_scenarios=filtered_scenarios,
            delta_nonneg=delta,
        )
        logger.info(
            "S(V1) capacity orientation: %s (PV–V1 corr=%.3f); delta stats mean=%.6f max=%.6f min=%.6f std=%.6f",
            orientation,
            corr,
            stats["mean"],
            stats["max"],
            stats["min"],
            stats["std"],
        )
        if v2_validation.get("available", False):
            logger.info(
                "V2 validation proxy=%s (corr=%.3f); variants=%s",
                v2_validation["proxy"].get("name"),
                v2_validation["proxy"].get("corr", float("nan")),
                v2_validation["variants_evaluated"],
            )
            proxy = v2_validation["proxy"]
            logger.info(
                "V2 proxy detail: name=%s, alpha=%.6f, beta=%.6f, corr=%.3f",
                proxy["name"],
                proxy["alpha"],
                proxy["beta"],
                proxy["corr"],
            )
            logger.info(
                "PV_raw means → Min: %.4f, Max: %.4f",
                v2_validation["pv_raw_means"]["min"],
                v2_validation["pv_raw_means"]["max"],
            )
            for key, val in v2_validation["validations"].items():
                logger.info(
                    "Variant %s: k̂=%.3f | E[Δ(V1−V2_k) | Min]=%.4f, E[Δ(V1−V2_k) | Max]=%.4f",
                    key,
                    val["k_hat_hist"],
                    val["mean_expected_delta_min"],
                    val["mean_expected_delta_max"],
                )
            mono = v2_validation["monotonicity_on_max"]
            logger.info("Monotonicity check on MaxSolar: %s → increasing=%s", mono["order"], mono["is_monotonic_increasing"])
            logger.info("corr(Δ_V1, PV_raw(Max)) = %.3f", v2_validation["corr_deltaV1_vs_pvraw_max"])
        else:
            logger.warning("V2 validation unavailable: %s", v2_validation.get("reason", "unknown"))

        out = {
            "trained_target": trained_target,
            "timestamps": list(timestamps_ref),
            "S_min": preds_min.tolist(),
            "S_max": preds_max.tolist(),
            "Delta": delta.tolist(),
            "delta_stats": stats,
            "delta_orientation": orientation,
            "pv_v1_corr": corr,
            "validation_v2": v2_validation,
        }
        return out

    def save_results(self, all_results: Dict, save_path: str):
        def conv(obj):
            if hasattr(obj, "strftime"):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, list):
                return [conv(x) for x in obj]
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            return obj

        final_output = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "target_location": self.target_location,
                "proxy_location": self.proxy_location,
                "model_path": self.model_path,
            },
            "results": conv(all_results),
        }
        with open(save_path, "w") as f:
            json.dump(final_output, f, indent=4)
        logger.info("Final results saved to %s", save_path)

    def analyze_pv_scaling(self, df: pd.DataFrame):
        peak_pv = df["passive_pv_generation_mw"].max()
        mean_pv = df["passive_pv_generation_mw"].mean()
        logger.info("Peak PV: %.6f MW, Mean PV: %.6f MW", peak_pv, mean_pv)
        return {"expected_mean_delta_v1_minus_v2": mean_pv}


def pick_top_pv_sites(data_path: str, n: int = 10) -> List[str]:
    df = pd.read_parquet(data_path, columns=["location", "timestamp", "passive_pv_generation_mw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    mask = (df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([6, 7])) & (df["timestamp"].dt.hour.between(11, 14))
    dfp = df.loc[mask].copy()
    dfp["pvpos"] = dfp["passive_pv_generation_mw"] > 0
    counts = dfp.groupby("location")["pvpos"].sum().sort_values(ascending=False)
    return counts.head(n).index.tolist()


def _extract_sv1_ts_delta(results: dict) -> tuple[List[pd.Timestamp], np.ndarray]:
    if isinstance(results, dict) and "by_variant" in results:
        v1 = results["by_variant"].get("V1", {})
        delta = np.array(v1.get("Delta", []))
        ts = results.get("timestamps", [])
        return ts, delta
    if isinstance(results, dict) and "Delta" in results:
        delta = np.array(results["Delta"])
        ts = results.get("timestamps", [])
        return ts, delta
    if isinstance(results, dict) and "sv1" in results and isinstance(results["sv1"], dict):
        delta = np.array(results["sv1"].get("Delta", []))
        ts = results.get("timestamps", [])
        return ts, delta
    return [], np.array([])


def plot_sv1_delta_vs_actual(df_site, timestamps, delta, site_name, out_dir):
    if len(timestamps) == 0 or len(delta) == 0:
        logger.warning("[%s] Nothing to plot (empty timestamps or delta).", site_name)
        return
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    df_delta = pd.DataFrame({"timestamp": ts, "sv1_delta": delta})
    df_act = df_site[["timestamp", "active_power_mw"]].copy()
    merged = df_delta.merge(df_act, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    if merged.empty:
        logger.warning("[%s] No overlapping timestamps between delta and actual.", site_name)
        return
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(merged["timestamp"], merged["active_power_mw"], linewidth=2.0, color=OCF_COLORS["dark_gray"], label="Actual Active Power (MW)")
        ax.plot(merged["timestamp"], merged["sv1_delta"], linewidth=2.0, color=OCF_COLORS["primary_orange"], label="S(V1) - Estimated Capacity (MW)")
        ax.set_title(f"{site_name} — Active Power and Estimated Capacity", fontsize=16, fontweight="bold", color=OCF_COLORS["very_dark_gray"], pad=20)
        ax.set_xlabel("Timestamp (UTC)", fontsize=12, color=OCF_COLORS["dark_gray"])
        ax.set_ylabel("MW", fontsize=12, color=OCF_COLORS["dark_gray"])
        ax.grid(True, alpha=0.4, color=OCF_COLORS["medium_beige"], linewidth=0.8)
        legend = ax.legend(frameon=True, facecolor=OCF_COLORS["off_white"], edgecolor=OCF_COLORS["medium_beige"], fontsize=11)
        for spine in ax.spines.values():
            spine.set_color(OCF_COLORS["medium_beige"])
            spine.set_linewidth(1.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=72))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        out_path = os.path.join(out_dir, f"{site_name}_sv1_delta_vs_actual.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS["off_white"], edgecolor="none")
        plt.close(fig)
    logger.info("[%s] Plot saved → %s", site_name, out_path)


def make_multi_site_sv1_plots(
    data_path: str,
    model_path: str,
    metadata_path: str,
    sites: List[str] | None = None,
    n_sites: int = 10,
    out_dir: str = "sv1_plots",
    n_periods: int = 200,
):
    if not sites:
        sites = pick_top_pv_sites(data_path, n=n_sites)
        logger.info("Auto-selected top PV sites (%d): %s", len(sites), sites)
    else:
        logger.info("Using provided sites (%d): %s", len(sites), sites)
    metadata = torch.load(metadata_path, map_location="cpu")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    all_metrics = []
    for loc in sites:
        try:
            analyzer = SingleSiteScenarioAnalyzer(model_path, metadata_path, data_path, loc)
            analyzer.model = model
            analyzer.metadata = metadata
            df_site = analyzer.load_and_prepare_data()
            if df_site is None or df_site.empty:
                logger.warning("[%s] Skipping — no data.", loc)
                continue
            high_pv = analyzer.find_high_pv_periods(df_site, n_periods=n_periods)
            if high_pv.empty:
                logger.warning("[%s] Skipping — no high-PV periods.", loc)
                continue
            df_analysis, val_cutoff = analyzer.prepare_data_splits(df_site)
            results = analyzer.run_analysis(df_analysis, high_pv, val_cutoff)
            ts, delta = _extract_sv1_ts_delta(results)
            if len(ts) and len(delta):
                plot_sv1_delta_vs_actual(df_site, ts, delta, loc, out_dir)
            metrics = collect_site_k_consistency(loc, results, scenario="MaxSolar", k_list=(1.0, 0.5, 0.1))
            all_metrics.extend(metrics)
        except Exception as e:
            logger.exception("[%s] Failed to create plot: %s", loc, e)
    try:
        plot_k_consistency_bars(all_metrics, out_path=os.path.join(out_dir, "multi_site_k_consistency.png"))
    except Exception as e:
        logger.exception("Failed to plot multi-site k consistency: %s", e)


VARIANT_COLORS = {
    "v2": OCF_COLORS["very_dark_gray"],
    "v2_1p0": OCF_COLORS["dark_gray"],
    "v2_0p5": OCF_COLORS["secondary_orange"],
    "v2_0p1": OCF_COLORS["primary_orange"],
}


def plot_scenario_timeseries(timestamps, s_min, s_max, delta, site_name, out_dir="sv1_plots"):
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    order = np.argsort(ts.values)
    ts = ts.iloc[order]
    s_min = np.asarray(s_min)[order]
    s_max = np.asarray(s_max)[order]
    delta = np.asarray(delta)[order]
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.scatter(ts, s_min, s=12, alpha=0.35, label="S_min (MinSolar)", color=OCF_COLORS["secondary_orange"])
        ax.scatter(ts, s_max, s=12, alpha=0.5, label="S_max (MaxSolar)", color=OCF_COLORS["dark_gray"])
        ax.fill_between(ts, s_min, s_max, alpha=0.10, color=OCF_COLORS["medium_beige"], label="Range (Max–Min)")
        ax.plot(ts, delta, linewidth=2.6, label="Δ (non-negative)", color=OCF_COLORS["primary_orange"])
        ax.set_title(f"{site_name} — Scenario Predictions", fontsize=16, fontweight="bold", color=OCF_COLORS["very_dark_gray"], pad=20)
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.4, color=OCF_COLORS["medium_beige"], linewidth=0.8)
        ax.legend(frameon=True, facecolor=OCF_COLORS["off_white"], edgecolor=OCF_COLORS["medium_beige"])
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
        out_path = os.path.join(out_dir, f"{site_name}_scenario_timeseries.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS["off_white"], edgecolor="none")
        plt.close(fig)
    logger.info("[%s] Plot saved → %s", site_name, out_path)


def plot_v2_validation_timeseries(timestamps, delta_nonneg, v2_validation, site_name, out_dir="sv1_plots"):
    if not v2_validation or not v2_validation.get("available", False):
        logger.warning("[%s] V2 validation not available; skipping validation plots.", site_name)
        return
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    order = np.argsort(ts.values)
    ts = ts.iloc[order]
    delta_nonneg = np.asarray(delta_nonneg)[order]
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(ts, delta_nonneg, linewidth=2.6, color=OCF_COLORS["primary_orange"], label="Δ S(V1) (non-negative)")
        seqs = v2_validation.get("variant_delta_seqs", {})
        if seqs:
            for var, seq in seqs.items():
                y = np.asarray(seq)[order]
                m = np.isfinite(y)
                if m.sum() < 3:
                    continue
                ax.plot(ts[m], y[m], linewidth=1.6, label=f"{var} (hist Δ = V1−{var})", color=VARIANT_COLORS.get(var, OCF_COLORS["dark_gray"]))
        else:
            pv_raw_max = np.asarray(v2_validation.get("pv_raw_max_seq", []))
            if len(pv_raw_max) == len(ts):
                pv_raw_max = pv_raw_max[order]
                m = np.isfinite(pv_raw_max)
                for var, rec in v2_validation.get("validations", {}).items():
                    k = rec.get("k_hat_hist", np.nan)
                    if not np.isfinite(k) or k <= 0:
                        continue
                    ax.plot(ts[m], k * pv_raw_max[m], linewidth=1.8, label=f"{var} expected (k̂={k:.2f})", color=VARIANT_COLORS.get(var, OCF_COLORS["dark_gray"]))
        ax.set_title(f"{site_name} — Validation: per-timestamp Δ variants", fontsize=16, fontweight="bold", color=OCF_COLORS["very_dark_gray"], pad=20)
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.4, color=OCF_COLORS["medium_beige"], linewidth=0.8)
        ax.legend(frameon=True, facecolor=OCF_COLORS["off_white"], edgecolor=OCF_COLORS["medium_beige"])
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
        out_path = os.path.join(out_dir, f"{site_name}_validation_timeseries.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS["off_white"], edgecolor="none")
        plt.close(fig)
    logger.info("[%s] Plot saved → %s", site_name, out_path)


def plot_v2_validation_bars(v2_validation, site_name, out_dir="sv1_plots"):
    if not v2_validation or not v2_validation.get("available", False):
        return
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k in ["v2_0p1", "v2_0p5", "v2_1p0", "v2"] if k in v2_validation.get("validations", {})]
    if not keys:
        return
    max_means = [v2_validation["validations"][k]["mean_expected_delta_max"] for k in keys]
    min_means = [v2_validation["validations"][k]["mean_expected_delta_min"] for k in keys]
    colors = [VARIANT_COLORS.get(k, OCF_COLORS["medium_beige"]) for k in keys]
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(keys))
        bars = ax.bar(x, max_means, color=colors, width=0.6, label="Expected Δ under MaxSolar")
        ax.bar(x, min_means, color="none", edgecolor=OCF_COLORS["dark_gray"], width=0.6, hatch="///", label="Expected Δ under MinSolar")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=0)
        ax.set_ylabel("MW")
        ax.set_title(f"{site_name} — V2 Validation (Expected Δ by Variant)", fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, color=OCF_COLORS["medium_beige"])
        ax.legend(frameon=True, facecolor=OCF_COLORS["off_white"], edgecolor=OCF_COLORS["medium_beige"])
        for b, v in zip(bars, max_means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        out_path = os.path.join(out_dir, f"{site_name}_validation_bars.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS["off_white"], edgecolor="none")
        plt.close(fig)
    logger.info("[%s] Plot saved → %s", site_name, out_path)


def _robust_k_stats(y, x, eps=1e-6):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x) & (np.abs(x) > eps)
    if m.sum() < 3:
        return {"k_hat": np.nan, "iqr": np.nan, "n": int(m.sum()), "r2": np.nan, "mape_pct": np.nan}
    r = y[m] / x[m]
    k_hat = np.median(r)
    iqr = np.percentile(r, 75) - np.percentile(r, 25)
    x_m, y_m = x[m], y[m]
    k_ols = np.dot(x_m, y_m) / np.dot(x_m, x_m)
    y_hat = k_ols * x_m
    ss_res = np.sum((y_m - y_hat) ** 2)
    ss_tot = np.sum((y_m - np.mean(y_m)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mape = np.mean(np.abs((r - k_hat) / np.maximum(np.abs(k_hat), eps))) * 100.0
    return {"k_hat": float(k_hat), "iqr": float(iqr), "n": int(m.sum()), "r2": float(r2), "mape_pct": float(mape)}


def collect_site_k_consistency(site_name: str, results: dict, scenario: str = "MaxSolar", k_list=(1.0, 0.5, 0.1), eps=1e-6):
    ts = pd.to_datetime(pd.Series(results.get("timestamps", [])), utc=True)
    s_max = np.asarray(results.get("S_max", []), dtype=float)
    v2_val = results.get("validation_v2", {}) or {}
    pv_raw_max = np.asarray(v2_val.get("pv_raw_max_seq", []), dtype=float)
    L = min(len(ts), len(s_max), len(pv_raw_max))
    ts, s_max, pv_raw_max = ts[:L], s_max[:L], pv_raw_max[:L]
    out = []
    for k in k_list:
        s_v2_k = s_max - k * pv_raw_max
        delta_k = s_max - s_v2_k
        stats = _robust_k_stats(delta_k, pv_raw_max, eps=eps)
        out.append({"site": site_name, "k_target": float(k), **stats})
    return out


def plot_k_consistency_bars(all_metrics: list, out_path="sv1_plots/multi_site_k_consistency.png"):
    if not all_metrics:
        logger.warning("No metrics to plot for k consistency.")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(all_metrics)
    sites = sorted(df["site"].unique().tolist())
    ks = sorted(df["k_target"].unique().tolist())
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(sites)), 6))
        width = 0.18
        x = np.arange(len(sites))
        k_to_color = {
            1.0: VARIANT_COLORS.get("v2", OCF_COLORS["very_dark_gray"]),
            0.5: VARIANT_COLORS.get("v2_0p5", OCF_COLORS["secondary_orange"]),
            0.1: VARIANT_COLORS.get("v2_0p1", OCF_COLORS["primary_orange"]),
        }
        for i, k in enumerate(ks):
            d = df[df["k_target"] == k].set_index("site").reindex(sites)
            k_hat = d["k_hat"].values.astype(float)
            iqr = d["iqr"].values.astype(float)
            r2 = d["r2"].values.astype(float)
            offs = (i - (len(ks) - 1) / 2) * (width + 0.02)
            bars = ax.bar(
                x + offs,
                k_hat,
                yerr=iqr,
                width=width,
                label=f"k={k:g}",
                capsize=3,
                color=k_to_color.get(k, OCF_COLORS["dark_gray"]),
                edgecolor=OCF_COLORS["very_dark_gray"],
                linewidth=0.6,
            )
            for xi, bh, r2i in zip(x + offs, k_hat, r2):
                if np.isfinite(bh) and np.isfinite(r2i):
                    ax.text(xi, bh + 0.02, f"R²={r2i:.2f}", ha="center", va="bottom", fontsize=8)
        for k in ks:
            ax.axhline(k, linestyle="--", linewidth=1, color=k_to_color.get(k, OCF_COLORS["medium_beige"]), alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=30, ha="right")
        ax.set_ylabel("Estimated k̂ (median ratio)")
        ax.set_title("Scenario-consistent validation: k̂ vs target k across sites", fontsize=16, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, color=OCF_COLORS["medium_beige"])
        ax.legend(frameon=True, facecolor=OCF_COLORS["off_white"], edgecolor=OCF_COLORS["medium_beige"])
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS["off_white"], edgecolor="none")
        plt.close(fig)
    logger.info("Multi-site k consistency plot saved → %s", out_path)


def pv_raw_scenario(df_scen: pd.DataFrame, proxy_col: str, alpha: float, beta: float, roll_window: Optional[int] = None) -> pd.Series:
    pv = df_scen[proxy_col] * alpha + beta
    if roll_window and roll_window > 1:
        pv = pv.rolling(roll_window, min_periods=1).mean()
    return pv.clip(lower=0.0)


def k_hat_from_existing_variants(
    df_scen: pd.DataFrame,
    sv1_col: str,
    variant_cols: Sequence[str],
    proxy_col: str,
    alpha: float,
    beta: float,
    roll_window: Optional[int] = None,
    pv_min_mw: float = 0.01,
    enforce_orientation: Optional[str] = None,
) -> pd.DataFrame:
    sv1 = df_scen[sv1_col].copy()
    if enforce_orientation == "max_minus_min" and (sv1.median() < 0):
        sv1 = -sv1
    elif enforce_orientation == "min_minus_max" and (sv1.median() > 0):
        sv1 = -sv1
    pv_s = pv_raw_scenario(df_scen, proxy_col, alpha, beta, roll_window)
    mask = pv_s > pv_min_mw
    out = []
    for col in variant_cols:
        if col not in df_scen.columns:
            out.append({"variant": col, "k_hat_median": np.nan, "k_hat_mean": np.nan, "n_used": 0})
            continue
        delta = sv1 - df_scen[col]
        ratio = np.where(mask, delta / pv_s, np.nan)
        ratio = pd.Series(ratio, index=df_scen.index).dropna()
        if len(ratio) == 0:
            out.append({"variant": col, "k_hat_median": np.nan, "k_hat_mean": np.nan, "n_used": 0})
        else:
            out.append(
                {"variant": col, "k_hat_median": float(np.median(ratio)), "k_hat_mean": float(np.mean(ratio)), "n_used": int(ratio.size)}
            )
    return pd.DataFrame(out)


def collect_multi_site_khats(
    site_to_df_scen: dict,
    sv1_col: str,
    variant_cols: Sequence[str],
    site_proxy_params: dict,
    pv_min_mw: float = 0.01,
) -> pd.DataFrame:
    rows = []
    for site, df_s in site_to_df_scen.items():
        params = site_proxy_params[site]
        kh = k_hat_from_existing_variants(
            df_scen=df_s,
            sv1_col=sv1_col,
            variant_cols=variant_cols,
            proxy_col=params["proxy_col"],
            alpha=params["alpha"],
            beta=params["beta"],
            roll_window=params.get("roll_window"),
            pv_min_mw=pv_min_mw,
            enforce_orientation=None,
        )
        kh["site"] = site
        rows.append(kh)
    return pd.concat(rows, ignore_index=True)


def plot_khats_bars(khats: pd.DataFrame, out_png: str):
    want = khats[khats["variant"].isin(["v2_1p0", "v2_0p5", "v2_0p1"])].copy()
    variants_order = ["v2_1p0", "v2_0p5", "v2_0p1"]
    want["variant"] = pd.Categorical(want["variant"], variants_order)
    sites = want["site"].unique()
    x = np.arange(len(sites))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(sites) * 0.8), 5))
    for i, v in enumerate(variants_order):
        vals = [want[(want.site == s) & (want.variant == v)]["k_hat_median"].mean() for s in sites]
        ax.bar(x + i * width - width, vals, width, label=v)
    ax.axhline(1.0, linestyle="--")
    ax.axhline(0.5, linestyle="--")
    ax.axhline(0.1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha="right")
    ax.set_ylabel("k̂ (median over timestamps)")
    ax.set_title("Validation: implied k̂ vs existing v2 variants (scenario-aligned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def main():
    TARGET_LOCATION = os.environ.get("TARGET_LOCATION", "peterborough_central_11kv")
    STAGE2_DATED = "/home/felix/output/all_locations_tft_ready_with_passive_pv_2025-04-02.parquet"
    STAGE2_LATEST = "/home/felix/output/all_locations_tft_ready_with_passive_pv.parquet"
    DATA_PATH = STAGE2_DATED if os.path.exists(STAGE2_DATED) else STAGE2_LATEST
    MODEL_PATH = "production_tft_model.ckpt"
    METADATA_PATH = "production_tft_metadata.pth"
    OUTPUT_PATH = f"{TARGET_LOCATION}_scenario_results_final.json"
    logger.info("Using data file: %s", DATA_PATH)
    logger.info("Starting FINAL Scenario Analysis for site: %s", TARGET_LOCATION)
    logger.info("=" * 60)
    analyzer = SingleSiteScenarioAnalyzer(MODEL_PATH, METADATA_PATH, DATA_PATH, TARGET_LOCATION)
    df_site = analyzer.load_and_prepare_data()
    if df_site is not None and not df_site.empty:
        high_pv_periods = analyzer.find_high_pv_periods(df_site, n_periods=500)
        if not high_pv_periods.empty:
            analyzer.load_model_and_setup()
            df_analysis, val_cutoff = analyzer.prepare_data_splits(df_site)
            results = analyzer.run_analysis(df_analysis, high_pv_periods, val_cutoff)
            if results:
                analyzer.save_results(results, OUTPUT_PATH)
                ts = results.get("timestamps", [])
                s_min = results.get("S_min", [])
                s_max = results.get("S_max", [])
                delta = results.get("Delta", [])
                v2_val = results.get("validation_v2", {})
                plot_scenario_timeseries(ts, s_min, s_max, delta, TARGET_LOCATION, out_dir="sv1_plots")
                plot_v2_validation_timeseries(ts, delta, v2_val, TARGET_LOCATION, out_dir="sv1_plots")
                plot_v2_validation_bars(v2_val, TARGET_LOCATION, out_dir="sv1_plots")
                analyzer.analyze_pv_scaling(df_site)
                logger.info("Single-site analysis complete. Now generating plots for 10 sites…")
                make_multi_site_sv1_plots(
                    data_path=DATA_PATH,
                    model_path=MODEL_PATH,
                    metadata_path=METADATA_PATH,
                    sites=None,
                    n_sites=10,
                    out_dir="sv1_plots",
                    n_periods=200,
                )
    logger.info("=" * 60)
    logger.info("All done.")


if __name__ == "__main__":
    main()
