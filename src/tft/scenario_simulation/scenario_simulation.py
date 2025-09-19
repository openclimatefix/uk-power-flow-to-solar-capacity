import os
import json
import logging
import warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from datetime import datetime
import hashlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml") as f:
    config = yaml.safe_load(f)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAINED_SITE_FALLBACK = [
    "belchamp_grid_11kv",
    "george_hill_primary_11kv",
    "manor_way_primary_11kv",
    "st_stephens_primary_11kv",
    "swaffham_grid_11kv",
]

OCF_COLORS = {
    'primary_orange': '#FF4901',
    'secondary_orange': '#FF8F73',
    'dark_gray': '#292B2B',
    'very_dark_gray': '#0C0D0D',
    'white': '#FFFFFF',
    'off_white': '#FFFBF5',
    'light_beige': '#F0ECE8',
    'medium_beige': '#D9D0CA'
}

OCF_PALETTE = [
    OCF_COLORS['primary_orange'],
    OCF_COLORS['secondary_orange'],
    OCF_COLORS['dark_gray'],
    OCF_COLORS['very_dark_gray']
]

def setup_ocf_plot_style():
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OCF_PALETTE)
    plt.rcParams['figure.facecolor'] = OCF_COLORS['off_white']
    plt.rcParams['axes.facecolor'] = OCF_COLORS['white']
    plt.rcParams['text.color'] = OCF_COLORS['very_dark_gray']
    plt.rcParams['axes.labelcolor'] = OCF_COLORS['dark_gray']
    plt.rcParams['xtick.color'] = OCF_COLORS['dark_gray']
    plt.rcParams['ytick.color'] = OCF_COLORS['dark_gray']
    plt.rcParams['grid.color'] = OCF_COLORS['medium_beige']
    plt.rcParams['grid.alpha'] = 0.6

def ensure_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location", "timestamp"]).copy()
    if "time_idx" not in df.columns or df["time_idx"].isna().any():
        df["time_idx"] = df.groupby("location")["timestamp"].transform(lambda s: pd.factorize(s.values)[0].astype(np.int64))
    return df

def _infer_pv_orientation(df: pd.DataFrame) -> tuple[int, float]:
    d = df.loc[
        (df["timestamp"].dt.month.isin([6, 7])) &
        (df["timestamp"].dt.hour.between(11, 14)),
        ["passive_pv_generation_mw", "active_power_mw"]
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

    def load_and_prepare_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from: {self.data_path}")
        df_full = pd.read_parquet(self.data_path)
        logger.info(f"Filtering for target location: {self.target_location}")
        df_site = df_full[df_full['location'] == self.target_location].copy()
        if df_site.empty:
            logger.error(f"Location '{self.target_location}' not found.")
            return None
        df_site['timestamp'] = pd.to_datetime(df_site['timestamp'], utc=True)
        df_site = ensure_time_idx(df_site)
        pv_check = df_site[
            (df_site['timestamp'].dt.year == 2024) &
            (df_site['timestamp'].dt.month.isin([6, 7])) &
            (df_site['timestamp'].dt.hour.between(11, 14)) &
            (df_site['passive_pv_generation_mw'] > 0)
        ]
        if pv_check.empty:
            logger.error(f"No PV>0 found for {self.target_location} in 2024-06/07 11–14 UTC.")
            return None
        logger.info(f"Data prepared for {self.target_location} with {len(df_site):,} records.")
        return df_site

    def find_high_pv_periods(self, df: pd.DataFrame, n_periods: int = 200) -> pd.DataFrame:
        logger.info(f"Finding top {n_periods} PEAK SOLAR periods...")
        df_summer = df[(df['timestamp'].dt.year == 2024) & (df['timestamp'].dt.month.isin([6, 7]))].copy()
        df_peak_hours = df_summer[(df_summer['timestamp'].dt.hour >= 11) & (df_summer['timestamp'].dt.hour <= 14)].copy()
        if 'tcc' in df_peak_hours.columns:
            df_peak_hours = df_peak_hours.sort_values('tcc')
        high_pv_periods = df_peak_hours.nlargest(n_periods, 'passive_pv_generation_mw')
        logger.info(f"Identified {len(high_pv_periods)} PEAK SOLAR periods for analysis.")
        return high_pv_periods

    def load_model_and_setup(self):
        logger.info("Loading TFT model and metadata...")
        self.metadata = torch.load(self.metadata_path, map_location='cpu')
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path, map_location='cpu')
        self.model.eval()
        logger.info("Model and metadata loaded successfully.")
        logger.info(f"Model trained target: '{self.get_model_target_name()}'")

    def prepare_data_splits(self, df: pd.DataFrame):
        val_cutoff_date = pd.Timestamp('2024-08-01', tz='UTC')
        val_cutoff = df[df['timestamp'] < val_cutoff_date]['time_idx'].max()
        logger.info(f"Data splits defined with validation cutoff at time_idx: {val_cutoff}")
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
                logger.warning(f"Could not extract trained locations from ds_params: {e}")
        if not trained_locs:
            trained_locs = TRAINED_SITE_FALLBACK[:]
            logger.info(f"Using fallback trained sites: {trained_locs}")
        self.trained_locations = trained_locs
        if self.target_location not in self.trained_locations:
            self.proxy_location = self.choose_proxy_location(self.target_location, self.trained_locations)
            logger.info(f"Target '{self.target_location}' not in training set. Using proxy '{self.proxy_location}'.")
            train_df.loc[:, "location"] = self.proxy_location
        else:
            self.proxy_location = self.target_location
            logger.info(f"Target '{self.target_location}' is a trained location (no proxy needed).")
        if ds_params:
            logger.info("Reconstructing dataset from model's saved dataset parameters.")
            ds_params = dict(ds_params)
            ds_params["target"] = self.get_model_target_name()
            self.training_dataset = TimeSeriesDataSet(train_df, **ds_params)
        else:
            logger.warning("Falling back to config-based dataset construction.")
            model_config = {**config['model'], 'target': self.get_model_target_name()}
            self.training_dataset = TimeSeriesDataSet(
                train_df,
                time_idx=model_config['time_idx'],
                target=model_config['target'],
                group_ids=model_config['group_ids'],
                max_encoder_length=model_config['max_encoder_length'],
                max_prediction_length=model_config['max_prediction_length'],
                static_categoricals=model_config['static_categoricals'],
                static_reals=model_config['static_reals'],
                time_varying_known_reals=model_config['time_varying_known_reals'],
                time_varying_unknown_reals=model_config['time_varying_unknown_reals'],
                add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
            )
        self._init_used_feature_lists(ds_params)
        logger.info(f"Trained locations: {self.trained_locations}")
        logger.info(f"Proxy in use: {self.proxy_location}")

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
            logger.warning(f"Scenario features NOT used by model (ignored): {mods_dropped}")
        return mods_kept

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
            "ssrd_w_m2", "ssrd_w_m2_roll_mean_3h", "ssr_w_m2",
            "irradiance_x_clear_sky", "solar_intensity_factor",
            "tcc", "solar_elevation_angle", "solar_zenith_angle",
            "ghi", "dhi", "dni", "cs_ghi"
        ]
        cols = [c for c in candidate_names if (c in df.columns) and (c in used)]
        if not cols:
            logger.warning("No usable solar/cloud features found in df ∩ used_features; scenarios will be empty.")
            return {"MinSolar": {}, "MaxSolar": {}}
        mask = (
            (df["timestamp"].dt.month.isin([6, 7])) &
            (df["timestamp"].dt.hour.between(11, 14))
        )
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
        return {"MinSolar": min_mods, "MaxSolar": max_mods}

    def prepare_single_prediction_input(self, encoder_data: pd.DataFrame, target_time_idx: int, modifications: Dict[str, Any]) -> Dict:
        temp_data = encoder_data.copy()
        target_row = temp_data.iloc[-1:].copy()
        target_row['time_idx'] = target_time_idx
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
        test_time_indices = high_pv_periods['time_idx'].unique()
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
        logger.info(f"Scenario feature counts: Min={len(filtered_scenarios.get('MinSolar', {}))}, Max={len(filtered_scenarios.get('MaxSolar', {}))}")
        timestamps_ref: List[pd.Timestamp] = []
        preds_min: List[float] = []
        preds_max: List[float] = []
        for scenario_name in ("MinSolar", "MaxSolar"):
            mods = filtered_scenarios.get(scenario_name, {})
            logger.info(f"  Scenario: {scenario_name} (applying {len(mods)} features)")
            cur_preds: List[float] = []
            for time_idx in test_time_indices:
                enc = df[(df['time_idx'] >= time_idx - self.encoder_length) & (df['time_idx'] < time_idx)].copy()
                if len(enc) < self.encoder_length:
                    continue
                if self.proxy_location:
                    enc.loc[:, 'location'] = self.proxy_location
                for f, v in mods.items():
                    if f in enc.columns:
                        enc.loc[:, f] = v
                if scenario_name == "MinSolar":
                    row_ts = df.loc[df['time_idx'] == time_idx, 'timestamp']
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
        timestamps_ref = timestamps_ref[:L]
        if L == 0:
            logger.error("No scenario predictions produced.")
            return {}
        sep = float(np.median(np.abs(preds_max - preds_min)))
        logger.info(f"Scenario separation (median |Max-Min|) = {sep:.6f}")
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
        logger.info(f"  S(V1) capacity orientation: {orientation} (PV–V1 corr={corr:.3f}); delta stats  mean={stats['mean']:.6f} max={stats['max']:.6f} min={stats['min']:.6f} std={stats['std']:.6f}")
        out = {
            "trained_target": trained_target,
            "timestamps": timestamps_ref,
            "S_min": preds_min.tolist(),
            "S_max": preds_max.tolist(),
            "Delta": delta.tolist(),
            "delta_stats": stats,
            "delta_orientation": orientation,
            "pv_v1_corr": corr,
        }
        return out

    def save_results(self, all_results: Dict, save_path: str):
        def conv(obj):
            if hasattr(obj, 'strftime'):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
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
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'target_location': self.target_location,
                'proxy_location': self.proxy_location,
                'model_path': self.model_path
            },
            'results': conv(all_results)
        }
        with open(save_path, 'w') as f:
            json.dump(final_output, f, indent=4)
        logger.info(f"Final results saved to {save_path}")

    def analyze_pv_scaling(self, df: pd.DataFrame):
        peak_pv = df['passive_pv_generation_mw'].max()
        mean_pv = df['passive_pv_generation_mw'].mean()
        logger.info(f"Peak PV: {peak_pv:.6f} MW, Mean PV: {mean_pv:.6f} MW")
        return {'expected_mean_delta_v1_minus_v2': mean_pv}

def pick_top_pv_sites(data_path: str, n: int = 10) -> list[str]:
    df = pd.read_parquet(data_path, columns=["location", "timestamp", "passive_pv_generation_mw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    mask = (
        (df["timestamp"].dt.year == 2024)
        & (df["timestamp"].dt.month.isin([6, 7]))
        & (df["timestamp"].dt.hour.between(11, 14))
    )
    dfp = df.loc[mask].copy()
    dfp["pvpos"] = dfp["passive_pv_generation_mw"] > 0
    counts = dfp.groupby("location")["pvpos"].sum().sort_values(ascending=False)
    return counts.head(n).index.tolist()

def _extract_sv1_ts_delta(results: dict) -> tuple[list[pd.Timestamp], np.ndarray]:
    if isinstance(results, dict) and "Delta" in results:
        delta = np.array(results["Delta"])
        ts = results.get("timestamps", [])
        return ts, delta
    return [], np.array([])

def plot_sv1_delta_vs_actual(df_site, timestamps, delta, site_name, out_dir):
    if len(timestamps) == 0 or len(delta) == 0:
        logger.warning(f"[{site_name}] Nothing to plot (empty timestamps or delta).")
        return
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    df_delta = pd.DataFrame({"timestamp": ts, "sv1_delta": delta})
    df_act = df_site[["timestamp", "active_power_mw"]].copy()
    merged = df_delta.merge(df_act, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    if merged.empty:
        logger.warning(f"[{site_name}] No overlapping timestamps between delta and actual.")
        return
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(merged["timestamp"], merged["active_power_mw"], linewidth=2.0, color=OCF_COLORS['dark_gray'], label="Actual Active Power (MW)")
        ax.plot(merged["timestamp"], merged["sv1_delta"], linewidth=2.0, color=OCF_COLORS['primary_orange'], label="S(V1) - Estimated Capacity (MW)")
        ax.set_title(f"{site_name} — Active Power and Estimated Capacity", fontsize=16, fontweight="bold", color=OCF_COLORS['very_dark_gray'], pad=20)
        ax.set_xlabel("Timestamp (UTC)", fontsize=12, color=OCF_COLORS['dark_gray'])
        ax.set_ylabel("MW", fontsize=12, color=OCF_COLORS['dark_gray'])
        ax.grid(True, alpha=0.4, color=OCF_COLORS['medium_beige'], linewidth=0.8)
        ax.legend(frameon=True, facecolor=OCF_COLORS['off_white'], edgecolor=OCF_COLORS['medium_beige'], fontsize=11)
        for spine in ax.spines.values():
            spine.set_color(OCF_COLORS['medium_beige'])
            spine.set_linewidth(1.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=72))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        out_path = os.path.join(out_dir, f"{site_name}_sv1_delta_vs_actual.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS['off_white'], edgecolor='none')
        plt.close(fig)
    logger.info(f"[{site_name}] Plot saved → {out_path}")

def plot_scenario_timeseries(timestamps, s_min, s_max, delta, site_name, out_dir="sv1_plots"):
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    with plt.rc_context():
        setup_ocf_plot_style()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(ts, s_min, label="S_min (MinSolar)", linewidth=2.0, color=OCF_COLORS['secondary_orange'])
        ax.plot(ts, s_max, label="S_max (MaxSolar)", linewidth=2.0, color=OCF_COLORS['dark_gray'])
        ax.fill_between(ts, s_min, s_max, alpha=0.15, color=OCF_COLORS['medium_beige'], label="Range (Max–Min)")
        ax.plot(ts, delta, label="Δ (non-negative)", linewidth=2.4, color=OCF_COLORS['primary_orange'])
        ax.set_title(f"{site_name} — Scenario Predictions", fontsize=16, fontweight="bold", color=OCF_COLORS['very_dark_gray'], pad=20)
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.4, color=OCF_COLORS['medium_beige'], linewidth=0.8)
        ax.legend(frameon=True, facecolor=OCF_COLORS['off_white'], edgecolor=OCF_COLORS['medium_beige'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=72))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        out_path = os.path.join(out_dir, f"{site_name}_scenario_timeseries.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=OCF_COLORS['off_white'], edgecolor='none')
        plt.close(fig)
    logger.info(f"[{site_name}] Plot saved → {out_path}")

def make_multi_site_sv1_plots(data_path: str, model_path: str, metadata_path: str, sites: list[str] | None = None, n_sites: int = 10, out_dir: str = "sv1_plots", n_periods: int = 200):
    if not sites:
        sites = pick_top_pv_sites(data_path, n=n_sites)
        logger.info(f"Auto-selected top PV sites ({len(sites)}): {sites}")
    else:
        logger.info(f"Using provided sites ({len(sites)}): {sites}")
    metadata = torch.load(metadata_path, map_location="cpu")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    for loc in sites:
        try:
            analyzer = SingleSiteScenarioAnalyzer(model_path, metadata_path, data_path, loc)
            analyzer.model = model
            analyzer.metadata = metadata
            df_site = analyzer.load_and_prepare_data()
            if df_site is None or df_site.empty:
                logger.warning(f"[{loc}] Skipping — no data.")
                continue
            high_pv = analyzer.find_high_pv_periods(df_site, n_periods=n_periods)
            if high_pv.empty:
                logger.warning(f"[{loc}] Skipping — no high-PV periods.")
                continue
            df_analysis, val_cutoff = analyzer.prepare_data_splits(df_site)
            results = analyzer.run_analysis(df_analysis, high_pv, val_cutoff)
            ts, delta = _extract_sv1_ts_delta(results)
            if len(ts) == 0 or len(delta) == 0:
                logger.warning(f"[{loc}] Skipping — no S(V1) delta returned.")
                continue
            plot_sv1_delta_vs_actual(df_site, ts, delta, loc, out_dir)
        except Exception as e:
            logger.exception(f"[{loc}] Failed to create plot: {e}")

def main():
    TARGET_LOCATION = os.environ.get("TARGET_LOCATION", "peterborough_central_11kv")
    STAGE2_DATED = "/home/felix/output/all_locations_tft_ready_with_passive_pv_2025-04-02.parquet"
    STAGE2_LATEST = "/home/felix/output/all_locations_tft_ready_with_passive_pv.parquet"
    DATA_PATH = STAGE2_DATED if os.path.exists(STAGE2_DATED) else STAGE2_LATEST
    MODEL_PATH = "production_tft_model.ckpt"
    METADATA_PATH = "production_tft_metadata.pth"
    OUTPUT_PATH = f"{TARGET_LOCATION}_scenario_results_final.json"
    logger.info(f"Using data file: {DATA_PATH}")
    logger.info(f"Starting FINAL Scenario Analysis for site: {TARGET_LOCATION}")
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
            plot_scenario_timeseries(ts, s_min, s_max, delta, TARGET_LOCATION, out_dir="sv1_plots")
        analyzer.analyze_pv_scaling(df_site)
    logger.info("Single-site analysis complete. Now generating plots for 10 sites…")
    make_multi_site_sv1_plots(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        sites=None,
        n_sites=10,
        out_dir="sv1_plots",
        n_periods=200
    )
    logger.info("=" * 60)
    logger.info("All done.")

if __name__ == "__main__":
    main()
