#!/usr/bin/env python3

import gc
import os
import json
import logging
import warnings
import yaml
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import lightning as L
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import matplotlib.pyplot as plt
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

with open("uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml") as f:
    config = yaml.safe_load(f)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _downcast_numeric_inplace(df: pd.DataFrame) -> None:
    try:
        float_cols = df.select_dtypes(include=["float64"]).columns
        int_cols = df.select_dtypes(include=["int64"]).columns
        for c in float_cols:
            df[c] = pd.to_numeric(df[c], downcast="float")
        for c in int_cols:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        if "location" in df.columns and df["location"].dtype != "category":
            df["location"] = df["location"].astype("category")
    except Exception:
        pass


class TFTScenarioAllLocationsAnalyzer:
    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.training_dataset = None
        self.encoder_length = 168
        self.trained_locations = {
            'belchamp_grid_11kv': 0,
            'george_hill_primary_11kv': 1,
            'manor_way_primary_11kv': 2,
            'st_stephens_primary_11kv': 3,
            'swaffham_grid_11kv': 4,
        }
        self.location_mapping = {v: k for k, v in self.trained_locations.items()}
        self.stream_dir = "scenario_stream"
        os.makedirs(self.stream_dir, exist_ok=True)
        self.train_cutoff = None
        self.val_cutoff = None
        self.trained_df_ref = None
        self.user_top_features: List[str] = None

    def load_model_and_setup(self):
        logger.info("Loading TFT model and metadata...")
        self.metadata = torch.load(self.metadata_path, map_location='cpu')
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            self.model_path, map_location='cpu'
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def assign_proxy_location(self, location_name: str) -> int:
        name_lower = str(location_name).lower()
        if any(r in name_lower for r in ['essex', 'chelmsford', 'colchester', 'harlow', 'brentwood']):
            return 1
        if any(r in name_lower for r in ['norfolk', 'norwich', 'swaffham', 'thetford']):
            return 4
        if any(r in name_lower for r in ['suffolk', 'bury', 'ipswich', 'sudbury']):
            return 0
        if any(r in name_lower for r in ['cambridge', 'huntingdon', 'peterborough', 'ely']):
            return 3
        if any(r in name_lower for r in ['kent', 'tunbridge', 'maidstone']):
            return 2
        first_letter = name_lower[:1]
        if first_letter <= 'e':
            return 0
        if first_letter <= 'j':
            return 1
        if first_letter <= 'n':
            return 2
        if first_letter <= 'r':
            return 3
        return 4

    def prepare_all_locations_data(self, df_full: pd.DataFrame):
        logger.info("Preparing data for ALL locations scenario analysis...")
        all_locations = df_full['location'].unique()
        untrained_locations = [loc for loc in all_locations if loc not in self.trained_locations]
        logger.info(f"Found {len(all_locations)} total locations")
        logger.info(f"Using {len(self.trained_locations)} trained locations for model reference")
        logger.info(f"Will analyze {len(untrained_locations)} untrained locations via zero-shot")
        trained_df = df_full[df_full['location'].isin(self.trained_locations)].copy()
        scenario_df = df_full
        max_time_idx = trained_df['time_idx'].max()
        train_cutoff = int(max_time_idx * config["train_split"])
        val_cutoff = int(max_time_idx * config["val_split"])
        logger.info("Data splits for training dataset:")
        logger.info(f"  Train: 0 to {train_cutoff}")
        logger.info(f"  Validation: {train_cutoff} to {val_cutoff}")
        logger.info(f"  Test: {val_cutoff} to {max_time_idx}")
        self.train_cutoff = train_cutoff
        self.val_cutoff = val_cutoff
        self.trained_df_ref = trained_df
        return list(self.trained_locations.keys()), untrained_locations, trained_df, scenario_df, val_cutoff

    def create_training_dataset(self, trained_df: pd.DataFrame, val_cutoff: int):
        model_config = config['model']
        self.training_dataset = TimeSeriesDataSet(
            trained_df[trained_df["time_idx"] <= val_cutoff],
            time_idx=model_config['time_idx'],
            target=model_config['target'],
            group_ids=model_config['group_ids'],
            max_encoder_length=model_config['max_encoder_length'],
            max_prediction_length=model_config['max_prediction_length'],
            static_categoricals=model_config['static_categoricals'],
            static_reals=model_config['static_reals'],
            time_varying_known_reals=model_config['time_varying_known_reals'],
            time_varying_unknown_reals=model_config['time_varying_unknown_reals'],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        logger.info(f"Training dataset created with {len(self.training_dataset)} samples")

    def _make_small_val_loader(self, max_batches: int = 12):
        if self.training_dataset is None or self.trained_df_ref is None:
            return None
        val_df = self.trained_df_ref[
            (self.trained_df_ref["time_idx"] >= self.train_cutoff) &
            (self.trained_df_ref["time_idx"] < self.val_cutoff)
        ].copy()
        if len(val_df) == 0:
            return None
        val_ds = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            data=val_df,
            predict=False,
            stop_randomization=True,
        )
        dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        def limited():
            for i, batch in enumerate(dl):
                if i >= max_batches:
                    break
                yield batch
        return limited()

    def compute_tft_variable_importance(self, top_k: int = 6) -> List[str]:
        try:
            small_val_loader = self._make_small_val_loader()
            if small_val_loader is None:
                raise RuntimeError("No validation loader for interpretation")
            importances: Dict[str, float] = {}
            for batch in small_val_loader:
                with torch.no_grad():
                    raw = self.model.forward(batch[0], return_loss=False, decode=True)
                try:
                    interp = self.model.interpret_output(raw, reduction="sum")
                except Exception:
                    continue
                for key in ("encoder_variables", "decoder_variables"):
                    var_imp = interp.get(key, {})
                    for v, s in var_imp.items():
                        importances[v] = importances.get(v, 0.0) + float(s)
            if not importances:
                raise RuntimeError("Empty importance map from interpretation")
            ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            top_feats = [n for n, _ in ranked if n not in ("time_idx", "encoder_length", "relative_time_idx")]
            logger.info(f"TFT top drivers (importance): {top_feats[:top_k]}")
            return top_feats[:top_k]
        except Exception:
            logger.warning("Variable importance failed; falling back to config vars.")
            model_cfg = config["model"]
            cand = model_cfg.get("time_varying_known_reals", []) + model_cfg.get("time_varying_unknown_reals", [])
            blacklist = {"time_idx", "encoder_length", "relative_time_idx", model_cfg.get("target", "")}
            cand = [c for c in cand if c not in blacklist]
            return cand[:top_k]

    def set_external_feature_importances(self, raw_text: str, available_columns: List[str], top_k: int = 10):
        items = []
        for line in str(raw_text).splitlines():
            if "-" in line:
                name, score = line.split("-", 1)
                name = name.strip()
                try:
                    s = float(score.strip().split()[0])
                except Exception:
                    s = 0.0
                items.append((name, s))
        alias_map = {
            "Solar Zenith Angle": ["solar_zenith_angle"],
            "Solar Intensity Factor": ["solar_intensity_factor", "solar_intensity"],
            "Weekend Indicator": ["is_weekend", "weekend_indicator"],
            "Instantaneous Surface Heat Flux (W/m²)": ["sshf_w_m2", "surface_heat_flux_w_m2"],
            "Hour Sine Encoding": ["hour_sin", "hour_sine"],
            "Wind Speed": ["wind_speed", "wind_speed_ms", "wind_speed_m_s"],
            "Weather Index": ["weather_index"],
            "Solar Radiation 3-Hour Rolling Mean (W/m²)": [
                "ssr_w_m2_roll_mean_3h", "solar_radiation_roll_mean_3h", "ssrd_w_m2_roll_mean_3h"
            ],
            "Year": ["year"],
            "Civil Twilight Indicator": ["is_civil_twilight", "civil_twilight_indicator"],
            "Wind Speed 3-Hour Rolling Mean": ["wind_speed_roll_mean_3h", "wind_speed_ms_roll_mean_3h"],
            "Solar Azimuth Angle": ["solar_azimuth_angle"],
        }
        cols_lower = {c.lower(): c for c in available_columns}
        picked: List[str] = []
        seen = set()

        def _try_pick(names: List[str]) -> bool:
            for cand in names:
                if cand in available_columns and cand not in seen:
                    picked.append(cand); seen.add(cand); return True
                cl = cand.lower()
                if cl in cols_lower and cols_lower[cl] not in seen:
                    picked.append(cols_lower[cl]); seen.add(cols_lower[cl]); return True
            return False

        for name, _ in items:
            if name in alias_map and _try_pick(alias_map[name]):
                continue
            clean = name.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            if clean in cols_lower and cols_lower[clean] not in seen:
                picked.append(cols_lower[clean]); seen.add(cols_lower[clean]); continue
            tokens = [t for t in name.lower().split() if len(t) > 2]
            best = None; best_hits = 0
            for c in available_columns:
                cl = c.lower()
                hits = sum(1 for t in tokens if t in cl)
                if hits > best_hits:
                    best_hits = hits; best = c
            if best and best not in seen and best_hits >= 2:
                picked.append(best); seen.add(best)

        if not picked:
            model_cfg = config["model"]
            fallback = model_cfg.get("time_varying_known_reals", []) + model_cfg.get("time_varying_unknown_reals", [])
            picked = [c for c in fallback if c not in {"time_idx", "relative_time_idx"}][:top_k]

        self.user_top_features = picked[:top_k]
        logger.info(f"Using externally provided top features (mapped): {self.user_top_features}")

    def _get_top_features_for_scenarios(self, top_k: int = 6) -> List[str]:
        if self.user_top_features:
            return self.user_top_features[:top_k]
        return self.compute_tft_variable_importance(top_k=top_k)

    @staticmethod
    def _ensure_time_parts_inplace(df: pd.DataFrame) -> None:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            if "month" not in df.columns:
                df.loc[:, "month"] = ts.dt.month
            if "hour" not in df.columns:
                df.loc[:, "hour"] = ts.dt.hour

    def define_scenarios(self, df_reference: pd.DataFrame = None) -> Dict[str, Dict]:
        donor_scope = "trained_only"
        if df_reference is None:
            df_reference = self.trained_df_ref
        top_features = self._get_top_features_for_scenarios(top_k=6)
        scenarios = {
            "MaxSolarAnalog": {
                "type": "analog", "quantile": 0.95,
                "top_features": top_features, "donor_scope": donor_scope,
            },
            "MinSolarAnalog": {
                "type": "analog", "quantile": 0.05,
                "top_features": top_features, "donor_scope": donor_scope,
            },
        }
        logger.info(f"Defined analogy scenarios using top features: {top_features}")
        return scenarios

    def apply_scenario_to_encoder_data(self, encoder_data: pd.DataFrame,
                                       scenario_name: str, modifications: Dict) -> pd.DataFrame:
        if modifications is None:
            return encoder_data.copy()
        df_mod = encoder_data.copy()
        if modifications.get("type") != "analog":
            if 'timestamp' in df_mod.columns:
                df_mod['timestamp'] = pd.to_datetime(df_mod['timestamp'])
                mask = df_mod['timestamp'].dt.hour.between(6, 18)
            else:
                mask = slice(None)
            for feature, new_value in modifications.items():
                if feature in df_mod.columns:
                    df_mod.loc[mask, feature] = new_value
                    base = feature.split('_')[0] if '_' not in feature else "_".join(feature.split('_')[:2])
                    rel = [c for c in df_mod.columns if c.startswith(base) and ('lag' in c or 'roll_mean' in c)]
                    for r in rel:
                        df_mod.loc[mask, r] = new_value
            return df_mod
        q = float(modifications["quantile"])
        top_feats = list(modifications.get("top_features", []))
        self._ensure_time_parts_inplace(df_mod)
        if modifications.get("donor_scope", "trained_only") == "trained_only":
            donor_df = self.trained_df_ref.copy()
        else:
            donor_df = self.trained_df_ref.copy() if self.trained_df_ref is not None else df_mod.copy()
        self._ensure_time_parts_inplace(donor_df)
        if "hour" in df_mod.columns:
            idxs = df_mod.index[df_mod["hour"].between(6, 18)].tolist()
        else:
            idxs = df_mod.index.tolist()
        for ridx in idxs:
            m = int(df_mod.at[ridx, "month"]) if "month" in df_mod.columns else None
            h = int(df_mod.at[ridx, "hour"]) if "hour" in df_mod.columns else None
            pool = donor_df.copy()
            if m is not None:
                pool = pool[pool["month"] == m]
            if h is not None:
                pool = pool[pool["hour"] == h]
            if len(pool) == 0 and m is not None and h is not None:
                pool = donor_df[(donor_df["month"].between(max(1, m-1), min(12, m+1))) & (donor_df["hour"] == h)]
            if len(pool) == 0:
                continue
            for feat in top_feats:
                if feat not in df_mod.columns or feat not in pool.columns:
                    continue
                try:
                    target_val = pool[feat].quantile(q)
                    df_mod.at[ridx, feat] = float(target_val)
                    base = feat.split('_')[0] if '_' not in feat else "_".join(feat.split('_')[:2])
                    rel = [c for c in df_mod.columns if c.startswith(base) and ('lag' in c or 'roll_mean' in c)]
                    for r in rel:
                        df_mod.at[ridx, r] = float(target_val)
                except Exception:
                    continue
        return df_mod

    def prepare_single_prediction_input(self, encoder_data: pd.DataFrame,
                                        target_time_idx: int, proxy_location: str):
        try:
            temp_data = encoder_data.copy()
            temp_data['location'] = proxy_location
            target_row = temp_data.iloc[-1:].copy()
            target_row['time_idx'] = target_time_idx
            target_row['location'] = proxy_location
            temp_data = pd.concat([temp_data, target_row], ignore_index=True)
            temp_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset, temp_data, predict=True, stop_randomization=True
            )
            temp_dataloader = temp_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            for batch in temp_dataloader:
                return batch
        except Exception:
            return None
        return None

    def make_single_prediction(self, input_batch) -> float:
        with torch.no_grad():
            x, y = input_batch
            prediction = self.model.forward(x)
            if isinstance(prediction, dict):
                pred_tensor = prediction.get('prediction', prediction.get('quantiles', prediction))
                if isinstance(pred_tensor, dict):
                    pred_tensor = pred_tensor.get(0.5, list(pred_tensor.values())[0])
            elif isinstance(prediction, (list, tuple)):
                pred_tensor = prediction[0]
            else:
                pred_tensor = prediction
            return float(pred_tensor.cpu().numpy().flatten()[0])

    def run_all_locations_scenario_analysis(self, scenario_df: pd.DataFrame,
                                            untrained_locations: List[str],
                                            sample_rate: int = 72) -> Dict[str, Any]:
        logger.info(f"Starting scenario analysis for ALL {len(untrained_locations)} untrained locations...")
        logger.info(f"Sampling every {sample_rate} hours for efficiency")
        scenarios = self.define_scenarios(self.trained_df_ref)
        results_by_location: Dict[str, Any] = {}
        for i, location in enumerate(untrained_locations):
            logger.info(f"Processing location {i+1}/{len(untrained_locations)}: {location}")
            proxy_id = self.assign_proxy_location(location)
            proxy_location = self.location_mapping[proxy_id]
            logger.info(f"  Using proxy: {proxy_location}")
            location_data = scenario_df[scenario_df['location'] == location]
            if len(location_data) < 1000:
                logger.warning(f"  Insufficient data for {location}: {len(location_data)} rows")
                continue
            all_time_indices = sorted(location_data['time_idx'].unique())
            valid_start_idx = self.encoder_length
            sampled_indices = all_time_indices[valid_start_idx::sample_rate]
            logger.info(f"  Processing {len(sampled_indices)} sampled time points")
            scenario_results: Dict[str, np.ndarray] = {}
            timestamps: List[Any] = []
            actuals: List[float] = []
            for scenario_name, modifications in scenarios.items():
                logger.info(f"    Running {scenario_name} scenario...")
                predictions: List[float] = []
                for j, target_time_idx in enumerate(sampled_indices):
                    try:
                        encoder_start = target_time_idx - self.encoder_length
                        encoder_data = location_data[
                            (location_data['time_idx'] >= encoder_start) &
                            (location_data['time_idx'] < target_time_idx)
                        ].copy()
                        if len(encoder_data) < self.encoder_length:
                            continue
                        modified_encoder_data = self.apply_scenario_to_encoder_data(
                            encoder_data, scenario_name, modifications
                        )
                        if scenario_name == list(scenarios.keys())[0]:
                            actual_row = location_data[location_data['time_idx'] == target_time_idx]
                            if len(actual_row) > 0:
                                actual_value = float(actual_row['active_power_mw'].iloc[0])
                                timestamp = actual_row['timestamp'].iloc[0] if 'timestamp' in actual_row.columns else target_time_idx
                                actuals.append(actual_value)
                                timestamps.append(str(timestamp))
                        input_data = self.prepare_single_prediction_input(
                            modified_encoder_data, target_time_idx, proxy_location
                        )
                        if input_data is not None:
                            prediction = self.make_single_prediction(input_data)
                            predictions.append(prediction)
                        else:
                            predictions.append(0.0)
                    except Exception:
                        predictions.append(0.0)
                        continue
                    if (j + 1) % 50 == 0:
                        logger.info(f"      {scenario_name}: {j + 1}/{len(sampled_indices)} predictions")
                scenario_results[scenario_name] = np.array(predictions, dtype=np.float32)
                logger.info(f"    {scenario_name} mean: {np.mean(predictions):.3f} MW")
            if 'MaxSolarAnalog' in scenario_results and 'MinSolarAnalog' in scenario_results:
                delta = scenario_results['MinSolarAnalog'] - scenario_results['MaxSolarAnalog']
                try:
                    out_path = os.path.join(self.stream_dir, f"{location}.json")
                    with open(out_path, "w") as f:
                        json.dump({
                            'proxy_used': proxy_location,
                            'proxy_id': int(proxy_id),
                            'timestamps': timestamps,
                            'actuals': [float(a) for a in actuals],
                            'MaxSolarAnalog': [float(x) for x in scenario_results['MaxSolarAnalog']],
                            'MinSolarAnalog': [float(x) for x in scenario_results['MinSolarAnalog']],
                            'Delta': [float(x) for x in delta],
                        }, f)
                except Exception:
                    pass
                results_by_location[location] = {
                    'proxy_used': proxy_location,
                    'proxy_id': int(proxy_id),
                    'delta_stats': {
                        'mean': float(np.mean(delta)),
                        'max': float(np.max(delta)),
                        'min': float(np.min(delta)),
                        'std': float(np.std(delta)),
                        'percentile_95': float(np.percentile(delta, 95)),
                        'percentile_05': float(np.percentile(delta, 5)),
                    },
                    'num_samples': int(len(timestamps)),
                }
                logger.info(f"  Solar capacity - Mean: {np.mean(delta):.3f}, Max: {np.max(delta):.3f}")
                del timestamps, actuals, delta
                for k in list(scenario_results.keys()):
                    del scenario_results[k]
                gc.collect()
            if (i + 1) % 5 == 0:
                gc.collect()
                logger.info(f"  Completed {i + 1}/{len(untrained_locations)} locations")
        return results_by_location

    def save_all_locations_results(self, all_results: Dict[str, Dict],
                                   save_path: str = "scenario_results_all_locations.json"):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return [float(x) for x in obj.tolist()]
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        serializable_results: Dict[str, Any] = {}
        for location, results in all_results.items():
            if not results:
                continue
            location_results = {}
            for key, value in results.items():
                location_results[key] = convert_to_serializable(value)
            serializable_results[location] = location_results

        if serializable_results:
            all_means = []
            proxy_perf: Dict[str, List[float]] = {}
            for loc, results in serializable_results.items():
                if 'delta_stats' in results:
                    mean_delta = results['delta_stats']['mean']
                    all_means.append(mean_delta)
                    proxy = results.get('proxy_used', 'unknown')
                    proxy_perf.setdefault(proxy, []).append(mean_delta)
            proxy_averages = {p: float(np.mean(v)) for p, v in proxy_perf.items()}
            summary = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'historical_analogy_scenario_analysis_all_locations',
                'total_locations': len(serializable_results),
                'locations_analyzed': list(serializable_results.keys()),
                'overall_stats': {
                    'avg_solar_capacity_mw': float(np.mean(all_means)) if all_means else 0.0,
                    'total_estimated_capacity_mw': float(np.sum(all_means)) if all_means else 0.0,
                    'max_solar_capacity_mw': float(np.max(all_means)) if all_means else 0.0,
                    'min_solar_capacity_mw': float(np.min(all_means)) if all_means else 0.0,
                    'std_solar_capacity_mw': float(np.std(all_means)) if all_means else 0.0,
                    'median_solar_capacity_mw': float(np.median(all_means)) if all_means else 0.0,
                },
                'proxy_performance': {
                    proxy: {
                        'avg_solar_capacity_mw': avg,
                        'num_locations': len(proxy_perf.get(proxy, [])),
                        'locations_using_proxy': len(proxy_perf.get(proxy, [])),
                    } for proxy, avg in proxy_averages.items()
                },
            }
            serializable_results['_summary'] = summary

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"All locations scenario results saved to: {save_path}")

        if serializable_results:
            rows = []
            for location, results in serializable_results.items():
                if location.startswith('_'):
                    continue
                if 'delta_stats' in results:
                    rows.append({
                        'location': location,
                        'proxy_used': results.get('proxy_used', ''),
                        'mean_solar_capacity_mw': results['delta_stats']['mean'],
                        'max_solar_capacity_mw': results['delta_stats']['max'],
                        'std_solar_capacity_mw': results['delta_stats']['std'],
                        'num_samples': results.get('num_samples', 0),
                    })
            if rows:
                summary_df = pd.DataFrame(rows).sort_values('mean_solar_capacity_mw', ascending=False)
                csv_path = save_path.replace('.json', '_summary.csv')
                summary_df.to_csv(csv_path, index=False)
                logger.info(f"Summary CSV saved to: {csv_path}")


def emergency_backup(results):
    import pickle
    with open(f'emergency_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Emergency backup saved")


def main():
    logger.info("Starting ALL LOCATIONS Scenario Analysis (Historical Analogue)...")
    analyzer = TFTScenarioAllLocationsAnalyzer(
        model_path="production_tft_model.ckpt",
        metadata_path="production_tft_metadata.pth",
    )
    analyzer.load_model_and_setup()
    logger.info(f"Loading data from {config['paths']['output_path']}")
    try:
        df_full = pd.read_parquet(config['paths']['output_path'])
        _downcast_numeric_inplace(df_full)
        logger.info(f"Loaded {len(df_full)} rows of data")
    except FileNotFoundError:
        logger.error("Data file not found")
        return
    FI_TEXT = """
Feature Importance Rankings
Solar Zenith Angle - 0.3388
Solar Intensity Factor - 0.2747
Weekend Indicator - 0.2572
Instantaneous Surface Heat Flux (W/m²) - 0.2170
Hour Sine Encoding - 0.2057
Wind Speed - 0.1924
Weather Index - 0.1874
Solar Radiation 3-Hour Rolling Mean (W/m²) - 0.1839
Year - 0.1770
Weekend Indicator - 0.1759
Civil Twilight Indicator - 0.1657
Wind Speed 3-Hour Rolling Mean - 0.1642
Solar Zenith Angle - 0.1591
Solar Radiation 3-Hour Rolling Mean (W/m²) - 0.1561
Solar Azimuth Angle - 0.1552
""".strip()
    analyzer.set_external_feature_importances(
        FI_TEXT, available_columns=df_full.columns.tolist(), top_k=10
    )
    trained_locations, untrained_locations, trained_df, scenario_df, val_cutoff = analyzer.prepare_all_locations_data(df_full)
    analyzer.create_training_dataset(trained_df, val_cutoff)
    logger.info(f"Running scenario analysis on ALL {len(untrained_locations)} locations...")
    all_results = analyzer.run_all_locations_scenario_analysis(
        scenario_df, untrained_locations, sample_rate=72
    )
    if not all_results:
        logger.error("No results generated")
        return
    emergency_backup(all_results)
    analyzer.save_all_locations_results(all_results)
    logger.info("\n" + "="*80)
    logger.info("ALL LOCATIONS SCENARIO ANALYSIS SUMMARY (Historical Analogue)")
    logger.info("="*80)
    all_means = [results['delta_stats']['mean'] for results in all_results.values() if 'delta_stats' in results]
    logger.info(f"Successfully analyzed {len(all_results)} locations")
    if all_means:
        logger.info(f"Total estimated solar capacity: {sum(all_means):.1f} MW")
        logger.info(f"Average solar capacity per location: {np.mean(all_means):.3f} MW")
        logger.info(f"Range: {min(all_means):.3f} to {max(all_means):.3f} MW")
    logger.info("\nFiles created:")
    logger.info(f"  - {analyzer.stream_dir}/ (per-location arrays streamed to disk)")
    logger.info("  - scenario_results_all_locations.json (stats; arrays are in stream dir)")
    logger.info("  - scenario_results_all_locations_summary.csv (summary table)")


if __name__ == "__main__":
    main()
