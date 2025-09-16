import logging
import warnings
import yaml
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def load_config(config_path: str = "uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_environment():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    plt.style.use('default')
    return logging.getLogger(__name__)


def assign_proxy_location(location_name: str) -> int:
    name_lower = location_name.lower()
    
    if any(region in name_lower for region in ['essex', 'chelmsford', 'colchester', 'harlow', 'brentwood']):
        return 1
    elif any(region in name_lower for region in ['norfolk', 'norwich', 'swaffham', 'thetford']):
        return 4
    elif any(region in name_lower for region in ['suffolk', 'bury', 'ipswich', 'sudbury']):
        return 0
    elif any(region in name_lower for region in ['cambridge', 'huntingdon', 'peterborough', 'ely']):
        return 3
    elif any(region in name_lower for region in ['kent', 'tunbridge', 'maidstone']):
        return 2
    else:
        first_letter = name_lower[0]
        if first_letter <= 'e':
            return 0
        elif first_letter <= 'j':
            return 1
        elif first_letter <= 'n':
            return 2
        elif first_letter <= 'r':
            return 3
        else:
            return 4


def define_scenarios() -> Dict[str, Dict]:
    return {
        'MaxSolar': {
            'solar_zenith_angle': 0.0,
            'solar_intensity_factor': 1.0,
            'solar_elevation_angle': 90.0,
            'ssrd_w_m2': 1200.0,
            'ssr_w_m2': 1200.0,
            'irradiance_x_clear_sky': 1.0,
            'weather_index': 1.0,
            'tcc': 0.0,
            'is_daylight': 1.0,
            't2m_c': 25.0,
        },
        'MinSolar': {
            'solar_zenith_angle': 90.0,
            'solar_intensity_factor': 0.0,
            'solar_elevation_angle': 0.0,
            'ssrd_w_m2': 0.0,
            'ssr_w_m2': 0.0,
            'irradiance_x_clear_sky': 0.0,
            'weather_index': 0.0,
            'tcc': 1.0,
            'is_daylight': 1.0,
            't2m_c': 5.0,
        }
    }


def apply_scenario_to_encoder_data(encoder_data: pd.DataFrame, 
                                 scenario_name: str, modifications: Dict) -> pd.DataFrame:
    if modifications is None:
        return encoder_data.copy()
        
    df_modified = encoder_data.copy()
    
    if 'timestamp' in df_modified.columns:
        df_modified['timestamp'] = pd.to_datetime(df_modified['timestamp'])
        mask = (df_modified['timestamp'].dt.hour >= 6) & (df_modified['timestamp'].dt.hour <= 18)
    else:
        mask = slice(None)
    
    for feature, new_value in modifications.items():
        if feature in df_modified.columns:
            if isinstance(mask, slice):
                df_modified.loc[:, feature] = new_value
            else:
                df_modified.loc[mask, feature] = new_value
            
            base_feature = feature.split('_')[0] + '_' + feature.split('_')[1] if len(feature.split('_')) > 1 else feature.split('_')[0]
            related_features = [col for col in df_modified.columns 
                              if col.startswith(base_feature) and ('lag' in col or 'roll_mean' in col)]
            
            for related_feature in related_features:
                if isinstance(mask, slice):
                    df_modified.loc[:, related_feature] = new_value
                else:
                    df_modified.loc[mask, related_feature] = new_value
    
    return df_modified


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return [float(x) for x in obj.tolist()]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def process_timestamps(timestamps):
    if not timestamps:
        return []
    
    if hasattr(timestamps[0], 'strftime'):
        return [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    elif isinstance(timestamps[0], np.datetime64):
        return [pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    else:
        return [str(ts) for ts in timestamps]


def create_summary_stats(all_deltas, proxy_performance):
    proxy_averages = {proxy: np.mean(values) for proxy, values in proxy_performance.items()}
    
    return {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'zero_shot_scenario_analysis_all_locations',
        'overall_stats': {
            'avg_solar_capacity_mw': float(np.mean(all_deltas)) if all_deltas else 0,
            'total_estimated_capacity_mw': float(np.sum(all_deltas)) if all_deltas else 0,
            'max_solar_capacity_mw': float(np.max(all_deltas)) if all_deltas else 0,
            'min_solar_capacity_mw': float(np.min(all_deltas)) if all_deltas else 0,
            'std_solar_capacity_mw': float(np.std(all_deltas)) if all_deltas else 0,
            'median_solar_capacity_mw': float(np.median(all_deltas)) if all_deltas else 0
        },
        'proxy_performance': {
            proxy: {
                'avg_solar_capacity_mw': float(avg_capacity),
                'num_locations': len(proxy_performance[proxy]),
                'locations_using_proxy': len(proxy_performance[proxy])
            }
            for proxy, avg_capacity in proxy_averages.items()
        }
    }


TRAINED_LOCATIONS = {
    'belchamp_grid_11kv': 0,
    'george_hill_primary_11kv': 1, 
    'manor_way_primary_11kv': 2,
    'st_stephens_primary_11kv': 3,
    'swaffham_grid_11kv': 4
}


LOCATION_MAPPING = {
    0: 'belchamp_grid_11kv',
    1: 'george_hill_primary_11kv', 
    2: 'manor_way_primary_11kv',
    3: 'st_stephens_primary_11kv',
    4: 'swaffham_grid_11kv'
}
