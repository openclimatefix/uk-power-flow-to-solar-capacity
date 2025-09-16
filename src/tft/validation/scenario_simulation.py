import gc
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from utils import (
    load_config, setup_environment, assign_proxy_location, 
    define_scenarios, apply_scenario_to_encoder_data,
    convert_to_serializable, process_timestamps, create_summary_stats,
    TRAINED_LOCATIONS, LOCATION_MAPPING
)


config = load_config()
logger = setup_environment()


class TFTScenarioAllLocationsAnalyzer:

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.training_dataset = None
        self.encoder_length = 168
        
    def load_model_and_setup(self):
        self.metadata = torch.load(self.metadata_path, map_location='cpu')
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            self.model_path, map_location='cpu'
        )
        self.model.eval()
        
    def prepare_all_locations_data(self, df_full: pd.DataFrame):
        all_locations = df_full['location'].unique()
        untrained_locations = [loc for loc in all_locations if loc not in TRAINED_LOCATIONS.keys()]
        
        trained_df = df_full[df_full['location'].isin(TRAINED_LOCATIONS.keys())].copy()
        scenario_df = df_full.copy()
        
        max_time_idx = trained_df['time_idx'].max()
        train_cutoff = int(max_time_idx * config["train_split"])
        
        return list(TRAINED_LOCATIONS.keys()), untrained_locations, trained_df, scenario_df, train_cutoff
        
    def create_training_dataset(self, trained_df: pd.DataFrame, train_cutoff: int):
        model_config = config['model']
        
        self.training_dataset = TimeSeriesDataSet(
            trained_df[trained_df["time_idx"] <= train_cutoff],
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
        
    def prepare_single_prediction_input(self, encoder_data: pd.DataFrame, 
                                      target_time_idx: int, proxy_location: str) -> Dict:
        try:
            temp_data = encoder_data.copy()
            temp_data['location'] = proxy_location
            
            target_row = temp_data.iloc[-1:].copy()
            target_row['time_idx'] = target_time_idx
            target_row['location'] = proxy_location
            temp_data = pd.concat([temp_data, target_row], ignore_index=True)
            
            temp_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                temp_data,
                predict=True,
                stop_randomization=True
            )
            
            temp_dataloader = temp_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )
            
            for batch in temp_dataloader:
                return batch
            
            return None
            
        except Exception as e:
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
        scenarios = define_scenarios()
        results_by_location = {}
        
        for i, location in enumerate(untrained_locations):
            proxy_id = assign_proxy_location(location)
            proxy_location = LOCATION_MAPPING[proxy_id]
            
            location_data = scenario_df[scenario_df['location'] == location].copy()
            
            if len(location_data) < 1000:
                continue
            
            all_time_indices = sorted(location_data['time_idx'].unique())
            valid_start_idx = self.encoder_length
            sampled_indices = all_time_indices[valid_start_idx::sample_rate]
            
            scenario_results = {}
            timestamps = []
            actuals = []
            
            for scenario_name, modifications in scenarios.items():
                predictions = []
                
                for j, target_time_idx in enumerate(sampled_indices):
                    try:
                        encoder_start = target_time_idx - self.encoder_length
                        encoder_data = location_data[
                            (location_data['time_idx'] >= encoder_start) & 
                            (location_data['time_idx'] < target_time_idx)
                        ].copy()
                        
                        if len(encoder_data) < self.encoder_length:
                            continue
                        
                        modified_encoder_data = apply_scenario_to_encoder_data(
                            encoder_data, scenario_name, modifications
                        )
                        
                        if scenario_name == list(scenarios.keys())[0]:
                            actual_row = location_data[location_data['time_idx'] == target_time_idx]
                            if len(actual_row) > 0:
                                actual_value = actual_row['active_power_mw'].iloc[0]
                                timestamp = actual_row['timestamp'].iloc[0] if 'timestamp' in actual_row.columns else target_time_idx
                                actuals.append(actual_value)
                                timestamps.append(timestamp)
                        
                        input_data = self.prepare_single_prediction_input(
                            modified_encoder_data, target_time_idx, proxy_location
                        )
                        
                        if input_data is not None:
                            prediction = self.make_single_prediction(input_data)
                            predictions.append(prediction)
                        else:
                            predictions.append(0.0)
                    
                    except Exception as e:
                        predictions.append(0.0)
                        continue
                
                scenario_results[scenario_name] = np.array(predictions)
            
            if 'MaxSolar' in scenario_results and 'MinSolar' in scenario_results:
                delta = scenario_results['MinSolar'] - scenario_results['MaxSolar']
                
                results_by_location[location] = {
                    'proxy_used': proxy_location,
                    'proxy_id': proxy_id,
                    'timestamps': timestamps,
                    'actuals': actuals,
                    'MaxSolar': scenario_results['MaxSolar'],
                    'MinSolar': scenario_results['MinSolar'],
                    'Delta': delta,
                    'delta_stats': {
                        'mean': float(np.mean(delta)),
                        'max': float(np.max(delta)),
                        'min': float(np.min(delta)),
                        'std': float(np.std(delta)),
                        'percentile_95': float(np.percentile(delta, 95)),
                        'percentile_05': float(np.percentile(delta, 5))
                    }
                }
            
            if (i + 1) % 50 == 0:
                gc.collect()
        
        return results_by_location
    
    def save_all_locations_results(self, all_results: Dict[str, Dict], 
                                  save_path: str = "scenario_results_all_locations.json"):
        serializable_results = {}
        
        for location, results in all_results.items():
            if not results:
                continue
                
            location_results = {}
            
            for key, value in results.items():
                if key == 'timestamps':
                    location_results[key] = process_timestamps(value)
                else:
                    location_results[key] = convert_to_serializable(value)
            
            serializable_results[location] = location_results
        
        if serializable_results:
            all_deltas = []
            proxy_performance = {}
            
            for loc, results in serializable_results.items():
                if 'delta_stats' in results:
                    mean_delta = results['delta_stats']['mean']
                    all_deltas.append(mean_delta)
                    
                    proxy = results.get('proxy_used', 'unknown')
                    if proxy not in proxy_performance:
                        proxy_performance[proxy] = []
                    proxy_performance[proxy].append(mean_delta)
            
            summary = create_summary_stats(all_deltas, proxy_performance)
            summary['total_locations'] = len(serializable_results)
            summary['locations_analyzed'] = list(serializable_results.keys())
            serializable_results['_summary'] = summary
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if serializable_results:
            summary_rows = []
            for location, results in serializable_results.items():
                if location.startswith('_'):
                    continue
                if 'delta_stats' in results:
                    summary_rows.append({
                        'location': location,
                        'proxy_used': results.get('proxy_used', ''),
                        'mean_solar_capacity_mw': results['delta_stats']['mean'],
                        'max_solar_capacity_mw': results['delta_stats']['max'],
                        'std_solar_capacity_mw': results['delta_stats']['std'],
                        'num_samples': len(results.get('timestamps', []))
                    })
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df = summary_df.sort_values('mean_solar_capacity_mw', ascending=False)
                csv_path = save_path.replace('.json', '_summary.csv')
                summary_df.to_csv(csv_path, index=False)


def main():

    analyzer = TFTScenarioAllLocationsAnalyzer(
        model_path="production_tft_model.ckpt",
        metadata_path="production_tft_metadata.pth"
    )
    
    analyzer.load_model_and_setup()
    
    try:
        df_full = pd.read_parquet(config['paths']['output_path'])
    except FileNotFoundError:
        return
    
    trained_locations, untrained_locations, trained_df, scenario_df, train_cutoff = analyzer.prepare_all_locations_data(df_full)
    analyzer.create_training_dataset(trained_df, train_cutoff)
    
    all_results = analyzer.run_all_locations_scenario_analysis(
        scenario_df, untrained_locations, sample_rate=72
    )
    
    if not all_results:
        return
    
    analyzer.save_all_locations_results(all_results)
    all_deltas = [results['delta_stats']['mean'] for results in all_results.values() if 'delta_stats' in results]
    
    print(f"Successfully analyzed {len(all_results)} locations")
    print(f"Total estimated solar capacity: {sum(all_deltas):.1f} MW")
    print(f"Average solar capacity per location: {np.mean(all_deltas):.3f} MW")
    print(f"Range: {min(all_deltas):.3f} to {max(all_deltas):.3f} MW")


if __name__ == "__main__":
    main()

