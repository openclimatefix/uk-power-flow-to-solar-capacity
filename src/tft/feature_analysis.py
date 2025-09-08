import json
import logging
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    with open("config_tft_initial.yaml") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise


def analyze_weather_features(model, test_data_df):
    training_params = model.dataset_parameters
    
    test_dataset = TimeSeriesDataSet.from_parameters(
        training_params,
        test_data_df,
        predict=True,
        stop_randomization=True
    )
    
    test_dataloader = test_dataset.to_dataloader(
        train=False, 
        batch_size=64,
        num_workers=0
    )
    
    raw_predictions = model.predict(
        test_dataloader, 
        mode="raw", 
        return_x=True,
        trainer_kwargs={"accelerator": "cpu"}
    )
    
    if hasattr(raw_predictions, 'output'):
        raw_output = raw_predictions.output
    elif hasattr(raw_predictions, '_asdict'):
        pred_dict = raw_predictions._asdict()
        raw_output = pred_dict.get('output', pred_dict)
    elif isinstance(raw_predictions, tuple) and len(raw_predictions) > 0:
        raw_output = raw_predictions[0]
    else:
        raw_output = raw_predictions
    
    if not isinstance(raw_output, dict):
        if hasattr(raw_output, '_asdict'):
            raw_output = raw_output._asdict()
        elif hasattr(raw_output, '__dict__'):
            raw_output = raw_output.__dict__
        else:
            raise ValueError(f"Cannot convert {type(raw_output)} to dictionary for interpretation")
    
    interpretation = model.interpret_output(raw_output, reduction="sum")
    
    encoder_importance = interpretation["encoder_variables"]
    decoder_importance = interpretation["decoder_variables"] 
    attention = interpretation["attention"]
    
    encoder_vars = list(model.encoder_variables)
    decoder_vars = list(model.decoder_variables)
    
    all_vars = encoder_vars + decoder_vars
    all_importance = np.concatenate([encoder_importance.numpy(), decoder_importance.numpy()])
    
    weather_categories = {
        'Solar': ['ssrd_w_m2', 'ssr_w_m2', 'str_w_m2', 'slhf_w_m2', 'sshf_w_m2', 'ishf_w_m2',
                  'ssrd_w_m2_lag_1h', 'ssrd_w_m2_lag_24h', 'ssrd_w_m2_roll_mean_3h',
                  'irradiance_x_clear_sky', 'solar_elevation_angle', 'solar_azimuth_angle',
                  'solar_zenith_angle', 'solar_intensity_factor'],
        'Temperature': ['t2m_c', 'd2m_c', 'skt_c', 't2m_c_lag_1h', 't2m_c_lag_24h', 
                       't2m_c_roll_mean_3h', 'temp_x_hour_sin', 'temp_x_hour_cos'],
        'Wind': ['u10', 'v10', 'wind_speed', 'wind_speed_lag_1h', 'wind_speed_lag_24h',
                 'wind_speed_roll_mean_3h', 'wind_dir_sin', 'wind_dir_cos'],
        'Atmospheric': ['tcc', 'blh', 'msl_hpa', 'sp_hpa', 'tp_mm', 'weather_index']
    }
    
    category_importance = {}
    for category, features in weather_categories.items():
        total = 0
        for feature in features:
            if feature in all_vars:
                idx = all_vars.index(feature)
                total += all_importance[idx]
        category_importance[category] = total
    
    feature_importance = list(zip(all_vars, all_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    category_ranking = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    
    encoder_len = model.hparams.max_encoder_length
    
    if hasattr(attention, 'cpu'):
        attention_np = attention.cpu().numpy()
    else:
        attention_np = attention
    
    if len(attention_np.shape) == 3:
        attention_mean = attention_np.mean(axis=(0, 1))
    elif len(attention_np.shape) == 2:
        attention_mean = attention_np.mean(axis=0)
    elif len(attention_np.shape) == 1:
        attention_mean = attention_np
    else:
        attention_mean = attention_np.mean(axis=tuple(range(len(attention_np.shape)-1)))
    
    if len(attention_mean) >= encoder_len:
        hist_attention = attention_mean[:encoder_len].mean()
        future_attention = attention_mean[encoder_len:].mean() if len(attention_mean) > encoder_len else 0.0
    else:
        hist_attention = attention_mean.mean()
        future_attention = 0.0
    
    logger.info("Feature importance analysis results:")
    logger.info(f"Top 10 features:")
    for i, (feature, imp) in enumerate(feature_importance[:10]):
        logger.info(f"{i+1:2d}. {feature:<35s} {imp:>8.4f}")
    
    logger.info("Weather category importance:")
    total_category_importance = sum(category_importance.values())
    for i, (cat, imp) in enumerate(category_ranking):
        pct = (imp / total_category_importance * 100) if total_category_importance > 0 else 0
        logger.info(f"{i+1}. {cat:<12s} {imp:>8.4f} ({pct:>5.1f}%)")
    
    total_attention = hist_attention + future_attention
    hist_pct = (hist_attention / total_attention * 100) if total_attention > 0 else 0
    future_pct = (future_attention / total_attention * 100) if total_attention > 0 else 0
    logger.info(f"Temporal attention - Historical: {hist_attention:.4f} ({hist_pct:.1f}%), Future: {future_attention:.4f} ({future_pct:.1f}%)")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    top_features = feature_importance[:25]
    features, importances = zip(*top_features)
    y_pos = np.arange(len(features))
    axes[0].barh(y_pos, importances, color='steelblue')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(features)
    axes[0].set_title('Top 25 Features by Importance', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    cats, scores = zip(*category_ranking)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[1].bar(cats, scores, color=colors[:len(cats)])
    axes[1].set_title('Weather Category Importance', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Importance Score')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + score*0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    time_importance = attention_mean
    axes[2].plot(time_importance, linewidth=2, color='darkred', label='Attention Weight')
    axes[2].axvline(x=encoder_len, color='blue', linestyle='--', linewidth=2, 
                   label=f'Prediction Start (t={encoder_len})')
    axes[2].set_title('Temporal Attention Pattern', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Index')
    axes[2].set_ylabel('Attention Weight')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].fill_between(range(encoder_len), time_importance[:encoder_len], 
                        alpha=0.3, color='green', label='Historical')
    axes[2].fill_between(range(encoder_len, len(time_importance)), 
                        time_importance[encoder_len:], alpha=0.3, color='orange', label='Future')

    plt.tight_layout()
    fig.suptitle("TFT Weather Feature Importance Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    output_file = 'weather_feature_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Analysis plots saved to {output_file}")
    plt.close()
    
    results = {
        'top_25_features': [{'feature': f, 'importance': float(i)} for f, i in feature_importance[:25]],
        'category_importance': [{'category': c, 'importance': float(i)} for c, i in category_ranking],
        'temporal_attention': {
            'historical_avg': float(hist_attention),
            'future_avg': float(future_attention),
            'historical_percentage': float(hist_pct),
            'future_percentage': float(future_pct)
        },
        'model_info': {
            'encoder_length': encoder_len,
            'total_encoder_vars': len(encoder_vars),
            'total_decoder_vars': len(decoder_vars),
            'total_features': len(all_vars)
        }
    }
    
    results_file = 'feature_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {results_file}")
    
    return results


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model_path = os.path.expanduser("~/production_tft_model.ckpt")
    
    logger.info("Starting TFT feature importance analysis")
    
    try:
        trained_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {model_path}")
        logger.error("Please run the training script first to create the model")
        return
    
    data_path = config['paths']['output_path']
    logger.info(f"Loading data from {data_path}")
    df_pandas_full = pd.read_parquet(data_path)
    
    np.random.seed(42)
    all_locations = df_pandas_full['location'].unique()
    selected_locations = np.random.choice(all_locations, size=5, replace=False)
    df_pandas = df_pandas_full[df_pandas_full['location'].isin(selected_locations)].copy()
    logger.info(f"Filtered data to 5 sites: {list(selected_locations)}")
    
    model_config = config['model']
    max_time_idx = df_pandas[model_config['time_idx']].max()
    test_data_df = df_pandas[lambda x: x.time_idx > int(max_time_idx * config["val_split"])]
    logger.info(f"Test data prepared: {len(test_data_df)} samples")
    
    try:
        results = analyze_weather_features(trained_model, test_data_df)
        logger.info("Analysis completed successfully")
        
        logger.info("Summary:")
        logger.info(f"Most important feature: {results['top_25_features'][0]['feature']}")
        logger.info(f"Most important category: {results['category_importance'][0]['category']}")
        logger.info(f"Model focuses {results['temporal_attention']['historical_percentage']:.1f}% on historical data")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
