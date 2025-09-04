import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting.utils import get_prediction_permutation_importance

def analyze_weather_features(model, test_dataloader):
    
    interpretation = model.interpret_output(test_dataloader, reduction="sum")
    encoder_importance = interpretation["encoder_variables"]
    decoder_importance = interpretation["decoder_variables"]
    attention = interpretation["attention"]
    encoder_vars = model.encoder_variables
    decoder_vars = model.decoder_variables
    all_vars = encoder_vars + decoder_vars
    all_importance = np.concatenate([encoder_importance, decoder_importance])
    
    # CATEGORIES RESTATED
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
    
    # CATEGORY IMPORTANCE
    category_importance = {}
    for category, features in weather_categories.items():
        total = sum(all_importance[all_vars.index(f)] for f in features if f in all_vars)
        category_importance[category] = total
    
    # RANK INDIVIDUAL FEATURES
    feature_importance = list(zip(all_vars, all_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    category_ranking = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    top_weather = [f for f, _ in feature_importance[:8] if any(f in cats for cats in weather_categories.values())]
    
    try:
        perm_importance = get_prediction_permutation_importance(
            model, test_dataloader, feature_names=top_weather, n_repeats=3
        )
        perm_sorted = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
    except:
        perm_sorted = []
    
    encoder_len = model.hparams.max_encoder_length
    if attention.size > 0:
        attention_mean = attention.mean(axis=(0,1)) if attention.ndim > 2 else attention.mean(axis=0)
        hist_attention = attention_mean[:encoder_len].mean()
        future_attention = attention_mean[encoder_len:].mean()
    else:
        hist_attention = future_attention = 0
    
    print("Top Weather Features - TFT Attention:")
    for i, (feature, imp) in enumerate(feature_importance[:15]):
        print(f"{i+1:2d}. {feature:35s} {imp:.4f}")
    
    print("\nWeather Categories:")
    for i, (cat, imp) in enumerate(category_ranking):
        print(f"{i+1}. {cat:12s} {imp:.4f}")
    
    if perm_sorted:
        print("\nPermutation Importance:")
        for feature, imp in perm_sorted:
            print(f"{feature:35s} {imp:.4f}")
    
    print(f"\nTemporal Attention:")
    print(f"Historical: {hist_attention:.4f}")
    print(f"Future: {future_attention:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    top_features = feature_importance[:12]
    features, importances = zip(*top_features)
    axes[0,0].barh(range(len(features)), importances)
    axes[0,0].set_yticks(range(len(features)))
    axes[0,0].set_yticklabels([f[:25] for f in features])
    axes[0,0].set_title('Top Weather Features')
    axes[0,0].invert_yaxis()
    
    cats, scores = zip(*category_ranking)
    colors = ['red', 'blue', 'green', 'orange']
    axes[0,1].bar(cats, scores, color=colors)
    axes[0,1].set_title('Weather Categories')
    axes[0,1].set_ylabel('Importance')
    
    if perm_sorted:
        common_features = [f for f, _ in perm_sorted]
        tft_vals = [all_importance[all_vars.index(f)] for f in common_features]
        perm_vals = [imp for _, imp in perm_sorted]
        
        axes[1,0].scatter(tft_vals, perm_vals)
        axes[1,0].set_xlabel('TFT Importance')
        axes[1,0].set_ylabel('Permutation Importance')
        axes[1,0].set_title('TFT vs Permutation')
        
        for i, f in enumerate(common_features):
            axes[1,0].annotate(f[:15], (tft_vals[i], perm_vals[i]), fontsize=8)
    
    if attention.size > 0 and attention.ndim > 1:
        time_importance = attention_mean if attention_mean.size > 0 else np.zeros(10)
        axes[1,1].plot(time_importance[:min(len(time_importance), 20)])
        axes[1,1].axvline(x=encoder_len, color='red', linestyle='--', label='Prediction start')
        axes[1,1].set_title('Temporal Attention Pattern')
        axes[1,1].set_xlabel('Time Steps')
        axes[1,1].set_ylabel('Attention Weight')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('weather_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'feature_ranking': dict(feature_importance[:20]),
        'category_importance': category_importance,
        'permutation_importance': dict(perm_sorted) if perm_sorted else {},
        'temporal_attention': {'historical': hist_attention, 'future': future_attention}
    }
