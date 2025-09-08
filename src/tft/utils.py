import os
import json

import logging


logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format=config['logging']['format'],
    handlers=handlers
)


def load_best_hyperparameters(results_path: str) -> Dict[str, Any]:
    with open(results_path, 'r') as f:
        ray_results = json.load(f)
    optimal_config = ray_results['best_params']
    
    logger.info("Best parameters from Ray Tune:")
    for key, value in optimal_config.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return optimal_config
