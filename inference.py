import yaml
import logging
import pandas as pd

from src.modeling import load_model
from src.utils import run_quality_assessment


def setup_logging(logging_config):
    logging.basicConfig(
        level=logging_config.get('level', 'INFO'),
        format=logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
        datefmt=logging_config.get('datefmt', '%Y-%m-%d %H:%M:%S')
    )


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config['logging_settings'])
    paths = config['paths']
    
    model = load_model(paths['model_output_dir'], paths['output_model_filename'])
    if model is None:
        return

    idx = pd.MultiIndex.from_product(
        [['site_a'], pd.to_datetime(pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC'))],
        names=['site_id', 'datetime']
    )
    new_X = pd.DataFrame(index=idx, data={'tcc_lag_1h': range(24), 't2m_lag_1h': range(24), 'ssrd_lag_1h': range(24)})
    

    predictions = model.predict(new_X)

    results = pd.DataFrame({'timestamp': new_X.index.get_level_values('datetime'), 'predicted_power_mw': predictions})
    print("\n--- Prediction Results ---")
    print(results)
    print("------------------------\n")

if __name__ == '__main__':
    main()
