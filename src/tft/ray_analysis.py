import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pytorch_forecasting import TemporalFusionTransformer
from ray import tune

from model import TFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    with open("uk-power-flow-to-solar-capacity/src/tft/config_tft_initial.yaml") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise


def load_best_model():
    experiment_path = config["analysis"]["experiment_path"]
    logger.info(f"Loading results from: {experiment_path}")
    
    try:
        tuner = tune.Tuner.restore(path=experiment_path, trainable="ray_objective_func")
        result_grid = tuner.get_results()
        
        metrics_config = config["analysis"]["metrics"]
        best_result = result_grid.get_best_result(
            metric=metrics_config["primary_metric"], 
            mode=metrics_config["mode"]
        )
        best_checkpoint = best_result.get_best_checkpoint(
            metric=metrics_config["primary_metric"], 
            mode=metrics_config["mode"]
        )
        
        val_loss = best_result.metrics[metrics_config["primary_metric"]]
        logger.info(f"Best trial's final validation loss: {val_loss:.4f}")
        
        model_path = os.path.join(best_checkpoint.path, "checkpoint")
        best_tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        logger.info("Best model loaded successfully")
        
        return best_tft_model, val_loss
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_data_and_setup_trainer():
    data_path = config["analysis"]["data_path"]
    logger.info(f"Loading data from: {data_path}")
    
    try:
        df_pandas = pd.read_parquet(data_path)
        logger.info(f"Loaded dataset with {len(df_pandas):,} rows")
        
        trainer_setup = TFTTrainer(df_pandas)
        trainer_setup.create_datasets()
        _, val_dataloader, _ = trainer_setup.create_dataloaders()
        logger.info("Validation dataloader created")
        
        return df_pandas, val_dataloader
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def create_interpretation_plot(model, val_dataloader):
    logger.info("Generating model interpretation...")
    
    try:
        interp_config = config["analysis"]["interpretation"]
        interpretation = model.interpret_output(
            val_dataloader,
            reduction=interp_config["reduction"],
            attention_prediction_horizon=interp_config["attention_prediction_horizon"],
        )
        
        figure = model.plot_interpretation(interpretation)
        plot_config = config["analysis"]["plots"]
        figure.suptitle(
            plot_config["main_title"], 
            fontsize=plot_config["main_title_fontsize"]
        )
        
        save_config = config["analysis"].get("save_plots", {})
        if save_config.get("enabled", False):
            save_plot(figure, save_config["interpretation_filename"], save_config)
        
        if plot_config.get("show_plots", True):
            plt.show()
        else:
            plt.close(figure)
            
        logger.info("Interpretation plot created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create interpretation plot: {e}")
        raise


def create_correlation_plot(df, feature_columns):
    logger.info("Creating feature correlation plot...")
    
    try:
        corr_config = config["analysis"]["correlation"]
        corr_matrix = df[feature_columns].corr(method=corr_config["method"])
        
        plot_config = config["analysis"]["plots"]
        figsize = tuple(plot_config["correlation_figsize"])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix, 
            annot=plot_config["correlation_annotate"], 
            cmap=plot_config["correlation_colormap"]
        )
        plt.title(
            plot_config["correlation_title"], 
            fontsize=plot_config["correlation_title_fontsize"]
        )
        
        save_config = config["analysis"].get("save_plots", {})
        if save_config.get("enabled", False):
            save_plot(plt.gcf(), save_config["correlation_filename"], save_config)
        
        if plot_config.get("show_plots", True):
            plt.show()
        else:
            plt.close()
            
        logger.info("Correlation plot created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create correlation plot: {e}")
        raise


def save_plot(figure, filename, save_config):
    try:
        output_dir = Path(save_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        figure.savefig(filepath, dpi=save_config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Plot saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")


def analyse_experiment_results():
    logger.info("Starting TFT experiment analysis...")
    
    try:
        best_model, val_loss = load_best_model()
        df_pandas, val_dataloader = load_data_and_setup_trainer()
        create_interpretation_plot(best_model, val_dataloader)
        
        feature_columns = best_model.hparams.time_varying_known_reals
        create_correlation_plot(df_pandas, feature_columns)
        
        logger.info("=" * 50)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"Best validation loss: {val_loss:.4f}")
        logger.info(f"Features analysed: {len(feature_columns)}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyse_experiment_results()
