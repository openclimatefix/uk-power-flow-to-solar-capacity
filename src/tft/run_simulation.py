import os
import logging
from scenario_simulation.analyser import SingleSiteScenarioAnalyzer
from scenario_simulation.plotting import plot_scenario_timeseries, make_multi_site_sv1_plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
