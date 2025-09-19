import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from pytorch_forecasting import TemporalFusionTransformer
from .analyser import SingleSiteScenarioAnalyzer

logger = logging.getLogger(__name__)

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

def _extract_sv1_ts_delta(results: dict):
    if isinstance(results, dict) and "Delta" in results:
        delta = np.array(results["Delta"])
        ts = results.get("timestamps", [])
        return ts, delta
    return [], np.array([])

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
