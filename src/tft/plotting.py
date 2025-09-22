from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def plot_tft_interpretation(results: dict[str, Any], config: dict[str, Any]) -> None:
    analysis_cfg = config.get("analysis", {})
    plots_cfg = analysis_cfg.get("plots", {})
    save_cfg = analysis_cfg.get("save_plots", {})

    main_title = plots_cfg.get("main_title", "TFT Weather Feature Importance Analysis")
    main_title_fontsize = int(plots_cfg.get("main_title_fontsize", 16))
    show_plots = bool(plots_cfg.get("show_plots", False))

    save_enabled = bool(save_cfg.get("enabled", True))
    out_dir = Path(save_cfg.get("output_dir", "."))
    out_png = out_dir / str(save_cfg.get("interpretation_filename", "tft_interpretation.png"))
    dpi = int(save_cfg.get("dpi", 300))

    top_features = results["top_25_features"]
    cats = results["category_importance"]
    time_importance = np.array(results["raw"]["attention_mean"], dtype=float)
    encoder_len = int(results["model_info"]["encoder_length"])

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    features = [x["feature"] for x in top_features]
    importances = [x["importance"] for x in top_features]
    y_pos = np.arange(len(features))
    axes[0].barh(y_pos, importances)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(features)
    axes[0].set_title("Top 25 Features by Importance", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Importance Score")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    cat_labels = [x["category"] for x in cats]
    cat_scores = [x["importance"] for x in cats]
    bars = axes[1].bar(cat_labels, cat_scores)
    axes[1].set_title("Weather Category Importance", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Importance Score")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, cat_scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height + max(cat_scores) * 0.01, f"{score:.3f}",
                     ha="center", va="bottom")

    axes[2].plot(time_importance, linewidth=2, label="Attention Weight")
    axes[2].axvline(x=encoder_len, linestyle="--", linewidth=2, label=f"Prediction Start (t={encoder_len})")
    axes[2].set_title("Temporal Attention Pattern", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Time Index")
    axes[2].set_ylabel("Attention Weight")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    hist_len = min(encoder_len, len(time_importance))
    axes[2].fill_between(range(hist_len), time_importance[:hist_len], alpha=0.3, label="Historical")
    if len(time_importance) > encoder_len:
        axes[2].fill_between(range(encoder_len, len(time_importance)), time_importance[encoder_len:], alpha=0.3, label="Future")

    plt.tight_layout()
    fig.suptitle(main_title, fontsize=main_title_fontsize, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.93)

    if save_enabled:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
