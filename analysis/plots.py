"""
analysis/plots.py – Matplotlib visualisations for HL Carry analysis

Generates static PNG files for:
- Cumulative PnL vs benchmarks
- Feature importance bar chart
- IC heatmap
- Regime/state timeline
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "text.color":        "#e6edf3",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def plot_cumulative_pnl(
    ml_returns:        pd.Series,
    always_in_returns: pd.Series | None = None,
    symbol:            str = "",
    output_dir:        str = "outputs",
) -> str:
    """Plot cumulative PnL for ML strategy vs Always-In benchmark."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/pnl_{symbol}.png"

    fig, ax = plt.subplots(figsize=(12, 5))

    cum_ml = (1 + ml_returns).cumprod()
    ax.plot(cum_ml.index, cum_ml.values, color="#58a6ff", linewidth=1.5,
            label=f"ML Carry [{symbol}]")

    if always_in_returns is not None:
        cum_ai = (1 + always_in_returns).cumprod()
        ax.plot(cum_ai.index, cum_ai.values, color="#f0883e", linewidth=1.0,
                linestyle="--", alpha=0.7, label="Always-In")

    ax.axhline(1.0, color="#8b949e", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalised)")
    ax.set_title(f"Cumulative PnL – {symbol}", color="#e6edf3")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved PnL chart → %s", path)
    return path


def plot_feature_importance(
    importances: pd.Series,
    symbol:      str = "",
    top_n:       int = 15,
    output_dir:  str = "outputs",
) -> str:
    """Horizontal bar chart of top-N feature importances."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/feature_importance_{symbol}.png"

    top = importances.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#58a6ff" if v >= top.median() else "#8b949e" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance – {symbol}", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved feature importance chart → %s", path)
    return path


def plot_ic_heatmap(
    ic_series:  pd.DataFrame,
    symbol:     str = "",
    output_dir: str = "outputs",
) -> str:
    """Heatmap: IC over time per feature (features × time)."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/ic_heatmap_{symbol}.png"

    # Resample to weekly for readability
    weekly = ic_series.resample("W").mean()
    data = weekly.T  # features as rows

    fig, ax = plt.subplots(figsize=(14, max(6, len(data) * 0.4)))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xticks(range(0, len(weekly.index), max(1, len(weekly.index) // 10)))
    ax.set_xticklabels(
        [str(d)[:10] for d in weekly.index[::max(1, len(weekly.index) // 10)]],
        rotation=45, fontsize=7,
    )
    plt.colorbar(im, ax=ax, label="IC (Spearman)")
    ax.set_title(f"IC Heatmap – {symbol}", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved IC heatmap → %s", path)
    return path


def plot_state_timeline(
    state_log:  pd.DataFrame,
    hl_rates:   pd.Series,
    symbol:     str = "",
    output_dir: str = "outputs",
) -> str:
    """
    Dual-axis plot: HL funding rate (line) + state overlay (background shading).
    HOLDING periods highlighted in green.
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/state_timeline_{symbol}.png"

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(hl_rates.index, hl_rates.values * 100, color="#8b949e",
            linewidth=0.8, alpha=0.8, label="HL rate (%)")

    # Shade HOLDING periods
    holding = state_log[state_log["state"] == "HOLDING"]
    if not holding.empty:
        holding_idx = holding.index
        prev = None
        for ts in holding_idx:
            if prev is None:
                start = ts
            prev = ts
        # Shade contiguous blocks
        in_block = False
        block_start = None
        for ts, row in state_log.iterrows():
            if row["state"] == "HOLDING" and not in_block:
                block_start = ts
                in_block = True
            elif row["state"] != "HOLDING" and in_block:
                ax.axvspan(block_start, ts, alpha=0.15, color="#3fb950")
                in_block = False
        if in_block and block_start:
            ax.axvspan(block_start, state_log.index[-1], alpha=0.15, color="#3fb950")

    ax.axhline(0, color="#30363d", linewidth=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Funding Rate (%)")
    ax.set_title(f"State Timeline – {symbol} (green = HOLDING)", color="#e6edf3")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved state timeline → %s", path)
    return path
