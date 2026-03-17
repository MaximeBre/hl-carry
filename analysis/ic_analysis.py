"""
analysis/ic_analysis.py – Information Coefficient analysis for feature quality

Computes:
- Rolling Spearman IC per feature
- ICIR (IC Information Ratio)
- IC decay (signal persistence / optimal holding period)
- Feature Stability Index (PSI-based drift detection)

Interpretation:
  IC    > 0.05 → weak signal
  IC    > 0.10 → moderate (typically profitable)
  IC    > 0.20 → strong
  ICIR  > 1.0  → very good
  ICIR  > 0.5  → good
  ICIR  < 0.3  → weak → consider removing
  FSI   < 0.10 → stable
  FSI   0.10–0.25 → slight drift
  FSI   > 0.25 → unstable (retrain) – flag if > 2.0 (critical)
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MIN_VALID_OBS = 15  # minimum non-NaN pairs for valid IC computation
_EXCLUDE_COLS = {"target", "timestamp", "future_rate", "fundingRate_next"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding leakage and >70% NaN columns."""
    cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in _EXCLUDE_COLS:
            continue
        null_frac = df[col].isna().mean()
        if null_frac > 0.70:
            logger.warning("Skipping column %r (%.0f%% NaN)", col, null_frac * 100)
            continue
        cols.append(col)
    return cols


def _spearman_safe(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation with NaN handling. Returns NaN if < MIN_VALID_OBS pairs."""
    mask = x.notna() & y.notna()
    if mask.sum() < MIN_VALID_OBS:
        return float("nan")
    corr, _ = spearmanr(x[mask], y[mask])
    return float(corr)


# ─────────────────────────────────────────────────────────────────────────────
# IC series
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_series(
    features: pd.DataFrame,
    target:   pd.Series,
    window:   int = 90 * 24,   # 90 days of hourly data
) -> pd.DataFrame:
    """
    Compute rolling Spearman IC between each feature and the target.

    Parameters
    ----------
    features : feature DataFrame (index = UTC timestamp)
    target   : target Series (same index)
    window   : rolling window size in observations (default 90 days × 24h)

    Returns
    -------
    DataFrame: index = timestamp, columns = feature names, values = IC
    """
    feat_cols = _get_feature_cols(features)
    aligned   = features[feat_cols].join(target.rename("__target__"), how="inner")
    aligned   = aligned.dropna(subset=["__target__"])

    ic_rows: list[dict] = []

    for end_idx in range(window, len(aligned) + 1):
        window_data = aligned.iloc[end_idx - window : end_idx]
        ts = aligned.index[end_idx - 1]
        row = {"timestamp": ts}
        for col in feat_cols:
            row[col] = _spearman_safe(window_data[col], window_data["__target__"])
        ic_rows.append(row)

    ic_df = pd.DataFrame(ic_rows).set_index("timestamp")
    return ic_df


# ─────────────────────────────────────────────────────────────────────────────
# ICIR
# ─────────────────────────────────────────────────────────────────────────────

def compute_icir(ic_series: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICIR = mean(IC) / std(IC) for each feature.

    Returns
    -------
    DataFrame with columns: mean_ic, std_ic, icir, ic_positive_pct, quality
    Sorted descending by |icir|.
    """
    rows = []
    for col in ic_series.columns:
        s = ic_series[col].dropna()
        if len(s) < 5:
            continue
        mean_ic = float(s.mean())
        std_ic  = float(s.std())
        icir    = mean_ic / std_ic if std_ic > 0 else 0.0
        ic_pos  = float((s > 0).mean())

        if abs(icir) >= 1.0:
            quality = "very_good"
        elif abs(icir) >= 0.5:
            quality = "good"
        elif abs(icir) >= 0.3:
            quality = "moderate"
        else:
            quality = "weak"

        rows.append({
            "feature":       col,
            "mean_ic":       mean_ic,
            "std_ic":        std_ic,
            "icir":          icir,
            "ic_positive_pct": ic_pos,
            "quality":       quality,
        })

    summary = (pd.DataFrame(rows)
                 .set_index("feature")
                 .sort_values("icir", key=abs, ascending=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# IC decay
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_decay(
    features: pd.DataFrame,
    target_rate: pd.Series,      # raw HL funding rate (for computing lagged targets)
    max_lag: int = 48,
) -> pd.DataFrame:
    """
    Compute IC at each lag from 1 to max_lag for the most informative features.
    Identifies optimal holding period (lag where IC halves).

    Returns DataFrame: rows = feature, columns = lag_1 … lag_{max_lag}, halflife
    """
    feat_cols = _get_feature_cols(features)[:10]  # top 10 for performance
    results: dict[str, dict] = {}

    for col in feat_cols:
        lag_ics = {}
        for lag in range(1, max_lag + 1):
            lagged_target = target_rate.shift(-lag)
            combined = features[col].to_frame().join(
                lagged_target.rename("lag_target"), how="inner"
            ).dropna()
            if len(combined) < MIN_VALID_OBS:
                lag_ics[f"lag_{lag}"] = float("nan")
                continue
            ic = _spearman_safe(combined[col], combined["lag_target"])
            lag_ics[f"lag_{lag}"] = ic

        # Compute half-life: first lag where |IC| < 0.5 * |IC_lag_1|
        ic1 = abs(lag_ics.get("lag_1", 0.0) or 0.0)
        halflife = max_lag  # default if IC never halves
        for lag in range(1, max_lag + 1):
            ic_lag = abs(lag_ics.get(f"lag_{lag}", 0.0) or 0.0)
            if ic1 > 0 and ic_lag < 0.5 * ic1:
                halflife = lag
                break

        lag_ics["halflife"] = halflife
        results[col] = lag_ics

    return pd.DataFrame(results).T


# ─────────────────────────────────────────────────────────────────────────────
# Feature Stability Index (PSI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_stability_index(
    feature: pd.Series,
    n_windows: int = 6,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index over rolling windows.

    Splits the series into n_windows equal-length segments.
    Uses the first segment as the reference distribution.
    Computes PSI between reference and each subsequent segment.
    Returns the maximum PSI (worst-case drift).

    PSI < 0.10  → stable
    PSI 0.10–0.25 → slight drift (monitor)
    PSI > 0.25  → unstable (retrain model)
    PSI > 2.0   → critical / distribution has fundamentally shifted
    """
    series = feature.dropna()
    if len(series) < n_windows * n_bins:
        return float("nan")

    window_size = len(series) // n_windows
    windows = [series.iloc[i * window_size : (i + 1) * window_size]
               for i in range(n_windows)]

    # Reference distribution (first window)
    ref = windows[0]
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)

    def hist_pct(w: pd.Series) -> np.ndarray:
        counts, _ = np.histogram(w, bins=bins)
        pct = counts / len(w)
        pct = np.where(pct == 0, 1e-6, pct)  # avoid log(0)
        return pct

    ref_pct = hist_pct(ref)
    max_psi = 0.0

    for window in windows[1:]:
        if len(window) == 0:
            continue
        cur_pct = hist_pct(window)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        max_psi = max(max_psi, psi)

    return max_psi


# ─────────────────────────────────────────────────────────────────────────────
# Full IC report pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ic_report(
    features: pd.DataFrame,
    target:   pd.Series,
    raw_rate: pd.Series,
    symbol:   str,
    output_dir: str = "outputs",
) -> dict:
    """
    Run the complete IC analysis and write CSV outputs.

    Outputs:
    - outputs/ic_report_{symbol}.csv       – Full IC time series
    - outputs/ic_summary_{symbol}.csv      – ICIR, halflife, quality label
    - outputs/fsi_report_{symbol}.csv      – Feature Stability Index

    Returns dict with summary data for dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_cols = _get_feature_cols(features)

    # 1. Rolling IC series
    logger.info("Computing IC series for %s…", symbol)
    ic_series = compute_ic_series(features, target)
    ic_series.to_csv(f"{output_dir}/ic_report_{symbol}.csv")

    # 2. ICIR summary
    icir_summary = compute_icir(ic_series)
    icir_summary.to_csv(f"{output_dir}/ic_summary_{symbol}.csv")

    # Print top / bottom features
    print(f"\n── IC Analysis: {symbol} ─────────────────────────")
    print("Top 5 features by |ICIR|:")
    print(icir_summary.head(5).to_string())
    print("\nBottom 5 features by |ICIR|:")
    print(icir_summary.tail(5).to_string())

    # Flag weak features
    weak = icir_summary[icir_summary["quality"] == "weak"].index.tolist()
    if weak:
        logger.warning("Weak features (|ICIR| < 0.3) for %s: %s", symbol, weak)

    # 3. FSI
    logger.info("Computing Feature Stability Index for %s…", symbol)
    fsi_rows = []
    for col in feat_cols:
        fsi = compute_feature_stability_index(features[col])
        status = "stable"
        if not np.isnan(fsi):
            if fsi > 2.0:
                status = "critical"
                logger.warning("FSI CRITICAL: %s / %s = %.2f", symbol, col, fsi)
            elif fsi > 0.25:
                status = "unstable"
            elif fsi > 0.10:
                status = "slight_drift"
        fsi_rows.append({"feature": col, "fsi": fsi, "status": status})

    fsi_df = pd.DataFrame(fsi_rows).set_index("feature")
    fsi_df.to_csv(f"{output_dir}/fsi_report_{symbol}.csv")

    return {
        "icir_summary": icir_summary,
        "fsi":          fsi_df,
        "weak_features": weak,
    }
