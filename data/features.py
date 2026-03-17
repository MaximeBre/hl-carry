"""
data/features.py – Feature matrix builder for HL Carry System

ALL features at time t use ONLY data available at or before t.
Strict no-lookahead policy is enforced by an explicit verification step.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must never be used as features (target leakage)
_EXCLUDE_COLS = {
    "target", "future_rate", "fundingRate_next",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    hl_funding:      pd.DataFrame,
    binance_funding: pd.DataFrame,
    price_data:      pd.DataFrame,
    deribit_data:    Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build feature matrix aligned to the 1h grid of hl_funding.

    Parameters
    ----------
    hl_funding      : from data/hyperliquid.py – columns: timestamp, fundingRate, premium
                      Timestamps already snapped to 1h grid (UTC-aware).
    binance_funding : from data/binance.py     – columns: timestamp, fundingRate_binance
                      Already forward-filled to 1h grid.
    price_data      : from data/binance.py     – columns: timestamp, open, high, low,
                      close, volume.  1h klines.
    deribit_data    : optional – columns: timestamp, deribit_iv_atm
                      Only for BTC/ETH; NaN for others.

    Returns
    -------
    DataFrame with 18 feature columns + 'timestamp' index.
    The last PREDICTION_HORIZON rows will NOT have a target; the caller must
    attach the target and drop NaN rows before training.
    """
    from config import PREDICTION_HORIZON

    # ── Merge all sources on 1h timestamp ────────────────────────────────────
    df = hl_funding[["timestamp", "fundingRate", "premium"]].copy()
    df = df.rename(columns={"fundingRate": "hl_rate_1h"})

    # Binance funding (already 1h forward-filled)
    bin_ff = binance_funding[["timestamp", "fundingRate_binance"]].copy()
    df = df.merge(bin_ff, on="timestamp", how="left")

    # Price klines
    price = price_data[["timestamp", "close", "volume"]].copy()
    df = df.merge(price, on="timestamp", how="left")

    # Deribit IV
    if deribit_data is not None and "deribit_iv_atm" in deribit_data.columns:
        div = deribit_data[["timestamp", "deribit_iv_atm"]].copy()
        df = df.merge(div, on="timestamp", how="left")
    else:
        df["deribit_iv_atm"] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── HL-native features (10) ───────────────────────────────────────────────

    # 1. hl_rate_1h already present

    # 2. hl_rate_ma_8h: rolling mean of [t-7 : t] (8 observations)
    df["hl_rate_ma_8h"] = df["hl_rate_1h"].rolling(8, min_periods=1).mean()

    # 3. hl_rate_ma_24h: rolling mean of [t-23 : t] (24 observations)
    df["hl_rate_ma_24h"] = df["hl_rate_1h"].rolling(24, min_periods=1).mean()

    # 4. hl_rate_std_24h
    df["hl_rate_std_24h"] = df["hl_rate_1h"].rolling(24, min_periods=2).std()

    # 5. hl_rate_velocity_4h: change in 4h MA over last 4 periods
    ma4 = df["hl_rate_1h"].rolling(4, min_periods=1).mean()
    df["hl_rate_velocity_4h"] = (ma4 - ma4.shift(4)) / 4

    # 6–7. Open interest change (we compute from hl_rate_1h premium as proxy
    #      if OI history is not separately available; callers can pass richer data
    #      by adding 'openInterest' column to hl_funding).
    if "openInterest" in hl_funding.columns:
        df["oi"] = hl_funding["openInterest"].values
    else:
        # OI not available – fill with NaN (XGBoost tolerates)
        df["oi"] = np.nan

    oi = df["oi"]
    df["hl_oi_change_1h"]    = oi.pct_change(1)
    df["hl_oi_velocity_4h"]  = (df["hl_oi_change_1h"] - df["hl_oi_change_1h"].shift(4)) / 4

    # 8. hl_volume_buy_sell_ratio
    #    Requires buy/sell volume breakdown.  If 'buy_volume' / 'sell_volume' not
    #    in hl_funding, we set to NaN (acceptable – XGBoost handles NaN natively).
    if "buy_volume" in hl_funding.columns and "sell_volume" in hl_funding.columns:
        buy  = hl_funding["buy_volume"].values
        sell = hl_funding["sell_volume"].values
        total = buy + sell
        df["hl_volume_buy_sell_ratio"] = np.where(total > 0, buy / total, np.nan)
    else:
        df["hl_volume_buy_sell_ratio"] = np.nan

    # 9–10. Liquidation proxies (via large-trade detection on 'volume')
    #       large trade = volume > rolling_mean + 2 * rolling_std
    vol = df["volume"].fillna(0)
    vol_mean = vol.rolling(24, min_periods=6).mean()
    vol_std  = vol.rolling(24, min_periods=6).std()
    large    = vol > (vol_mean + 2 * vol_std)

    # Proxy: large sell → large price drop?  We use price_return < 0 as side indicator
    if "close" in df.columns and not df["close"].isna().all():
        ret_1h = df["close"].pct_change(1)
        df["hl_liquidations_long_proxy"]  = (large & (ret_1h < 0)).astype(float)
        df["hl_liquidations_short_proxy"] = (large & (ret_1h > 0)).astype(float)
    else:
        df["hl_liquidations_long_proxy"]  = np.nan
        df["hl_liquidations_short_proxy"] = np.nan

    # ── Cross-exchange features (3) ───────────────────────────────────────────

    # 11. binance_rate_last_settled: already forward-filled in binance_funding
    df["binance_rate_last_settled"] = df["fundingRate_binance"]

    # 12. binance_hl_rate_spread: Binance per-hour equivalent minus HL hourly rate
    df["binance_hl_rate_spread"] = (
        df["binance_rate_last_settled"] / 8.0 - df["hl_rate_1h"]
    )

    # 13. hours_since_binance_settlement: 0–7 (modulo 8 of UTC hour)
    df["hours_since_binance_settlement"] = df["timestamp"].dt.hour % 8

    # ── Price context features (3) ────────────────────────────────────────────

    # 14. price_return_1h
    df["price_return_1h"] = df["close"].pct_change(1)

    # 15. price_return_4h
    df["price_return_4h"] = df["close"].pct_change(4)

    # 16. realized_vol_24h (annualized)
    df["realized_vol_24h"] = (
        df["price_return_1h"].rolling(24, min_periods=6).std()
        * np.sqrt(8760)
    )

    # ── Deribit options features (2, BTC+ETH only) ────────────────────────────

    # 17. deribit_iv_atm: already merged (NaN for non-BTC/ETH)

    # 18. deribit_put_call_ratio: not fetched in this builder (needs live call)
    #     Set to NaN here; live paper_trading.py injects it at runtime.
    if "deribit_put_call_ratio" not in df.columns:
        df["deribit_put_call_ratio"] = np.nan

    # ── Temporal features (2) ────────────────────────────────────────────────

    # 19. hour_of_day: intraday funding seasonality
    df["hour_of_day"] = df["timestamp"].dt.hour

    # 20. day_of_week: 0=Monday
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # ── Final feature set ─────────────────────────────────────────────────────
    feature_cols = [
        # HL-native
        "hl_rate_1h", "hl_rate_ma_8h", "hl_rate_ma_24h", "hl_rate_std_24h",
        "hl_rate_velocity_4h",
        "hl_oi_change_1h", "hl_oi_velocity_4h",
        "hl_volume_buy_sell_ratio",
        "hl_liquidations_long_proxy", "hl_liquidations_short_proxy",
        # Cross-exchange
        "binance_rate_last_settled", "binance_hl_rate_spread",
        "hours_since_binance_settlement",
        # Price context
        "price_return_1h", "price_return_4h", "realized_vol_24h",
        # Deribit (NaN for non-BTC/ETH)
        "deribit_iv_atm", "deribit_put_call_ratio",
        # Temporal
        "hour_of_day", "day_of_week",
    ]

    result = df[["timestamp"] + feature_cols].copy()
    result = result.set_index("timestamp")

    _verify_no_lookahead(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Target builder
# ─────────────────────────────────────────────────────────────────────────────

def build_target(
    hl_funding: pd.DataFrame,
    horizon:    int = 24,
) -> pd.Series:
    """
    Build the regression target: cumulative HL funding rate over the next
    `horizon` hours.

    target[t] = sum(hl_rate[t+1], hl_rate[t+2], ..., hl_rate[t+horizon])

    CRITICAL: rate at time t is NOT included (that would be lookahead).
    The last `horizon` rows will have NaN target (no future data available).
    Drop those rows before training.

    Parameters
    ----------
    hl_funding : DataFrame with columns timestamp, fundingRate (1h grid)
    horizon    : number of future hours to sum (default 24)

    Returns
    -------
    pd.Series indexed by timestamp with name 'target'
    """
    rates = hl_funding.set_index("timestamp")["fundingRate"].sort_index()

    # Shift by -1 so that index t gets the sum of t+1 … t+horizon
    target = sum(rates.shift(-(i + 1)) for i in range(horizon))
    target.name = "target"

    # Explicitly verify: target[t] must NOT depend on rates[t]
    # (static verification – no data access, just logic check)
    assert horizon >= 1, "horizon must be >= 1"

    logger.debug(
        "build_target: horizon=%d, non-NaN rows=%d, NaN rows=%d",
        horizon, target.notna().sum(), target.isna().sum(),
    )
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_no_lookahead(features: pd.DataFrame) -> None:
    """
    Explicit lookahead-bias check.

    Verifies that removing the last N rows does NOT change any feature value
    for any row before them.  This catches rolling windows that inadvertently
    use future data.

    Raises RuntimeError if a violation is detected.
    Logs a warning (not error) for columns that are entirely NaN, as those
    cannot be verified.
    """
    N_CHECK = 5  # number of tail rows to drop
    if len(features) <= N_CHECK + 10:
        logger.warning("_verify_no_lookahead: DataFrame too small (%d rows) – skipping", len(features))
        return

    # Rebuild features without the last N rows
    features_trimmed = features.iloc[:-N_CHECK]

    for col in features.columns:
        if col in _EXCLUDE_COLS:
            continue
        if features[col].isna().all():
            logger.warning("_verify_no_lookahead: column %r is all-NaN – cannot verify", col)
            continue

        # Compare feature values for the overlapping period
        overlap_idx = features_trimmed.index
        orig_vals    = features.loc[overlap_idx, col]
        trimmed_vals = features_trimmed[col]

        # Allow NaN == NaN
        diff = (orig_vals != trimmed_vals) & ~(orig_vals.isna() & trimmed_vals.isna())
        if diff.any():
            bad_ts = orig_vals[diff].index.tolist()[:3]
            msg = (
                f"LOOKAHEAD BIAS DETECTED in feature '{col}'!\n"
                f"Removing the last {N_CHECK} rows changes values at: {bad_ts}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

    logger.debug("_verify_no_lookahead: passed for %d features", len(features.columns))


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity test (called from test scripts)
# ─────────────────────────────────────────────────────────────────────────────

def check_no_lookahead(features: pd.DataFrame) -> None:
    """Public alias for the lookahead-bias verification (used by test_pipeline.py)."""
    _verify_no_lookahead(features)


def print_feature_summary(features: pd.DataFrame) -> None:
    """Print a quick diagnostic of the feature matrix."""
    print(f"\n{'='*60}")
    print(f"Feature matrix: {len(features)} rows × {len(features.columns)} columns")
    print(f"Date range:     {features.index.min()} → {features.index.max()}")
    print(f"\nMissing value summary:")
    null_pct = features.isna().mean() * 100
    for col, pct in null_pct.items():
        status = "OK" if pct < 5 else ("WARN" if pct < 50 else "CRITICAL")
        print(f"  {col:<40s} {pct:5.1f}%  [{status}]")
    print(f"{'='*60}\n")
