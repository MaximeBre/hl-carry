"""
execution/paper_trading.py – Hourly execution loop

Called every hour by GitHub Actions.
Manages per-asset state, fetches latest data, runs ML predictions,
applies the state machine, and commits updated state to git.

Error handling policy:
- HL API failure:      log warning, skip hour, increment api_fail_count
- Binance API failure: use last known values (stale but OK for features)
- Deribit failure:     set deribit features to NaN (XGBoost handles natively)
- Model not found:     stay FLAT, log error
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path when run as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SYMBOLS, DERIBIT_SYMBOLS, PAPER_CAPITAL, PREDICTION_HORIZON,
    ENTRY_THRESHOLD, HOLD_THRESHOLD, MAX_POSITION_PCT,
    KELLY_FRACTION,
)
from data.hyperliquid import (
    get_hl_market_data, snap_hl_funding_to_1h_grid, get_hl_funding_history,
)
from data.binance import (
    get_binance_funding_history, get_binance_klines_1h, get_binance_spot_price,
)
from data.deribit import get_deribit_iv, get_deribit_put_call_ratio
from data.features import build_features
from models.rate_predictor import load_model, predict
from execution.state_machine import (
    PositionState, step, state_to_dict, state_from_dict,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)

STATE_PATH   = Path("state/state.json")
PERF_LOG     = Path("outputs/performance_log.csv")
MODEL_DIR    = "models/saved"
LOOKBACK_H   = 72   # hours of data to fetch for feature computation


# ─────────────────────────────────────────────────────────────────────────────
# State I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict[str, PositionState]:
    """Load per-asset states from state/state.json."""
    if not STATE_PATH.exists():
        logger.info("No state.json found – initialising fresh state")
        return {s: PositionState(symbol=s) for s in SYMBOLS}

    with open(STATE_PATH) as f:
        raw = json.load(f)

    states = {}
    for symbol in SYMBOLS:
        d = raw.get("positions", {}).get(symbol)
        if d:
            states[symbol] = state_from_dict(d)
        else:
            states[symbol] = PositionState(symbol=symbol)
    return states


def save_state(
    states:          dict[str, PositionState],
    portfolio_value: float,
    run_count:       int,
    extra:           dict | None = None,
) -> None:
    """Persist state to state/state.json."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_update":     datetime.now(timezone.utc).isoformat(),
        "portfolio_value": portfolio_value,
        "run_count":       run_count,
        "positions":       {s: state_to_dict(p) for s, p in states.items()},
    }
    if extra:
        payload.update(extra)
    with open(STATE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("State saved to %s", STATE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Performance logging
# ─────────────────────────────────────────────────────────────────────────────

def log_performance(
    ts:              pd.Timestamp,
    portfolio_value: float,
    predictions:     dict[str, float],
    actions:         dict[str, str],
) -> None:
    """Append one row to the hourly performance log."""
    PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp":       ts.isoformat(),
        "portfolio_value": portfolio_value,
        **{f"pred_{s}":    predictions.get(s, float("nan")) for s in SYMBOLS},
        **{f"action_{s}":  actions.get(s, "")               for s in SYMBOLS},
    }
    df_new = pd.DataFrame([row])
    if PERF_LOG.exists():
        df_new.to_csv(PERF_LOG, mode="a", header=False, index=False)
    else:
        df_new.to_csv(PERF_LOG, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch for one asset
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_data(symbol: str) -> dict | None:
    """
    Fetch last LOOKBACK_H hours of data for *symbol*.

    Returns dict with:
      hl_funding, binance_funding, price_data, deribit_data, market_data
    or None if HL API is unavailable.
    """
    from config import SYMBOL_MAP
    now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - LOOKBACK_H * 3_600_000

    # HL funding history (CRITICAL – if this fails, skip hour)
    try:
        hl_funding = get_hl_funding_history(symbol, start_ms, now_ms)
        hl_funding = snap_hl_funding_to_1h_grid(hl_funding)
        if hl_funding.empty:
            raise ValueError("Empty HL funding response")
    except Exception as exc:
        logger.error("HL funding fetch failed for %s: %s", symbol, exc)
        return None

    # HL market data (OI, vol, mark price)
    try:
        market_data = get_hl_market_data(symbol)
    except Exception as exc:
        logger.warning("HL market data fetch failed for %s: %s", symbol, exc)
        market_data = {}

    # Binance funding (non-critical – use stale if unavailable)
    binance_symbol = SYMBOL_MAP.get(symbol, symbol + "USDT")
    try:
        binance_funding = get_binance_funding_history(binance_symbol, start_ms, now_ms)
    except Exception as exc:
        logger.warning("Binance funding failed for %s: %s – using empty DF", symbol, exc)
        binance_funding = pd.DataFrame(columns=["timestamp", "fundingRate_binance"])

    # Binance 1h klines
    try:
        price_data = get_binance_klines_1h(binance_symbol, start_ms, now_ms)
    except Exception as exc:
        logger.warning("Binance klines failed for %s: %s – using empty DF", symbol, exc)
        price_data = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Deribit IV (BTC + ETH only; gracefully skip others)
    deribit_data = None
    if symbol in DERIBIT_SYMBOLS:
        try:
            iv  = get_deribit_iv(symbol)
            pcr = get_deribit_put_call_ratio(symbol)
            now_ts = pd.Timestamp.utcnow().floor("h").tz_localize("UTC")
            deribit_data = pd.DataFrame([{
                "timestamp":            now_ts,
                "deribit_iv_atm":       iv,
                "deribit_put_call_ratio": pcr,
            }])
        except Exception as exc:
            logger.warning("Deribit data failed for %s: %s – NaN features", symbol, exc)

    return {
        "hl_funding":     hl_funding,
        "binance_funding": binance_funding,
        "price_data":     price_data,
        "deribit_data":   deribit_data,
        "market_data":    market_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction for one asset
# ─────────────────────────────────────────────────────────────────────────────

def predict_for_asset(symbol: str, data: dict) -> float | None:
    """
    Build features for the latest hour and return the predicted
    cumulative 24h funding rate.
    Returns None if the model file is missing.
    """
    try:
        model, scaler = load_model(symbol, MODEL_DIR)
    except FileNotFoundError:
        logger.error("No trained model for %s – staying FLAT", symbol)
        return None

    try:
        features = build_features(
            hl_funding=data["hl_funding"],
            binance_funding=data["binance_funding"],
            price_data=data["price_data"],
            deribit_data=data["deribit_data"],
        )
    except Exception as exc:
        logger.error("Feature build failed for %s: %s", symbol, exc)
        return None

    if features.empty:
        logger.warning("Empty feature matrix for %s", symbol)
        return None

    # Use the most recent row
    X_latest = features.iloc[[-1]].values
    try:
        pred = float(predict(model, scaler, X_latest)[0])
    except Exception as exc:
        logger.error("Prediction failed for %s: %s", symbol, exc)
        return None

    logger.info("[%s] prediction=%.5f", symbol, pred)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_size_fn(portfolio_value: float):
    def size_fn(prediction: float, pv: float) -> float:
        from config import MAX_POSITION_PCT, KELLY_FRACTION
        kelly = min(max(prediction / 0.002, 0.0), 1.0) * KELLY_FRACTION
        return min(kelly * pv, MAX_POSITION_PCT * pv)
    return size_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main hourly run
# ─────────────────────────────────────────────────────────────────────────────

def run_hourly() -> None:
    """
    Main entry point – called every hour by GitHub Actions.

    Steps:
    1. Load state from state/state.json
    2. Fetch latest data per asset
    3. Build features + predict
    4. Apply state machine
    5. (Paper) simulate fills + log PnL
    6. Save updated state.json
    """
    now = pd.Timestamp.utcnow().floor("h").tz_localize("UTC") \
          if not pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow().floor("h")

    logger.info("── Hourly run started: %s ──────────────────────", now.isoformat())

    # Load state
    states = load_state()
    portfolio_value = PAPER_CAPITAL  # TODO: track cumulatively from perf log

    if PERF_LOG.exists():
        try:
            perf_df = pd.read_csv(PERF_LOG)
            if not perf_df.empty:
                portfolio_value = float(perf_df["portfolio_value"].iloc[-1])
        except Exception:
            pass

    run_count_path = Path("state/run_count.txt")
    run_count = 0
    if run_count_path.exists():
        try:
            run_count = int(run_count_path.read_text().strip())
        except Exception:
            pass
    run_count += 1
    run_count_path.write_text(str(run_count))

    predictions: dict[str, float] = {}
    actions:     dict[str, str]   = {}

    for symbol in SYMBOLS:
        logger.info("Processing %s…", symbol)

        # Fetch data
        data = fetch_latest_data(symbol)
        api_ok = data is not None

        if not api_ok:
            logger.warning("Skipping %s this hour (data fetch failed)", symbol)
            actions[symbol] = "skip_api_fail"
            # Still step state machine so api_fail_count increments
            pred = None
        else:
            pred = predict_for_asset(symbol, data)

        predictions[symbol] = pred if pred is not None else float("nan")

        # Realized vol for circuit breaker
        rvol = 0.0
        if api_ok and data and not data["price_data"].empty:
            try:
                closes = data["price_data"]["close"].pct_change(1)
                rvol = float(closes.std() * np.sqrt(8760))
            except Exception:
                pass

        # HL ADL (not exposed publicly; default 0)
        adl_level = 0.0

        # Step state machine
        pos = states[symbol]
        size_fn = _make_size_fn(portfolio_value)
        pos, action = step(
            pos,
            prediction=pred if pred is not None else 0.0,
            realized_vol_24h=rvol,
            adl_level=adl_level,
            api_ok=api_ok,
            position_size_fn=size_fn,
            portfolio_value=portfolio_value,
        )
        states[symbol] = pos
        actions[symbol] = action

        # Paper trading: simulate PnL
        if action in ("hold",) and api_ok and data:
            latest_hl_rate = 0.0
            if not data["hl_funding"].empty:
                latest_hl_rate = float(data["hl_funding"]["fundingRate"].iloc[-1])
            funding_pnl = pos.position_size * latest_hl_rate
            portfolio_value += funding_pnl
            logger.info("[%s] funding_pnl=%.4f USDT (rate=%.6f)", symbol, funding_pnl, latest_hl_rate)

    # Persist
    save_state(states, portfolio_value, run_count)
    log_performance(now, portfolio_value, predictions, actions)

    logger.info("── Hourly run complete. Portfolio=%.2f USDT ──", portfolio_value)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_hourly()
