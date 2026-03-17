"""
data/deribit.py – Deribit options data for BTC and ETH

IMPORTANT: Deribit data is only available for BTC and ETH.
For SOL, AVAX, XRP → set deribit_iv_atm = NaN, deribit_put_call_ratio = NaN.
XGBoost handles NaN features natively.

KNOWN LIMITATIONS:
- Deribit historical IV via public API is limited in depth.
  We use DVOL index when available; otherwise fall back to realized volatility
  computed from price data.  This limitation is clearly documented below and
  in the feature builder.
- All Deribit API failures are caught and logged; features are set to NaN
  rather than silently returning zeros.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import DERIBIT_API_BASE

logger = logging.getLogger(__name__)

_SESSION = requests.Session()

_CURRENCY_MAP = {
    "BTC": "BTC",
    "ETH": "ETH",
}


# ── Low-level helper ──────────────────────────────────────────────────────────

def _get(path: str, params: dict, retries: int = 3) -> dict:
    url = f"{DERIBIT_API_BASE}/{path}"
    import time
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                logger.error("Deribit GET %s failed: %s", path, exc)
                return {}
            time.sleep(1.5 ** attempt)
    return {}


# ── Current IV (snapshot) ─────────────────────────────────────────────────────

def get_deribit_iv(currency: str = "BTC") -> float:
    """
    Return ATM implied volatility for the nearest perpetual option.

    Uses BTC-PERPETUAL or ETH-PERPETUAL ticker 'mark_iv' field as a proxy
    for ATM IV.  Returns NaN on any failure.
    """
    perp_map = {"BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL"}
    instrument = perp_map.get(currency)
    if instrument is None:
        return float("nan")

    data = _get("ticker", {"instrument_name": instrument})
    try:
        result = data.get("result", {})
        iv = result.get("mark_iv")
        if iv is not None:
            return float(iv) / 100.0  # convert percent → fraction
    except Exception as exc:
        logger.warning("get_deribit_iv(%s): %s", currency, exc)
    return float("nan")


def get_deribit_put_call_ratio(currency: str = "BTC") -> float:
    """
    Compute put/call ratio from open interest.
    Returns sum(put OI) / sum(call OI) for all listed options.
    Returns NaN on any failure.
    """
    data = _get("get_book_summary_by_currency",
                {"currency": currency, "kind": "option"})
    try:
        results = data.get("result", [])
        put_oi = sum(float(r["open_interest"]) for r in results
                     if r.get("instrument_name", "").endswith("-P"))
        call_oi = sum(float(r["open_interest"]) for r in results
                      if r.get("instrument_name", "").endswith("-C"))
        if call_oi > 0:
            return put_oi / call_oi
    except Exception as exc:
        logger.warning("get_deribit_put_call_ratio(%s): %s", currency, exc)
    return float("nan")


# ── Historical IV ─────────────────────────────────────────────────────────────

def get_deribit_historical_iv(
    currency: str,
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    """
    Fetch historical ATM implied volatility aligned to 1h timestamps.

    Attempts to use Deribit DVOL index (hourly volatility index).
    DVOL is available for BTC only via:
      GET /public/get_tradingview_chart_data
      instrument_name=DVOL, resolution=60

    FALLBACK: If the DVOL request fails or returns insufficient data,
    we return an empty DataFrame with columns [timestamp, deribit_iv_atm].
    The feature builder then sets this column to NaN so XGBoost can handle it.

    LIMITATION: Deribit DVOL only covers BTC.  For ETH we currently have
    no hourly historical IV series.  ETH features will be NaN until a
    suitable data source is integrated.
    """
    dvol_instruments = {"BTC": "DVOL"}

    instrument = dvol_instruments.get(currency)
    if instrument is None:
        logger.info(
            "get_deribit_historical_iv: no DVOL for %s – returning empty", currency
        )
        return pd.DataFrame(columns=["timestamp", "deribit_iv_atm"])

    # Deribit expects timestamps in seconds
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_time // 1000,
        "end_timestamp":   end_time   // 1000,
        "resolution":      "60",          # 60-minute candles
    }
    data = _get("get_tradingview_chart_data", params)

    try:
        result = data.get("result", {})
        ticks  = result.get("ticks", [])
        closes = result.get("close", [])
        if not ticks:
            raise ValueError("Empty DVOL response")

        df = pd.DataFrame({
            "timestamp":      pd.to_datetime(ticks, unit="ms", utc=True),
            "deribit_iv_atm": [float(c) / 100.0 for c in closes],
        })
        df = (df.drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))
        logger.info("get_deribit_historical_iv: %d rows for %s", len(df), currency)
        return df

    except Exception as exc:
        logger.warning(
            "get_deribit_historical_iv(%s): DVOL failed (%s) – "
            "deribit_iv_atm will be NaN", currency, exc
        )
        return pd.DataFrame(columns=["timestamp", "deribit_iv_atm"])
