"""
data/binance.py – Binance API wrapper for HL Carry System

Fetches:
  - Funding rate history (perpetual futures, 8-hourly) → forward-filled to 1h
  - Spot price (current)
  - 1h OHLCV klines (spot)

SYMBOL FORMAT:
  HL uses "BTC", "ETH", etc.
  Binance perp  uses "BTCUSDT", "ETHUSDT", etc.
  Binance spot  uses "BTCUSDT", "ETHUSDT", etc.
  Use SYMBOL_MAP from config.py to convert.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config import BINANCE_FUTURES, BINANCE_SPOT, SYMBOL_MAP

logger = logging.getLogger(__name__)

_SESSION = requests.Session()

# Maximum records per Binance request
_LIMIT = 1000


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _get(base_url: str, path: str, params: dict,
         retries: int = 3, backoff: float = 1.0) -> list | dict:
    url = f"{base_url}{path}"
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (1.5 ** attempt)
            logger.warning("Binance GET %s error (attempt %d/%d): %s – retry %.1fs",
                           path, attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def _ms_to_utc(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


# ── Funding rate ──────────────────────────────────────────────────────────────

def get_binance_funding_history(
    symbol: str,
    start_time: int,
    end_time: Optional[int] = None,
    limit: int = _LIMIT,
) -> pd.DataFrame:
    """
    Fetch Binance perpetual funding rate history for *symbol* (e.g. "BTCUSDT").
    Paginates until end_time or < limit records returned.

    Returns
    -------
    DataFrame: timestamp (UTC, 1h grid, forward-filled), fundingRate_binance
    The raw 8h rates are forward-filled across 8 hourly slots so that every
    1h row in the merged DataFrame has a valid 'last settled' Binance rate.

    CRITICAL: We use ONLY the LAST SETTLED 8h rate, NOT the predicted/next rate.
    This avoids lookahead bias.
    """
    if end_time is None:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[dict] = []
    cursor = start_time

    while cursor < end_time:
        params = {
            "symbol":    symbol,
            "startTime": cursor,
            "endTime":   end_time,
            "limit":     limit,
        }
        try:
            data = _get(BINANCE_FUTURES, "/fapi/v1/fundingRate", params)
        except Exception as exc:
            logger.error("get_binance_funding_history failed for %s: %s", symbol, exc)
            break

        if not data:
            break

        all_rows.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

        if len(data) < limit:
            break

        time.sleep(0.12)  # respect weight limits

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "fundingRate_binance"])

    df = pd.DataFrame(all_rows)
    df["timestamp_raw"] = df["fundingTime"].apply(lambda ms: _ms_to_utc(int(ms)))
    df["fundingRate_binance"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = (df[["timestamp_raw", "fundingRate_binance"]]
            .drop_duplicates("timestamp_raw")
            .sort_values("timestamp_raw")
            .reset_index(drop=True))

    # Forward-fill to 1h grid:
    # Binance settles at 00:00, 08:00, 16:00 UTC.  Each settled rate is
    # valid until the next settlement → repeat for 8 hourly slots.
    start_ts = df["timestamp_raw"].min().floor("h")
    end_ts   = _ms_to_utc(end_time).ceil("h")
    full_1h_index = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")

    df = df.set_index("timestamp_raw")
    df = df.reindex(full_1h_index)
    df.index.name = "timestamp"

    # Forward-fill: rate at settlement time propagates forward
    df["fundingRate_binance"] = df["fundingRate_binance"].ffill()

    df = df.reset_index()
    df = df.rename(columns={"index": "timestamp"})
    return df


def get_binance_spot_price(symbol: str) -> float:
    """Current Binance spot price for *symbol* (e.g. "BTCUSDT")."""
    data = _get(BINANCE_SPOT, "/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def get_binance_klines_1h(
    symbol: str,
    start_time: int,
    end_time: Optional[int] = None,
    limit: int = _LIMIT,
) -> pd.DataFrame:
    """
    Fetch 1-hour OHLCV klines from Binance Spot for *symbol*.

    Returns
    -------
    DataFrame: timestamp (UTC open time), open, high, low, close, volume
    Used for: price_return_1h, price_return_4h, realized_vol_24h features.
    """
    if end_time is None:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[list] = []
    cursor = start_time

    while cursor < end_time:
        params = {
            "symbol":    symbol,
            "interval":  "1h",
            "startTime": cursor,
            "endTime":   end_time,
            "limit":     limit,
        }
        try:
            data = _get(BINANCE_SPOT, "/api/v3/klines", params)
        except Exception as exc:
            logger.error("get_binance_klines_1h failed for %s: %s", symbol, exc)
            break

        if not data:
            break

        all_rows.extend(data)
        last_open_ts = int(data[-1][0])
        if last_open_ts <= cursor:
            break
        cursor = last_open_ts + 1  # next candle open

        if len(data) < limit:
            break

        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "_ignore",
    ])
    df["timestamp"] = df["open_time"].apply(lambda ms: _ms_to_utc(int(ms)))
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (df[["timestamp", "open", "high", "low", "close", "volume"]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    return df


def get_binance_hours_since_settlement(ts: pd.Timestamp) -> int:
    """
    Return how many full hours have elapsed since the last Binance 8h settlement
    for the given UTC timestamp.
    Binance settles at 00:00, 08:00, 16:00 UTC.
    Result is in [0, 7].
    """
    return ts.hour % 8
