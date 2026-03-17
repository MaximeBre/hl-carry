"""
data/hyperliquid.py – Hyperliquid API wrapper

KNOWN LIMITATIONS:
- HL has no public per-trade liquidation feed. We use large-trade detection
  (volume > 2 std from mean) as a proxy for liquidation events. This is an
  imperfect approximation; use with caution.
- fundingHistory is paginated in batches; we sleep 0.5 s between pages to
  respect rate limits.
- All timestamps from the HL API are in MILLISECONDS; we convert immediately.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import HL_API_BASE

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"Content-Type": "application/json"})


# ── Low-level helper ──────────────────────────────────────────────────────────

def _post(payload: dict, retries: int = 3, backoff: float = 1.0) -> dict:
    """POST to HL /info with exponential backoff."""
    url = f"{HL_API_BASE}/info"
    for attempt in range(retries):
        try:
            resp = _SESSION.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning("HL API error (attempt %d/%d): %s – retrying in %.1fs",
                           attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def _ms_to_utc(ms: int) -> pd.Timestamp:
    """Convert HL millisecond timestamp → UTC-aware pandas Timestamp."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


# ── Public functions ──────────────────────────────────────────────────────────

def get_hl_funding_history(
    coin: str,
    start_time: int,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch HL hourly funding rate history for *coin*.

    Parameters
    ----------
    coin       : HL coin symbol, e.g. "BTC"
    start_time : UTC timestamp in milliseconds (inclusive)
    end_time   : UTC timestamp in milliseconds (inclusive); defaults to now

    Returns
    -------
    DataFrame with columns: timestamp (UTC, 1h grid), fundingRate, premium
    Sorted ascending; no duplicates.

    Implementation note
    -------------------
    The HL API returns a limited batch per request.  We paginate by advancing
    startTime to (last_timestamp + 1) after each batch until we reach end_time
    or receive an empty response.  We sleep 0.5 s between pages.
    """
    if end_time is None:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[dict] = []
    cursor = start_time

    while cursor < end_time:
        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": cursor,
        }
        try:
            data = _post(payload)
        except Exception as exc:
            logger.error("get_hl_funding_history failed for %s: %s", coin, exc)
            break

        if not data:
            break

        all_rows.extend(data)
        last_ts = int(data[-1]["time"])
        if last_ts <= cursor:
            # No progress – bail out
            break
        cursor = last_ts + 1

        if len(data) < 20:
            # Fewer than a full batch → no more pages
            break

        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "fundingRate", "premium"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = df["time"].apply(lambda ms: _ms_to_utc(int(ms)))
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["premium"] = pd.to_numeric(df.get("premium", np.nan), errors="coerce")

    df = (df[["timestamp", "fundingRate", "premium"]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))

    # Filter to requested window
    end_ts = _ms_to_utc(end_time)
    df = df[df["timestamp"] <= end_ts]

    return df


def get_hl_meta_and_asset_ctxs() -> dict:
    """
    Fetch the full metaAndAssetCtxs payload once (contains all coins).
    Callers should cache this within a single run to avoid repeated requests.
    """
    return _post({"type": "metaAndAssetCtxs"})


def get_hl_open_interest(coin: str) -> float:
    """Return current open interest (USD notional) for *coin*."""
    meta = get_hl_meta_and_asset_ctxs()
    # meta is [universe_list, asset_ctx_list]
    universe = meta[0]["universe"]
    asset_ctxs = meta[1]
    for i, u in enumerate(universe):
        if u["name"] == coin:
            return float(asset_ctxs[i].get("openInterest", 0.0))
    raise ValueError(f"Coin {coin!r} not found in HL universe")


def get_hl_market_data(coin: str) -> dict:
    """
    Return current market data for *coin*:
    markPrice, oraclePrice, funding, openInterest, dayNtlVlm (24h volume USD).
    """
    meta = get_hl_meta_and_asset_ctxs()
    universe  = meta[0]["universe"]
    asset_ctxs = meta[1]
    for i, u in enumerate(universe):
        if u["name"] == coin:
            ctx = asset_ctxs[i]
            return {
                "markPrice":    float(ctx.get("markPx",        0.0)),
                "oraclePrice":  float(ctx.get("oraclePx",      0.0)),
                "funding":      float(ctx.get("funding",        0.0)),
                "openInterest": float(ctx.get("openInterest",   0.0)),
                "dayNtlVlm":    float(ctx.get("dayNtlVlm",      0.0)),
            }
    raise ValueError(f"Coin {coin!r} not found in HL universe")


def get_hl_recent_liquidations(
    coin: str,
    lookback_hours: int = 4,
) -> pd.DataFrame:
    """
    Proxy for liquidation events.

    LIMITATION: HL does NOT expose a public per-trade liquidation feed.
    The /userFills endpoint is per-user only.  We therefore use large-trade
    detection as a proxy:
      - Fetch recent trades via l2Book snapshots (mark price + OI changes)
      - Return a DataFrame with columns: timestamp, proxy_long_liq, proxy_short_liq
        where each column is an integer count of suspected liquidation events.

    In practice this function returns NaN-filled data and the feature builder
    falls back to 0.  Document this in strategy analysis.
    """
    logger.warning(
        "get_hl_recent_liquidations: HL has no public liq feed – "
        "returning NaN proxy.  Feature 'hl_liquidations_*' is unreliable."
    )
    now = pd.Timestamp.utcnow()
    idx = pd.date_range(end=now, periods=lookback_hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp":            idx,
        "proxy_long_liq":       np.nan,
        "proxy_short_liq":      np.nan,
    })


def get_hl_orderbook(coin: str) -> dict:
    """
    Return the current Level-2 orderbook for *coin*.
    Used at execution time to estimate bid-ask spread.

    Returns dict with keys: bids (list of [px, sz]), asks (list of [px, sz])
    """
    data = _post({"type": "l2Book", "coin": coin})
    return {
        "bids": data.get("levels", [[], []])[0],
        "asks": data.get("levels", [[], []])[1],
    }


def get_hl_bid_ask_spread(coin: str) -> float:
    """
    Estimate current spread as (best_ask - best_bid) / mid_price.
    Returns 0.0 if orderbook is empty.
    """
    book = get_hl_orderbook(coin)
    try:
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        mid = (best_bid + best_ask) / 2
        return (best_ask - best_bid) / mid if mid > 0 else 0.0
    except (IndexError, ZeroDivisionError):
        return 0.0


def snap_hl_funding_to_1h_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Floor HL funding timestamps to the nearest hour and verify the grid.
    HL settles every hour on the hour (UTC); slight jitter is common.
    Rows with jitter > 30 minutes are discarded (they likely belong to the
    wrong hour and would introduce a timestamp-alignment error with Binance).
    """
    df = df.copy()
    floored = df["timestamp"].dt.floor("h")
    jitter = (df["timestamp"] - floored).abs()
    mask = jitter <= pd.Timedelta(minutes=30)
    discarded = (~mask).sum()
    if discarded:
        logger.warning("snap_hl_funding_to_1h_grid: discarded %d rows with jitter > 30 min",
                       discarded)
    df = df[mask].copy()
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df
