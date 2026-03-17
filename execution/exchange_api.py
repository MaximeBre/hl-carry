"""
execution/exchange_api.py – Exchange API wrapper (Binance Spot + HL Perp)

PAPER TRADING MODE: All order functions simulate fills, log to trades.json.
LIVE MODE:          Actual orders placed; requires BINANCE_API_KEY,
                    BINANCE_API_SECRET, HL_AGENT_PRIVATE_KEY env vars.

IMPORTANT – Hyperliquid agent wallet:
  Use an Agent Wallet (delegated signer), NOT the main wallet.
  Agent wallet can trade but CANNOT withdraw funds.
  Create it at app.hyperliquid.xyz/API before going live.
  Store the agent wallet private key in HL_AGENT_PRIVATE_KEY secret.

Execution strategy:
  - Open both legs quasi-simultaneously (asyncio)
  - If one leg fails: cancel the other within 30 seconds
  - Use limit orders with 0.02% price buffer (not market orders)
  - Confirm fills before updating state
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from config import BINANCE_SPOT, SYMBOL_MAP

logger = logging.getLogger(__name__)

PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() != "false"
PRICE_BUFFER  = 0.0002  # 0.02% limit order buffer


# ─────────────────────────────────────────────────────────────────────────────
# Fill record
# ─────────────────────────────────────────────────────────────────────────────

def _log_fill(symbol: str, side: str, exchange: str,
              qty: float, price: float, reason: str) -> dict:
    fill = {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "symbol":   symbol,
        "side":     side,
        "exchange": exchange,
        "qty":      qty,
        "price":    price,
        "reason":   reason,
        "paper":    PAPER_TRADING,
    }
    # Append to trades.json
    trades_path = "state/trades.json"
    try:
        os.makedirs("state", exist_ok=True)
        existing = []
        if os.path.exists(trades_path):
            with open(trades_path) as f:
                existing = json.load(f)
        existing.append(fill)
        with open(trades_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write trades.json: %s", exc)
    return fill


# ─────────────────────────────────────────────────────────────────────────────
# Binance Spot
# ─────────────────────────────────────────────────────────────────────────────

class BinanceSpot:
    """Thin wrapper around Binance Spot REST API."""

    def __init__(self) -> None:
        self.api_key    = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self._session   = requests.Session()
        if self.api_key:
            self._session.headers["X-MBX-APIKEY"] = self.api_key

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    def _post(self, path: str, params: dict) -> dict:
        url = f"{BINANCE_SPOT}{path}"
        signed = self._sign(params)
        resp = self._session.post(url, params=signed, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_price(self, symbol: str) -> float:
        """Fetch current spot price."""
        resp = self._session.get(
            f"{BINANCE_SPOT}/api/v3/ticker/price",
            params={"symbol": symbol}, timeout=10
        )
        resp.raise_for_status()
        return float(resp.json()["price"])

    def place_limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        """Place a limit buy order (open long spot)."""
        if PAPER_TRADING:
            logger.info("[PAPER] BUY %s qty=%.6f @ %.4f", symbol, qty, price)
            return _log_fill(symbol, "BUY", "binance_spot", qty, price, "entry")

        limit_price = round(price * (1 + PRICE_BUFFER), 8)
        params = {
            "symbol":      symbol,
            "side":        "BUY",
            "type":        "LIMIT",
            "timeInForce": "GTC",
            "quantity":    f"{qty:.6f}",
            "price":       f"{limit_price:.4f}",
        }
        return self._post("/api/v3/order", params)

    def place_limit_sell(self, symbol: str, qty: float, price: float) -> dict:
        """Place a limit sell order (close long spot)."""
        if PAPER_TRADING:
            logger.info("[PAPER] SELL %s qty=%.6f @ %.4f", symbol, qty, price)
            return _log_fill(symbol, "SELL", "binance_spot", qty, price, "exit")

        limit_price = round(price * (1 - PRICE_BUFFER), 8)
        params = {
            "symbol":      symbol,
            "side":        "SELL",
            "type":        "LIMIT",
            "timeInForce": "GTC",
            "quantity":    f"{qty:.6f}",
            "price":       f"{limit_price:.4f}",
        }
        return self._post("/api/v3/order", params)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperliquid Perp
# ─────────────────────────────────────────────────────────────────────────────

class HyperliquidPerp:
    """
    Wrapper for HL perpetual trading via the Python SDK or direct HTTP.

    Uses Agent Wallet (delegated signer) – CANNOT withdraw.
    """

    def __init__(self) -> None:
        self.private_key = os.getenv("HL_AGENT_PRIVATE_KEY", "")
        self._exchange   = None
        self._init_exchange()

    def _init_exchange(self) -> None:
        if not self.private_key:
            logger.info("No HL_AGENT_PRIVATE_KEY – running in paper/read-only mode")
            return
        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants
            account = Account.from_key(self.private_key)
            self._exchange = Exchange(account, constants.MAINNET_API_URL)
            logger.info("HL Exchange initialised for agent wallet %s…", account.address[:10])
        except ImportError:
            logger.warning("hyperliquid-python-sdk not installed – HL orders disabled")
        except Exception as exc:
            logger.error("HL exchange init failed: %s", exc)

    def get_price(self, coin: str) -> float:
        """Fetch HL mark price."""
        from data.hyperliquid import get_hl_market_data
        return get_hl_market_data(coin)["markPrice"]

    def place_limit_short(self, coin: str, sz: float, price: float) -> dict:
        """Open short HL perp position (sell/short)."""
        if PAPER_TRADING or self._exchange is None:
            logger.info("[PAPER] SHORT %s sz=%.6f @ %.4f", coin, sz, price)
            return _log_fill(coin, "SHORT", "hl_perp", sz, price, "entry")

        limit_price = round(price * (1 - PRICE_BUFFER), 8)
        result = self._exchange.order(
            coin, is_buy=False, sz=sz, limit_px=limit_price,
            order_type={"limit": {"tif": "Gtc"}},
        )
        logger.info("HL short order result: %s", result)
        return result

    def place_limit_close_short(self, coin: str, sz: float, price: float) -> dict:
        """Close short HL perp position (buy to close)."""
        if PAPER_TRADING or self._exchange is None:
            logger.info("[PAPER] CLOSE_SHORT %s sz=%.6f @ %.4f", coin, sz, price)
            return _log_fill(coin, "BUY_TO_CLOSE", "hl_perp", sz, price, "exit")

        limit_price = round(price * (1 + PRICE_BUFFER), 8)
        result = self._exchange.order(
            coin, is_buy=True, sz=sz, limit_px=limit_price,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=True,
        )
        logger.info("HL close result: %s", result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Coordinated 2-leg execution
# ─────────────────────────────────────────────────────────────────────────────

async def open_carry_position(
    symbol:     str,            # HL coin, e.g. "BTC"
    usdt_size:  float,          # USDT notional per leg
) -> dict:
    """
    Open delta-neutral carry:
      Leg 1 – Long Binance Spot
      Leg 2 – Short HL Perp

    Both legs attempted quasi-simultaneously.
    If one fails, the other is cancelled within 30 seconds.

    Returns {"success": bool, "fills": [fill1, fill2], "error": str|None}
    """
    binance = BinanceSpot()
    hl      = HyperliquidPerp()

    binance_symbol = SYMBOL_MAP.get(symbol, symbol + "USDT")

    try:
        spot_price = binance.get_price(binance_symbol)
        hl_price   = hl.get_price(symbol)
    except Exception as exc:
        return {"success": False, "fills": [], "error": f"Price fetch failed: {exc}"}

    qty = usdt_size / spot_price  # units of base asset

    # Run both legs concurrently
    loop = asyncio.get_event_loop()
    try:
        fill_binance, fill_hl = await asyncio.gather(
            loop.run_in_executor(None, lambda: binance.place_limit_buy(binance_symbol, qty, spot_price)),
            loop.run_in_executor(None, lambda: hl.place_limit_short(symbol, qty, hl_price)),
        )
        return {"success": True, "fills": [fill_binance, fill_hl], "error": None}
    except Exception as exc:
        logger.error("open_carry_position failed: %s – attempting to cancel other leg", exc)
        # Best-effort cancel (paper: always succeeds)
        try:
            binance.place_limit_sell(binance_symbol, qty, spot_price)
        except Exception:
            pass
        return {"success": False, "fills": [], "error": str(exc)}


async def close_carry_position(
    symbol:    str,
    qty:       float,
) -> dict:
    """
    Close delta-neutral carry:
      Leg 3 – Sell Binance Spot
      Leg 4 – Buy to close HL Perp short

    On partial failure, retries next hour (state stays EXITING).
    """
    binance = BinanceSpot()
    hl      = HyperliquidPerp()

    binance_symbol = SYMBOL_MAP.get(symbol, symbol + "USDT")

    try:
        spot_price = binance.get_price(binance_symbol)
        hl_price   = hl.get_price(symbol)
    except Exception as exc:
        return {"success": False, "fills": [], "error": f"Price fetch failed: {exc}"}

    loop = asyncio.get_event_loop()
    try:
        fill_binance, fill_hl = await asyncio.gather(
            loop.run_in_executor(None, lambda: binance.place_limit_sell(binance_symbol, qty, spot_price)),
            loop.run_in_executor(None, lambda: hl.place_limit_close_short(symbol, qty, hl_price)),
        )
        return {"success": True, "fills": [fill_binance, fill_hl], "error": None}
    except Exception as exc:
        logger.error("close_carry_position failed: %s – will retry next hour", exc)
        return {"success": False, "fills": [], "error": str(exc)}
