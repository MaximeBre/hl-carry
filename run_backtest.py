"""
run_backtest.py – Full backtest pipeline

Loads OOF predictions from run_training.py (or fetches fresh data),
runs the carry strategy simulation per asset, and outputs:
  - outputs/backtest_{symbol}.json   (metrics + trade log)
  - outputs/dashboard.html           (updated dashboard)
  - PNG charts per asset

Usage:
  python run_backtest.py [--symbol BTC] [--days 730]

STOPS and reports findings if any asset has:
  - Sharpe < 1.0 on the full series, OR
  - trade_count < 30
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STOP_SHARPE   = 1.0    # below this → STOP before execution layer
STOP_TRADES   = 30     # below this → STOP (not statistically valid)


def backtest_asset(symbol: str, days: int) -> dict:
    from config import (
        SYMBOL_MAP, DERIBIT_SYMBOLS, PAPER_CAPITAL, PREDICTION_HORIZON,
    )
    from data.hyperliquid import get_hl_funding_history, snap_hl_funding_to_1h_grid
    from data.binance import get_binance_klines_1h
    from backtest.simulator import backtest_carry, always_in_benchmark, never_in_benchmark
    from backtest.metrics import compute_metrics, print_metrics, compare_with_benchmarks
    from analysis.plots import (
        plot_cumulative_pnl, plot_state_timeline,
    )

    logger.info("══════════════════════════════════════════")
    logger.info("  Backtest  [symbol=%s, days=%d]", symbol, days)
    logger.info("══════════════════════════════════════════")

    binance_symbol = SYMBOL_MAP[symbol]
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3_600_000

    # ── Load OOF predictions ──────────────────────────────────────────────
    pred_path = Path(f"outputs/predictions_{symbol}.parquet")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No OOF predictions for {symbol}. Run run_training.py first."
        )

    pred_df     = pd.read_parquet(pred_path)
    predictions = pred_df["prediction"]
    actuals     = pred_df["target"]
    hl_rates    = pred_df["hl_rate_1h"]

    # Price series (fetch fresh for vol circuit breaker)
    logger.info("Fetching price data for circuit breaker…")
    try:
        price_data = get_binance_klines_1h(binance_symbol, start_ms, now_ms)
        prices = price_data.set_index("timestamp")["close"].reindex(predictions.index).ffill()
    except Exception as exc:
        logger.warning("Price fetch failed: %s – using flat prices", exc)
        prices = pd.Series(1.0, index=predictions.index)

    # ── Run backtest ──────────────────────────────────────────────────────
    logger.info("Running backtest simulation…")
    result = backtest_carry(
        predictions     = predictions,
        actuals         = actuals,
        hl_rates        = hl_rates,
        prices          = prices,
        initial_capital = PAPER_CAPITAL,
    )

    # ── Benchmarks ────────────────────────────────────────────────────────
    ai_returns = always_in_benchmark(hl_rates)
    ni_returns = never_in_benchmark(hl_rates)

    comparison = compare_with_benchmarks(
        result.returns, ai_returns, ni_returns, result.trades
    )
    ml_metrics = comparison["ml"]
    print_metrics(ml_metrics, symbol=symbol)

    # ── Threshold checks ──────────────────────────────────────────────────
    sharpe_ok = ml_metrics["sharpe_ratio"] >= STOP_SHARPE
    trades_ok = ml_metrics["trade_count"]  >= STOP_TRADES

    if not sharpe_ok:
        logger.error(
            "STOP: [%s] Sharpe=%.3f < %.1f – review model before execution layer",
            symbol, ml_metrics["sharpe_ratio"], STOP_SHARPE,
        )
    if not trades_ok:
        logger.error(
            "STOP: [%s] trade_count=%d < %d – not statistically valid",
            symbol, ml_metrics["trade_count"], STOP_TRADES,
        )

    # ── Charts ────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    try:
        plot_cumulative_pnl(result.returns, ai_returns, symbol=symbol)
        plot_state_timeline(result.state_log, hl_rates, symbol=symbol)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    # ── Regime split ──────────────────────────────────────────────────────
    logger.info("Computing regime split…")
    ret_30d   = prices.pct_change(720).reindex(predictions.index)
    bull_mask = ret_30d > 0
    bear_mask = ~bull_mask

    regime_results = {}
    for regime, mask in [("bull", bull_mask), ("bear", bear_mask)]:
        if mask.sum() < 500:
            logger.info("  [%s] %s regime: too few rows (%d) – skip", symbol, regime, mask.sum())
            continue
        r_result = backtest_carry(
            predictions[mask], actuals[mask],
            hl_rates[mask],    prices[mask],
        )
        r_metrics = compute_metrics(r_result.returns, r_result.trades)
        regime_results[regime] = {
            "sharpe":   round(r_metrics["sharpe_ratio"], 4),
            "trades":   r_metrics["trade_count"],
            "return":   round(r_metrics["total_return"], 5),
        }
        ok = r_metrics["sharpe_ratio"] >= 0
        logger.info("  %s %s regime: Sharpe=%.3f  trades=%d",
                    "✓" if ok else "✗", regime.upper(),
                    r_metrics["sharpe_ratio"], r_metrics["trade_count"])

    # ── Serialise trades ──────────────────────────────────────────────────
    trade_dicts = []
    for t in result.trades:
        trade_dicts.append({
            "entry_time":        str(t.entry_time),
            "exit_time":         str(t.exit_time),
            "entry_prediction":  round(t.entry_prediction, 7),
            "exit_prediction":   round(t.exit_prediction, 7),
            "position_size":     round(t.position_size, 4),
            "funding_pnl":       round(t.funding_pnl, 6),
            "fee_cost":          round(t.fee_cost, 6),
            "basis_pnl":         round(t.basis_pnl, 6),
            "total_pnl":         round(t.total_pnl, 6),
            "hours_held":        t.hours_held,
            "exit_reason":       t.exit_reason,
        })

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "symbol":         symbol,
        "backtest_date":  datetime.now(timezone.utc).isoformat(),
        "n_periods":      len(result.returns),
        "metrics":        {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in ml_metrics.items() if k != "exit_reasons"},
        "exit_reasons":   ml_metrics.get("exit_reasons", {}),
        "benchmarks": {
            "always_in": {
                "sharpe": round(comparison["always_in"]["sharpe_ratio"], 4),
                "return": round(comparison["always_in"]["total_return"], 5),
            },
            "never_in": {"sharpe": 0.0, "return": 0.0},
        },
        "regime_results": regime_results,
        "trades":         trade_dicts,
        "deploy_ready":   sharpe_ok and trades_ok,
    }

    out_path = f"outputs/backtest_{symbol}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Backtest results saved → %s", out_path)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="HL Carry Backtest")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all)")
    parser.add_argument("--days",   type=int, default=730, help="Days of history (default: 730)")
    args = parser.parse_args()

    from config import SYMBOLS
    symbols = [args.symbol] if args.symbol else SYMBOLS

    stop_flags: list[str] = []
    all_ok = True

    for sym in symbols:
        try:
            result = backtest_asset(sym, args.days)
            if not result.get("deploy_ready"):
                stop_flags.append(sym)
                all_ok = False
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            all_ok = False
        except Exception as exc:
            logger.error("[%s] Backtest failed: %s", sym, exc, exc_info=True)
            all_ok = False

    # ── Regenerate dashboard ──────────────────────────────────────────────
    try:
        from generate_dashboard import generate_dashboard
        generate_dashboard()
    except Exception as exc:
        logger.warning("Dashboard generation failed: %s", exc)

    # ── Final verdict ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════")
    if stop_flags:
        logger.error(
            "STOP – The following assets did NOT meet deployment criteria: %s\n"
            "  Review model and data before proceeding to execution layer.",
            stop_flags,
        )
    else:
        logger.info("ALL assets passed deployment criteria – ready for Phase 6 validation gate")
    logger.info("══════════════════════════════════════════")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
