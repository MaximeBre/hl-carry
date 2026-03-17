"""
backtest/metrics.py – Performance metrics for HL Carry System

CRITICAL: Sharpe is computed on the FULL hourly return series,
including 0-return FLAT periods. Do NOT compute only on in-market returns.
"""

import logging

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR
from backtest.simulator import Trade

logger = logging.getLogger(__name__)


def compute_metrics(
    returns: pd.Series,
    trades:  list[Trade],
) -> dict:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    returns : Full hourly return series (including 0-return FLAT periods).
              MUST cover the entire backtest period.
    trades  : List of completed Trade objects from the simulator.

    Returns
    -------
    dict with all metrics documented below.

    Statistical validity note:
    trade_count >= 30 is the minimum for meaningful statistics.
    Below that, Sharpe/Calmar/win_rate are unreliable – flagged via
    'statistically_valid'.
    """
    r = returns.dropna()
    n = len(r)

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return = float((1 + r).prod() - 1)
    n_years = n / PERIODS_PER_YEAR

    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Sharpe: mean / std * sqrt(periods_per_year)
    # CRITICAL: use FULL series including 0s (FLAT periods drag Sharpe appropriately)
    mean_r = float(r.mean())
    std_r  = float(r.std())
    sharpe = (mean_r / std_r * np.sqrt(PERIODS_PER_YEAR)) if std_r > 0 else 0.0

    # Sortino: downside deviation only
    downside = r[r < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 1e-9
    sortino  = (mean_r / down_std * np.sqrt(PERIODS_PER_YEAR)) if down_std > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Calmar: annualised return / abs(max drawdown)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else float("inf")

    # ── Trade-level metrics ───────────────────────────────────────────────────
    trade_count    = len(trades)
    completed      = [t for t in trades if t.exit_time is not None]

    if completed:
        pnls     = [t.total_pnl for t in completed]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls)

        gross_profit = sum(wins) if wins else 0.0
        gross_loss   = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss

        avg_hold_hours = np.mean([t.hours_held for t in completed])
        avg_pnl_per_trade = np.mean(pnls)
    else:
        win_rate = pnls = losses = wins = 0.0
        profit_factor = 0.0
        avg_hold_hours = 0.0
        avg_pnl_per_trade = 0.0

    # ── Time in market ────────────────────────────────────────────────────────
    time_in_market_pct = float((returns != 0).mean())

    # ── Statistical validity ──────────────────────────────────────────────────
    statistically_valid = trade_count >= 30

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    exit_reasons: dict[str, int] = {}
    for t in completed:
        reason = getattr(t, "exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    metrics = {
        # Return
        "total_return":        total_return,
        "annualized_return":   annualized_return,
        "sharpe_ratio":        sharpe,
        "sortino_ratio":       sortino,
        "calmar_ratio":        calmar,
        # Drawdown
        "max_drawdown":        max_drawdown,
        # Trades
        "trade_count":         trade_count,
        "win_rate":            win_rate,
        "profit_factor":       profit_factor,
        "avg_hold_hours":      avg_hold_hours,
        "avg_pnl_per_trade":   avg_pnl_per_trade,
        "exit_reasons":        exit_reasons,
        # Time in market
        "time_in_market_pct":  time_in_market_pct,
        # Validity
        "statistically_valid": statistically_valid,
        # Meta
        "n_hourly_periods":    n,
        "n_years":             n_years,
    }

    if not statistically_valid:
        logger.warning(
            "Only %d trades – metrics are NOT statistically significant (need >= 30)",
            trade_count,
        )

    return metrics


def print_metrics(metrics: dict, symbol: str = "") -> None:
    """Pretty-print metrics to stdout."""
    label = f" [{symbol}]" if symbol else ""
    valid_str = "✓ statistically valid" if metrics["statistically_valid"] else "✗ NOT statistically valid (<30 trades)"
    print(f"\n{'='*55}")
    print(f"  Performance Metrics{label}   {valid_str}")
    print(f"{'='*55}")
    print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
    print(f"  Ann. Return:         {metrics['annualized_return']*100:>8.2f}%")
    print(f"  Sharpe (full series):{metrics['sharpe_ratio']:>8.3f}")
    print(f"  Sortino:             {metrics['sortino_ratio']:>8.3f}")
    print(f"  Calmar:              {metrics['calmar_ratio']:>8.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Trade Count:         {metrics['trade_count']:>8d}")
    print(f"  Win Rate:            {metrics['win_rate']*100:>8.1f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:>8.3f}")
    print(f"  Avg Hold (hours):    {metrics['avg_hold_hours']:>8.1f}")
    print(f"  Time in Market:      {metrics['time_in_market_pct']*100:>8.1f}%")
    if metrics["exit_reasons"]:
        print(f"  Exit Reasons:        {metrics['exit_reasons']}")
    print(f"{'='*55}\n")


def compare_with_benchmarks(
    ml_returns:        pd.Series,
    always_in_returns: pd.Series,
    never_in_returns:  pd.Series,
    trades:            list[Trade],
) -> dict:
    """
    Compare ML strategy against Always-In and Never-In benchmarks.
    Returns dict with metrics for all three strategies.
    """
    from backtest.metrics import compute_metrics

    ml_metrics  = compute_metrics(ml_returns, trades)
    ai_metrics  = compute_metrics(always_in_returns, [])
    ni_metrics  = compute_metrics(never_in_returns,  [])

    print("\n── Strategy Comparison ──────────────────────────────")
    for label, m in [("ML Strategy", ml_metrics),
                     ("Always-In",   ai_metrics),
                     ("Never-In",    ni_metrics)]:
        print(f"  {label:<14} Sharpe={m['sharpe_ratio']:6.3f}  "
              f"Return={m['total_return']*100:7.2f}%  "
              f"MaxDD={m['max_drawdown']*100:6.2f}%  "
              f"Trades={m['trade_count']:3d}")

    return {
        "ml":       ml_metrics,
        "always_in": ai_metrics,
        "never_in":  ni_metrics,
    }


def fee_sensitivity_test(
    predictions:  pd.Series,
    hl_rates:     pd.Series,
    prices:       pd.Series,
    fee_levels:   list[float] | None = None,
) -> pd.DataFrame:
    """
    Run backtest at multiple fee levels to verify robustness.

    The strategy MUST be profitable at 0.25% to be considered robust.

    Returns DataFrame: fee_pct, sharpe, total_return, trade_count
    """
    from backtest.simulator import backtest_carry
    from backtest.metrics  import compute_metrics
    import config

    if fee_levels is None:
        fee_levels = [0.0015, 0.002, 0.0025, 0.003]

    rows = []
    for fee in fee_levels:
        # Temporarily override config fee
        config.TOTAL_ROUND_TRIP_FEE = fee
        result  = backtest_carry(predictions, None, hl_rates, prices)
        metrics = compute_metrics(result.returns, result.trades)
        rows.append({
            "fee_pct":       fee * 100,
            "sharpe":        metrics["sharpe_ratio"],
            "total_return":  metrics["total_return"],
            "trade_count":   metrics["trade_count"],
            "profitable":    metrics["total_return"] > 0,
        })

    # Restore default
    config.TOTAL_ROUND_TRIP_FEE = 0.002

    df = pd.DataFrame(rows)
    print("\n── Fee Sensitivity Test ─────────────────────────────")
    print(df.to_string(index=False))
    robust = df[df["fee_pct"] == 0.025]["profitable"].any()
    if not robust:
        logger.warning("STRATEGY NOT PROFITABLE AT 0.25%% FEES – review before deployment")
    return df
