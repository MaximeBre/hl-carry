"""
backtest/simulator.py – Carry strategy simulation

Delta-neutral carry: Long Binance Spot + Short HL Perp.
PnL source: hourly HL funding rate received as short.

CRITICAL rules (non-negotiable):
- Sharpe computed on FULL return series including 0-return FLAT periods
- Entry cost deducted at entry, exit cost at exit (4-leg round-trip split 50/50)
- basis_noise models imperfect delta-neutrality (Binance spot vs HL perp)
- Half-Kelly position sizing with MAX_POSITION_PCT cap
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    ENTRY_THRESHOLD, HOLD_THRESHOLD, MIN_HOLD_HOURS,
    TOTAL_ROUND_TRIP_FEE, MAX_POSITION_PCT, MAX_HOLD_HOURS,
    VOL_CIRCUIT_BREAKER, KELLY_FRACTION, BASIS_NOISE_STD,
    PAPER_CAPITAL, PERIODS_PER_YEAR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:        pd.Timestamp
    exit_time:         pd.Timestamp | None = None
    entry_prediction:  float = 0.0
    exit_prediction:   float = 0.0
    position_size:     float = 0.0
    funding_pnl:       float = 0.0
    fee_cost:          float = 0.0
    basis_pnl:         float = 0.0
    hours_held:        int   = 0
    exit_reason:       str   = ""

    @property
    def total_pnl(self) -> float:
        return self.funding_pnl + self.basis_pnl - self.fee_cost


@dataclass
class BacktestResult:
    returns:    pd.Series    # hourly returns (0 when FLAT)
    cum_pnl:    pd.Series    # cumulative PnL in USDT
    trades:     list[Trade]
    state_log:  pd.DataFrame  # timestamp, state, position_size, pred


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

def _half_kelly_size(
    predicted_rate: float,
    rate_std: float,
    portfolio_value: float,
) -> float:
    """
    Half-Kelly fraction based on predicted rate / rate std.
    Capped at MAX_POSITION_PCT of portfolio.

    kelly_fraction = (predicted_rate / rate_std) * KELLY_FRACTION
    position_size  = min(kelly * portfolio, MAX_POSITION_PCT * portfolio)
    """
    if rate_std <= 0 or predicted_rate <= 0:
        return 0.0
    kelly = (predicted_rate / rate_std) * KELLY_FRACTION
    kelly = max(0.0, min(kelly, 1.0))  # clamp to [0, 1]
    size = kelly * portfolio_value
    return min(size, MAX_POSITION_PCT * portfolio_value)


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest
# ─────────────────────────────────────────────────────────────────────────────

def backtest_carry(
    predictions:     pd.Series,    # index = UTC timestamp
    actuals:         pd.Series,    # actual cumulative rate (unused in sim, for comparison)
    hl_rates:        pd.Series,    # actual 1h HL funding rate (received as short)
    prices:          pd.Series,    # 1h close prices (for vol circuit breaker)
    config:          dict | None = None,
    initial_capital: float = PAPER_CAPITAL,
    rng_seed:        int = 42,
) -> BacktestResult:
    """
    Simulate the carry strategy using ML predictions.

    State machine:
    - FLAT  → HOLDING : prediction > ENTRY_THRESHOLD
    - HOLDING → FLAT  : prediction < HOLD_THRESHOLD AND hours_held >= MIN_HOLD_HOURS
    - HOLDING → FLAT  : realized_vol_24h > VOL_CIRCUIT_BREAKER (force exit)
    - HOLDING → FLAT  : hours_held >= MAX_HOLD_HOURS (stale position)

    Returns BacktestResult with full hourly return series.
    """
    rng = np.random.default_rng(rng_seed)

    # Align all series to predictions index
    timestamps = predictions.index
    hl_r   = hl_rates.reindex(timestamps).fillna(0.0)
    prices_ = prices.reindex(timestamps).ffill()
    preds  = predictions.copy()

    # Pre-compute realized_vol_24h for circuit breaker
    price_ret = prices_.pct_change(1)
    rvol_24h  = price_ret.rolling(24, min_periods=6).std() * np.sqrt(PERIODS_PER_YEAR)

    # Pre-compute rate std (rolling 24h) for Kelly sizing
    rate_std_24h = hl_r.rolling(24, min_periods=6).std().fillna(1e-6)

    # Initialise state
    state          = "FLAT"
    position_size  = 0.0
    hours_held     = 0
    portfolio_value = float(initial_capital)

    returns   : list[float]           = []
    cum_pnl   : list[float]           = [0.0]
    state_rows: list[dict]            = []
    trades    : list[Trade]           = []
    current_trade: Trade | None       = None

    for i, ts in enumerate(timestamps):
        pred    = float(preds.iloc[i])
        rate_1h = float(hl_r.iloc[i])
        rv      = float(rvol_24h.iloc[i]) if not np.isnan(rvol_24h.iloc[i]) else 0.0
        r_std   = float(rate_std_24h.iloc[i])

        hourly_return = 0.0
        exit_reason   = ""

        # ── Circuit breaker check ─────────────────────────────────────────
        circuit_break = rv > VOL_CIRCUIT_BREAKER and state == "HOLDING"
        stale_exit    = hours_held >= MAX_HOLD_HOURS and state == "HOLDING"

        if circuit_break:
            exit_reason = "vol_circuit_breaker"
        elif stale_exit:
            exit_reason = "max_hold"

        # ── State transitions ─────────────────────────────────────────────
        if state == "FLAT":
            if pred > ENTRY_THRESHOLD:
                # ENTER
                position_size = _half_kelly_size(pred, r_std, portfolio_value)
                if position_size > 0:
                    entry_fee = position_size * TOTAL_ROUND_TRIP_FEE / 2
                    hourly_return = -entry_fee / portfolio_value
                    portfolio_value += hourly_return * portfolio_value
                    state = "HOLDING"
                    hours_held = 1
                    current_trade = Trade(
                        entry_time=ts,
                        entry_prediction=pred,
                        position_size=position_size,
                        fee_cost=entry_fee,
                    )

        elif state == "HOLDING":
            # Funding PnL: receive the hourly HL rate as short
            basis_noise = float(rng.normal(0, BASIS_NOISE_STD))
            funding_pnl = position_size * (rate_1h + basis_noise)

            # Check exit conditions
            should_exit = (
                exit_reason != ""
                or (pred < HOLD_THRESHOLD and hours_held >= MIN_HOLD_HOURS)
            )

            if not should_exit:
                # HOLD: accrue funding
                hourly_return = funding_pnl / portfolio_value
                portfolio_value += funding_pnl
                hours_held += 1
                if current_trade is not None:
                    current_trade.funding_pnl += funding_pnl
                    current_trade.basis_pnl   += basis_noise * position_size
                    current_trade.hours_held   = hours_held
            else:
                # EXIT
                exit_fee = position_size * TOTAL_ROUND_TRIP_FEE / 2
                net_pnl  = funding_pnl - exit_fee
                hourly_return = net_pnl / portfolio_value
                portfolio_value += net_pnl

                if current_trade is not None:
                    current_trade.funding_pnl += funding_pnl
                    current_trade.basis_pnl   += basis_noise * position_size
                    current_trade.fee_cost    += exit_fee
                    current_trade.hours_held   = hours_held
                    current_trade.exit_time    = ts
                    current_trade.exit_prediction = pred
                    current_trade.exit_reason  = exit_reason or "prediction_below_hold"
                    trades.append(current_trade)
                    current_trade = None

                state         = "FLAT"
                position_size = 0.0
                hours_held    = 0

        returns.append(hourly_return)
        cum_pnl.append(cum_pnl[-1] + hourly_return * initial_capital)

        state_rows.append({
            "timestamp":    ts,
            "state":        state,
            "position_size": position_size,
            "prediction":   pred,
            "portfolio_value": portfolio_value,
        })

    # Close any open trade at end of data
    if state == "HOLDING" and current_trade is not None:
        current_trade.exit_time     = timestamps[-1]
        current_trade.exit_reason   = "end_of_data"
        current_trade.hours_held    = hours_held
        trades.append(current_trade)

    returns_series = pd.Series(returns, index=timestamps, name="hourly_return")
    cum_series     = pd.Series(cum_pnl[1:], index=timestamps, name="cum_pnl")
    state_df       = pd.DataFrame(state_rows).set_index("timestamp")

    return BacktestResult(
        returns=returns_series,
        cum_pnl=cum_series,
        trades=trades,
        state_log=state_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def always_in_benchmark(
    hl_rates: pd.Series,
    initial_capital: float = PAPER_CAPITAL,
) -> pd.Series:
    """
    Always-In benchmark: holds the carry position 100% of the time.
    Deducts round-trip fee once at start and once at end.
    Returns hourly return series.
    """
    returns = hl_rates.copy()
    returns.iloc[0]  -= TOTAL_ROUND_TRIP_FEE / 2
    returns.iloc[-1] -= TOTAL_ROUND_TRIP_FEE / 2
    return returns.rename("always_in_return")


def never_in_benchmark(hl_rates: pd.Series) -> pd.Series:
    """Never-In benchmark: all zeros (cash)."""
    return pd.Series(0.0, index=hl_rates.index, name="never_in_return")
