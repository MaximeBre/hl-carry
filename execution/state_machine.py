"""
execution/state_machine.py – Position state management for live/paper trading

States: FLAT → ENTERING → HOLDING → EXITING → FLAT

Circuit breakers:
- realized_vol_24h > 5%        → force exit
- HL API unreachable 3× in row → freeze (no new positions)
- ADL indicator > 0.8          → force exit
- Position age > 7 days        → force exit (stale)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from config import (
    ENTRY_THRESHOLD, HOLD_THRESHOLD, MIN_HOLD_HOURS, MAX_HOLD_HOURS,
    VOL_CIRCUIT_BREAKER, ADL_DANGER_THRESHOLD, MAX_API_FAIL_COUNT,
)

logger = logging.getLogger(__name__)

State = Literal["FLAT", "ENTERING", "HOLDING", "EXITING"]


# ─────────────────────────────────────────────────────────────────────────────
# State data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    """Live state for a single asset."""
    symbol:          str
    state:           State         = "FLAT"
    position_size:   float         = 0.0      # USDT notional
    entry_time:      datetime | None = None
    entry_price:     float         = 0.0
    entry_prediction: float        = 0.0
    hours_held:      int           = 0
    api_fail_count:  int           = 0
    frozen:          bool          = False    # API-failure freeze
    last_update:     datetime | None = None
    history:         list[dict]    = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        return self.hours_held >= MAX_HOLD_HOURS

    @property
    def adl_breached(self) -> bool:
        return getattr(self, "_adl_level", 0.0) > ADL_DANGER_THRESHOLD

    def record_transition(self, from_state: str, to_state: str, reason: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "ts": ts, "from": from_state, "to": to_state, "reason": reason,
        }
        self.history.append(entry)
        logger.info("[%s] %s → %s (%s)", self.symbol, from_state, to_state, reason)


# ─────────────────────────────────────────────────────────────────────────────
# State machine step
# ─────────────────────────────────────────────────────────────────────────────

def step(
    pos:              PositionState,
    prediction:       float,
    realized_vol_24h: float,
    adl_level:        float = 0.0,
    api_ok:           bool  = True,
    position_size_fn  = None,   # callable(prediction, portfolio_value) → float
    portfolio_value:  float = 0.0,
) -> tuple[PositionState, str]:
    """
    Advance the state machine by one hourly step.

    Parameters
    ----------
    pos               : current PositionState (mutated in place and returned)
    prediction        : model's predicted cumulative 24h funding rate
    realized_vol_24h  : annualized realized vol over last 24h
    adl_level         : ADL indicator level [0, 1] (0.8 = danger zone)
    api_ok            : False if HL API call failed this hour
    position_size_fn  : optional callable for dynamic position sizing
    portfolio_value   : current portfolio value for sizing

    Returns
    -------
    (updated PositionState, action_taken)
    action_taken: "enter" | "hold" | "exit" | "retry_exit" | "wait" | "freeze" | "none"
    """
    pos.last_update = datetime.now(timezone.utc)
    pos._adl_level  = adl_level

    # ── API failure tracking ──────────────────────────────────────────────
    if not api_ok:
        pos.api_fail_count += 1
        if pos.api_fail_count >= MAX_API_FAIL_COUNT:
            pos.frozen = True
            logger.warning("[%s] HL API failed %d× – frozen (no new entries)",
                           pos.symbol, pos.api_fail_count)
            if pos.state == "FLAT":
                return pos, "freeze"
    else:
        pos.api_fail_count = 0
        pos.frozen = False

    # ── Circuit breakers ──────────────────────────────────────────────────
    vol_breaker = (realized_vol_24h > VOL_CIRCUIT_BREAKER
                   and pos.state in ("HOLDING", "ENTERING"))
    adl_breaker = (adl_level > ADL_DANGER_THRESHOLD
                   and pos.state in ("HOLDING", "ENTERING"))

    # ── State: FLAT ───────────────────────────────────────────────────────
    if pos.state == "FLAT":
        if pos.frozen:
            return pos, "freeze"

        if prediction > ENTRY_THRESHOLD:
            prev = pos.state
            pos.state        = "ENTERING"
            pos.entry_time   = datetime.now(timezone.utc)
            pos.hours_held   = 0
            pos.entry_prediction = prediction
            if position_size_fn is not None:
                pos.position_size = position_size_fn(prediction, portfolio_value)
            pos.record_transition(prev, "ENTERING", f"pred={prediction:.5f}>threshold")
            return pos, "enter"
        return pos, "none"

    # ── State: ENTERING ───────────────────────────────────────────────────
    elif pos.state == "ENTERING":
        if vol_breaker or adl_breaker:
            reason = "vol_circuit_breaker" if vol_breaker else "adl_breaker"
            pos.record_transition("ENTERING", "EXITING", reason)
            pos.state = "EXITING"
            return pos, "exit"

        # Assume fills confirmed (paper trading: instant)
        pos.record_transition("ENTERING", "HOLDING", "fills_confirmed")
        pos.state = "HOLDING"
        pos.hours_held = 1
        return pos, "hold"

    # ── State: HOLDING ────────────────────────────────────────────────────
    elif pos.state == "HOLDING":
        pos.hours_held += 1

        # Circuit breakers → force exit
        if vol_breaker:
            reason = f"vol={realized_vol_24h:.3f}>threshold"
            pos.record_transition("HOLDING", "EXITING", reason)
            pos.state = "EXITING"
            return pos, "exit"

        if adl_breaker:
            pos.record_transition("HOLDING", "EXITING", f"adl={adl_level:.2f}>threshold")
            pos.state = "EXITING"
            return pos, "exit"

        if pos.is_stale:
            pos.record_transition("HOLDING", "EXITING", f"stale pos age={pos.hours_held}h")
            pos.state = "EXITING"
            return pos, "exit"

        # Normal exit signal
        if prediction < HOLD_THRESHOLD and pos.hours_held >= MIN_HOLD_HOURS:
            pos.record_transition("HOLDING", "EXITING",
                                  f"pred={prediction:.5f}<hold_threshold after {pos.hours_held}h")
            pos.state = "EXITING"
            return pos, "exit"

        return pos, "hold"

    # ── State: EXITING ────────────────────────────────────────────────────
    elif pos.state == "EXITING":
        # Attempt to close both legs
        # In paper trading: always succeeds instantly
        # In live: exchange_api handles this; on failure → stay EXITING (retry next hour)
        prev = pos.state
        pos.state        = "FLAT"
        pos.position_size = 0.0
        pos.entry_time   = None
        pos.hours_held   = 0
        pos.record_transition(prev, "FLAT", "exit_confirmed")
        return pos, "none"

    return pos, "none"


# ─────────────────────────────────────────────────────────────────────────────
# State serialisation
# ─────────────────────────────────────────────────────────────────────────────

def state_to_dict(pos: PositionState) -> dict:
    """Convert PositionState to JSON-serialisable dict for state.json."""
    return {
        "symbol":           pos.symbol,
        "state":            pos.state,
        "position_size":    pos.position_size,
        "entry_time":       pos.entry_time.isoformat() if pos.entry_time else None,
        "entry_price":      pos.entry_price,
        "entry_prediction": pos.entry_prediction,
        "hours_held":       pos.hours_held,
        "api_fail_count":   pos.api_fail_count,
        "frozen":           pos.frozen,
        "last_update":      pos.last_update.isoformat() if pos.last_update else None,
        "history":          pos.history[-20:],  # keep last 20 transitions
    }


def state_from_dict(d: dict) -> PositionState:
    """Restore PositionState from a JSON dict (loaded from state.json)."""
    pos = PositionState(symbol=d["symbol"])
    pos.state           = d.get("state", "FLAT")
    pos.position_size   = d.get("position_size", 0.0)
    pos.entry_price     = d.get("entry_price", 0.0)
    pos.entry_prediction = d.get("entry_prediction", 0.0)
    pos.hours_held      = d.get("hours_held", 0)
    pos.api_fail_count  = d.get("api_fail_count", 0)
    pos.frozen          = d.get("frozen", False)
    pos.history         = d.get("history", [])

    et = d.get("entry_time")
    if et:
        pos.entry_time = datetime.fromisoformat(et)

    lu = d.get("last_update")
    if lu:
        pos.last_update = datetime.fromisoformat(lu)

    return pos
