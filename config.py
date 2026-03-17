"""
HL Carry System – Central Configuration
All constants live here. Import from other modules only.
"""

# ── Assets ────────────────────────────────────────────────────────────────────
SYMBOLS = ["BTC", "ETH", "SOL", "AVAX", "XRP"]
# DOGE / LINK excluded until HL liquidity confirmed

# Map from HL coin name to Binance USDT-perpetual symbol
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "AVAX": "AVAXUSDT",
    "XRP": "XRPUSDT",
}

# Assets that have Deribit options data
DERIBIT_SYMBOLS = {"BTC", "ETH"}

# ── Capital ───────────────────────────────────────────────────────────────────
PAPER_CAPITAL = 1000  # EUR / USDT
CAPITAL_SPLIT = {
    "binance_spot": 0.5,   # 50% long Binance spot
    "hl_margin":    0.5,   # 50% short HL perp margin
}

# ── Fee Model (4-leg round-trip) ──────────────────────────────────────────────
# Leg 1: Binance Spot open (taker)  0.075% with BNB discount
# Leg 2: HL Perp open   (taker)    0.010%
# Leg 3: Binance Spot close (taker) 0.075%
# Leg 4: HL Perp close  (taker)    0.010%
# Slippage estimate                 0.030%
FEES = {
    "binance_spot_maker":  0.00075,
    "binance_spot_taker":  0.001,
    "hl_perp_maker":       0.00002,   # 0.002%
    "hl_perp_taker":       0.0001,    # 0.010%
    "slippage":            0.0003,    # 0.030% conservative
}
# Conservative all-taker + slippage total round-trip cost
TOTAL_ROUND_TRIP_FEE = 0.002   # 0.20%

# ── ML / Prediction ───────────────────────────────────────────────────────────
PREDICTION_HORIZON = 24        # hours ahead to predict (cumulative funding)
MIN_HOLD_HOURS     = 24        # minimum hold after entry (fee amortisation)
ENTRY_THRESHOLD    = 0.002     # predicted cum_rate > 0.20% → enter
HOLD_THRESHOLD     = 0.001     # predicted cum_rate < 0.10% → exit

# ── Walk-Forward Validation ───────────────────────────────────────────────────
ROLLING_WINDOW_DAYS = 90       # 90 days = 2 160 hourly observations
STEP_DAYS           = 7        # step forward by 7 days each iteration
EMBARGO_PERIODS     = 24       # 24-hour embargo in Purged K-Fold
N_FOLDS             = 5        # purged k-fold folds within each window

# ── Execution / Risk ──────────────────────────────────────────────────────────
EXECUTION_INTERVAL_HOURS = 1
MAX_POSITION_PCT         = 0.25   # max 25% of portfolio per asset
ADL_DANGER_THRESHOLD     = 0.8    # pause if HL ADL indicator > 80%
MAX_HOLD_HOURS           = 168    # force-exit after 7 days (stale position)

# Circuit-breaker thresholds
VOL_CIRCUIT_BREAKER   = 0.05   # 5% realized_vol_24h → force exit
MAX_API_FAIL_COUNT    = 3      # consecutive HL API failures → freeze

# ── Optuna / XGBoost Search Space ────────────────────────────────────────────
OPTUNA_TRIALS = 50
EARLY_STOPPING_ROUNDS = 50

# ── API ───────────────────────────────────────────────────────────────────────
HL_API_BASE      = "https://api.hyperliquid.xyz"
BINANCE_FUTURES  = "https://fapi.binance.com"
BINANCE_SPOT     = "https://api.binance.com"
DERIBIT_API_BASE = "https://www.deribit.com/api/v2/public"

# ── Time ──────────────────────────────────────────────────────────────────────
PERIODS_PER_YEAR = 8760    # 24 * 365 hourly periods

# ── Basis Risk Model ──────────────────────────────────────────────────────────
# basis_noise in backtest: models imperfect delta-neutrality
# Modelled as normal(0, BASIS_NOISE_STD) per hour
BASIS_NOISE_STD = 0.0001   # 0.01% per hour (very small)

# ── Half-Kelly Position Sizing ────────────────────────────────────────────────
KELLY_FRACTION = 0.5       # Half-Kelly for conservatism
