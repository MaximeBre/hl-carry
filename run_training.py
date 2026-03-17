"""
run_training.py – Full model training pipeline

Runs Walk-Forward training for all assets and saves:
  - models/saved/{symbol}_model.json     (XGBoost model)
  - models/saved/{symbol}_scaler.pkl     (StandardScaler)
  - outputs/training_report_{symbol}.json
  - outputs/predictions_{symbol}.parquet (OOF predictions + actuals for validate.py)

Usage:
  python run_training.py [--symbol BTC] [--days 730]
  python run_training.py --all --days 730
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_asset(symbol: str, days: int) -> dict:
    """
    Full training pipeline for one asset.
    Returns a training report dict.
    """
    from config import (
        SYMBOL_MAP, DERIBIT_SYMBOLS, ROLLING_WINDOW_DAYS, STEP_DAYS,
        N_FOLDS, EMBARGO_PERIODS, PREDICTION_HORIZON,
    )
    from data.hyperliquid import get_hl_funding_history, snap_hl_funding_to_1h_grid
    from data.binance import get_binance_funding_history, get_binance_klines_1h
    from data.deribit import get_deribit_historical_iv
    from data.features import build_features, build_target
    from models.validation import rolling_walk_forward
    from models.rate_predictor import train_model, predict
    from analysis.ic_analysis import compute_ic_series, compute_icir
    from analysis.plots import plot_feature_importance

    logger.info("══════════════════════════════════════════")
    logger.info("  Training  [symbol=%s, days=%d]", symbol, days)
    logger.info("══════════════════════════════════════════")

    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3_600_000
    binance_symbol = SYMBOL_MAP[symbol]

    # ── Fetch data ────────────────────────────────────────────────────────
    logger.info("Fetching HL funding history (%d days)…", days)
    hl_raw = get_hl_funding_history(symbol, start_ms, now_ms)
    hl_f   = snap_hl_funding_to_1h_grid(hl_raw)
    logger.info("  HL rows: %d", len(hl_f))

    logger.info("Fetching Binance funding history…")
    try:
        bn_f = get_binance_funding_history(binance_symbol, start_ms, now_ms)
    except Exception as exc:
        logger.warning("  Binance funding failed: %s – empty DF", exc)
        bn_f = pd.DataFrame(columns=["timestamp", "fundingRate_binance"])

    logger.info("Fetching Binance 1h klines…")
    try:
        price_data = get_binance_klines_1h(binance_symbol, start_ms, now_ms)
    except Exception as exc:
        logger.warning("  Binance klines failed: %s – empty DF", exc)
        price_data = pd.DataFrame()

    # Deribit historical (BTC/ETH only)
    deribit_data = None
    if symbol in DERIBIT_SYMBOLS:
        logger.info("Fetching Deribit historical IV…")
        try:
            deribit_data = get_deribit_historical_iv(symbol, start_ms, now_ms)
            logger.info("  Deribit rows: %d", len(deribit_data))
        except Exception as exc:
            logger.warning("  Deribit historical failed: %s – NaN features", exc)

    # ── Build feature matrix ──────────────────────────────────────────────
    logger.info("Building features…")
    features = build_features(hl_f, bn_f, price_data, deribit_data)
    target   = build_target(hl_f, horizon=PREDICTION_HORIZON)

    # Align features + target, drop NaN target (last PREDICTION_HORIZON rows)
    data = features.copy()
    data["target"] = target
    data = data.dropna(subset=["target"])

    # Drop warm-up NaN rows (first 24h)
    data = data.dropna(subset=[c for c in data.columns
                                if c not in ["deribit_iv_atm", "deribit_put_call_ratio", "target"]])

    logger.info("  Dataset after cleanup: %d rows (%d features)",
                len(data), len(features.columns))

    if len(data) < ROLLING_WINDOW_DAYS * 24 * 2:
        logger.warning("  Dataset may be too short for Walk-Forward (%d rows)", len(data))

    # ── IC Analysis ───────────────────────────────────────────────────────
    logger.info("Computing IC / ICIR…")
    feat_cols = [c for c in features.columns]
    try:
        ic_df = compute_ic_series(data[feat_cols], data["target"])
        icir  = {col: compute_icir(ic_df[col]) for col in feat_cols if col in ic_df.columns}
        top_features = sorted(icir, key=lambda c: abs(icir[c]), reverse=True)[:10]
        logger.info("  Top features by |ICIR|:")
        for f in top_features:
            logger.info("    %-40s ICIR = %+.4f", f, icir.get(f, 0.0))
        weak = [f for f in feat_cols if abs(icir.get(f, 0)) < 0.3]
        if weak:
            logger.warning("  Features with |ICIR| < 0.3 (consider removing): %s", weak)
    except Exception as exc:
        logger.warning("  IC analysis failed: %s", exc)
        icir = {}

    # ── Walk-Forward Training ─────────────────────────────────────────────
    logger.info("Running Walk-Forward training…")
    feat_only = data.drop(columns=["target"])
    wf_result = rolling_walk_forward(
        features    = feat_only,
        target      = data["target"],
        train_fn    = train_model,
        predict_fn  = predict,
        window_days = ROLLING_WINDOW_DAYS,
        step_days   = STEP_DAYS,
        n_folds     = N_FOLDS,
        embargo     = EMBARGO_PERIODS,
    )

    oof_preds    = wf_result.oof_predictions
    fold_metrics = wf_result.metrics_per_fold
    feat_imp     = wf_result.feature_importances.to_dict()
    best_model   = wf_result.best_model
    best_scaler  = wf_result.best_scaler

    # ── Validate fold signal distribution ────────────────────────────────
    from config import ENTRY_THRESHOLD
    pct_flat = (oof_preds < ENTRY_THRESHOLD).mean()
    pct_hold = (oof_preds >= ENTRY_THRESHOLD).mean()
    logger.info("  Signal distribution: FLAT=%.1f%%  ENTRY=%.1f%%",
                pct_flat * 100, pct_hold * 100)
    if pct_flat > 0.95:
        logger.error("  ✗ Model is trivially conservative (>95%% FLAT signals)")
    if pct_hold > 0.80:
        logger.error("  ✗ Model is trivially aggressive (>80%% ENTRY signals)")

    # ── Per-fold summary ──────────────────────────────────────────────────
    logger.info("  Per-fold metrics:")
    for i, fm in enumerate(fold_metrics):
        logger.info("    Fold %d: IC=%.4f  MAE=%.6f  RMSE=%.6f",
                    i + 1, fm.ic, fm.mae, fm.rmse)

    positive_ic_folds = sum(1 for fm in fold_metrics if fm.ic > 0)
    logger.info("  Folds with positive IC: %d / %d", positive_ic_folds, len(fold_metrics))

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs("models/saved", exist_ok=True)
    from models.rate_predictor import save_model
    save_model(best_model, best_scaler, symbol, "models/saved")
    logger.info("  Model saved → models/saved/%s_*", symbol)

    # ── Save OOF predictions (for validate.py) ────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    aligned = data.loc[oof_preds.index].copy()
    pred_df = pd.DataFrame({
        "prediction": oof_preds,
        "target":     aligned["target"],
        "hl_rate_1h": aligned["hl_rate_1h"],
        "price":      aligned["price_return_1h"],  # proxy; replace with actual price if needed
    })
    pred_df.to_parquet(f"outputs/predictions_{symbol}.parquet")
    logger.info("  OOF predictions saved → outputs/predictions_%s.parquet", symbol)

    # ── Feature importance plot ───────────────────────────────────────────
    try:
        feat_imp_series = pd.Series(feat_imp).sort_values(ascending=False)
        plot_feature_importance(feat_imp_series, symbol=symbol)
    except Exception as exc:
        logger.warning("  Feature importance plot failed: %s", exc)

    # ── Training report ───────────────────────────────────────────────────
    report = {
        "symbol":            symbol,
        "training_date":     datetime.now(timezone.utc).isoformat(),
        "n_rows":            len(data),
        "n_features":        len(feat_cols),
        "icir_top10":        {f: round(icir.get(f, 0), 5) for f in top_features} if icir else {},
        "weak_features":     weak if "weak" in dir() else [],
        "positive_ic_folds": positive_ic_folds,
        "n_folds":           len(fold_metrics),
        "fold_metrics":      [
            {"fold": fm.fold_idx, "ic": round(fm.ic, 5),
             "mae": round(fm.mae, 7), "rmse": round(fm.rmse, 7),
             "n_train": fm.n_train, "n_test": fm.n_test}
            for fm in fold_metrics
        ],
        "pct_flat":          round(pct_flat, 4),
        "pct_entry":         round(pct_hold, 4),
        "feature_importances": {k: round(float(v), 6) for k, v in feat_imp.items()},
    }

    report_path = f"outputs/training_report_{symbol}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Training report saved → %s", report_path)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="HL Carry Model Training")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all)")
    parser.add_argument("--days",   type=int, default=730, help="Days of history (default: 730)")
    parser.add_argument("--all",    action="store_true",   help="Train all SYMBOLS")
    args = parser.parse_args()

    from config import SYMBOLS
    symbols = SYMBOLS if (args.all or args.symbol is None) else [args.symbol]

    all_ok = True
    for sym in symbols:
        try:
            report = train_asset(sym, args.days)
            pos_ic = report["positive_ic_folds"]
            total  = report["n_folds"]
            if pos_ic < total // 2:
                logger.warning("[%s] Only %d/%d folds with positive IC – model may be weak",
                               sym, pos_ic, total)
        except Exception as exc:
            logger.error("[%s] Training failed: %s", sym, exc)
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
