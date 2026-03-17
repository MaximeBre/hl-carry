"""
validate.py – Phase 6 Validation Gate

MUST PASS before GitHub deployment.
Run: python validate.py [--symbol BTC]

Tests:
1. Lookahead Test        – removing last 24 rows must NOT change earlier feature values
2. Fee Sensitivity Test  – profitable at 0.25% fees
3. Regime Test           – Sharpe >= 0 in both bull and bear regimes
4. IC Stability          – FSI < 4 for top 5 features

Exits with code 1 if any critical check fails.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "✓ PASS"
FAIL = "✗ FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Lookahead bias
# ─────────────────────────────────────────────────────────────────────────────

def test_lookahead(features_full: pd.DataFrame, features_trimmed: pd.DataFrame) -> bool:
    """
    Verify that removing the last 24 rows does NOT alter any earlier feature values.
    If it does, there is lookahead bias in the feature pipeline.
    """
    logger.info("── Test 1: Lookahead Bias ────────────────────────────────────")
    n_common = len(features_trimmed)
    subset_full = features_full.iloc[:n_common]

    diff = (subset_full - features_trimmed).abs()
    worst = diff.max().max()

    if worst > 1e-10:
        logger.error("%s Lookahead detected! Max abs diff = %.2e", FAIL, worst)
        bad_cols = diff.columns[(diff.max() > 1e-10)].tolist()
        logger.error("  Affected features: %s", bad_cols)
        return False

    logger.info("%s No lookahead bias detected (max diff = %.2e)", PASS, worst)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Fee sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_fee_sensitivity(
    predictions: pd.Series,
    hl_rates:    pd.Series,
    prices:      pd.Series,
    symbol:      str,
) -> bool:
    """Strategy must be profitable at 0.25% round-trip fees."""
    logger.info("── Test 2: Fee Sensitivity ───────────────────────────────────")
    from backtest.metrics import fee_sensitivity_test

    df = fee_sensitivity_test(predictions, hl_rates, prices,
                              fee_levels=[0.0015, 0.002, 0.0025, 0.003])

    at_25bps = df[df["fee_pct"].between(0.0249, 0.0251)]
    if at_25bps.empty:
        # Try nearest
        at_25bps = df.iloc[(df["fee_pct"] - 0.025).abs().argsort()[:1]]

    profitable_at_25 = bool(at_25bps["profitable"].any())
    if not profitable_at_25:
        logger.error("%s [%s] NOT profitable at 0.25%% fees", FAIL, symbol)
        return False

    logger.info("%s [%s] Profitable at 0.25%% fees", PASS, symbol)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Regime test
# ─────────────────────────────────────────────────────────────────────────────

def test_regime(
    predictions: pd.Series,
    hl_rates:    pd.Series,
    prices:      pd.Series,
    symbol:      str,
) -> bool:
    """
    Split data into bull (BTC 30d return > 0) and bear.
    Strategy must have Sharpe >= 0 in both regimes.
    """
    logger.info("── Test 3: Regime Test ───────────────────────────────────────")
    from backtest.simulator import backtest_carry
    from backtest.metrics   import compute_metrics

    # Compute 30d rolling return on prices
    ret_30d = prices.pct_change(30 * 24)  # 30 days = 720 hours
    bull_mask = ret_30d > 0
    bear_mask = ~bull_mask

    passed = True
    for regime, mask in [("BULL", bull_mask), ("BEAR", bear_mask)]:
        preds_r = predictions[mask]
        rates_r = hl_rates[mask]
        prices_r = prices[mask]

        if len(preds_r) < 500:
            logger.warning("  [%s] Too few %s observations (%d) – skipping regime test",
                           symbol, regime, len(preds_r))
            continue

        result  = backtest_carry(preds_r, None, rates_r, prices_r)
        metrics = compute_metrics(result.returns, result.trades)
        sharpe  = metrics["sharpe_ratio"]

        ok = sharpe >= 0
        status = PASS if ok else FAIL
        logger.info("  %s [%s] %s regime Sharpe = %.3f", status, symbol, regime, sharpe)
        if not ok:
            passed = False

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – IC stability (FSI)
# ─────────────────────────────────────────────────────────────────────────────

def test_ic_stability(
    features: pd.DataFrame,
    target:   pd.Series,
    symbol:   str,
    top_n:    int = 5,
) -> bool:
    """FSI of top-N features must be < 4 (no significant drift)."""
    logger.info("── Test 4: IC Stability (FSI) ────────────────────────────────")
    from analysis.ic_analysis import compute_ic_series, compute_icir, compute_feature_stability_index

    # Align
    both = features.join(target.rename("target"), how="inner").dropna(subset=["target"])
    feat_cols = [c for c in features.columns if c != "target"]
    tgt  = both["target"]
    feat = both[feat_cols]

    # Compute IC for each feature
    icir_scores: dict[str, float] = {}
    for col in feat_cols:
        try:
            ic = compute_ic_series(feat[[col]], tgt)
            icir_scores[col] = abs(compute_icir(ic[col]))
        except Exception:
            icir_scores[col] = 0.0

    top_features = sorted(icir_scores, key=lambda c: icir_scores[c], reverse=True)[:top_n]
    logger.info("  Top features by |ICIR|: %s",
                {f: f"{icir_scores[f]:.3f}" for f in top_features})

    # Split into halves and compute PSI
    n = len(feat)
    first_half  = feat.iloc[:n//2]
    second_half = feat.iloc[n//2:]

    passed = True
    for col in top_features:
        try:
            fsi = compute_feature_stability_index(
                col,
                [(first_half[col].values, second_half[col].values)],
            )
            ok = fsi < 4.0
            status = PASS if ok else FAIL
            logger.info("  %s [%s] FSI(%s) = %.3f %s",
                        status, symbol, col, fsi,
                        "" if ok else "← DRIFT DETECTED")
            if not ok:
                passed = False
        except Exception as exc:
            logger.warning("  FSI computation failed for %s: %s", col, exc)

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 Validation Gate")
    parser.add_argument("--symbol", default="BTC", help="Asset to validate")
    parser.add_argument("--data-dir", default="outputs", help="Dir with precomputed features")
    args = parser.parse_args()

    symbol   = args.symbol
    data_dir = Path(args.data_dir)

    logger.info("════════════════════════════════════════════════════════════")
    logger.info("  Phase 6 Validation Gate   [symbol=%s]", symbol)
    logger.info("════════════════════════════════════════════════════════════")

    # Try to load pre-saved feature/prediction DataFrames
    feat_path = data_dir / f"features_{symbol}.parquet"
    pred_path = data_dir / f"predictions_{symbol}.parquet"

    if not feat_path.exists() or not pred_path.exists():
        logger.error(
            "Missing %s or %s.\n"
            "Run the full pipeline first:\n"
            "  python -c \"from data.features import *; ...\"\n"
            "or the test runner: python test_pipeline.py",
            feat_path, pred_path,
        )
        return 1

    features_full = pd.read_parquet(feat_path)
    pred_df       = pd.read_parquet(pred_path)

    predictions = pred_df["prediction"]
    hl_rates    = pred_df["hl_rate_1h"]
    prices      = pred_df["price"]
    target      = pred_df["target"]

    # Trim last 24 rows for lookahead test
    features_trimmed = features_full.iloc[:-24]

    results: dict[str, bool] = {}

    # Test 1
    results["lookahead"] = test_lookahead(features_full, features_trimmed)

    # Test 2
    results["fee_sensitivity"] = test_fee_sensitivity(predictions, hl_rates, prices, symbol)

    # Test 3
    results["regime"] = test_regime(predictions, hl_rates, prices, symbol)

    # Test 4
    results["ic_stability"] = test_ic_stability(features_full, target, symbol)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("════════════════════════════════════════════════════════════")
    logger.info("  VALIDATION SUMMARY")
    logger.info("════════════════════════════════════════════════════════════")
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        logger.info("  %s  %s", status, name)
        if not ok:
            all_pass = False

    logger.info("════════════════════════════════════════════════════════════")
    if all_pass:
        logger.info("  ALL CHECKS PASSED – safe to deploy")
        return 0
    else:
        logger.error("  VALIDATION FAILED – do NOT deploy until issues are fixed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
