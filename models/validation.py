"""
models/validation.py – Walk-Forward Validation with Purged K-Fold

Key design decisions (derived from System 1 lessons):
- ROLLING window (NOT expanding) – avoids old regime data biasing the model
- Embargo of 24h between train and test folds
- Per-fold StandardScaler (NEVER fit on full dataset)
- Collect OOF predictions, IC per fold, feature importances
"""

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from config import (
    ROLLING_WINDOW_DAYS, STEP_DAYS, N_FOLDS, EMBARGO_PERIODS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    fold_idx:    int
    window_start: pd.Timestamp
    window_end:   pd.Timestamp
    ic:          float        # Spearman IC between predictions and actuals
    mae:         float
    rmse:        float
    n_train:     int
    n_test:      int


@dataclass
class WalkForwardResult:
    oof_predictions:     pd.Series        # Index = timestamp
    oof_actuals:         pd.Series        # Index = timestamp
    metrics_per_fold:    list[FoldMetrics]
    feature_importances: pd.Series        # Averaged across all folds
    best_model:          object = None    # Model with highest IC across all folds
    best_scaler:         object = None    # Corresponding StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Purged K-Fold split generator
# ─────────────────────────────────────────────────────────────────────────────

def purged_kfold_split(
    n: int,
    n_folds: int = N_FOLDS,
    embargo: int = EMBARGO_PERIODS,
) -> Iterator[tuple[list[int], list[int]]]:
    """
    Purged K-Fold Cross-Validation for time-series data.

    - Folds are contiguous time blocks (no shuffling).
    - `embargo` observations are removed from the boundary between
      train and test to prevent near-future leakage.

    Yields
    ------
    (train_indices, test_indices) for each fold.
    """
    fold_size = n // n_folds

    for i in range(n_folds):
        test_start = i * fold_size
        test_end   = (i + 1) * fold_size if i < n_folds - 1 else n

        # Train: everything before test (minus embargo) + after test (minus embargo)
        # For a rolling-window setup this is called inside the outer window, so
        # "after test" portion exists within the window.
        train_before = list(range(0, max(0, test_start - embargo)))
        train_after  = list(range(min(n, test_end + embargo), n))
        train_indices = train_before + train_after
        test_indices  = list(range(test_start, test_end))

        if not train_indices or not test_indices:
            logger.warning("purged_kfold_split: fold %d has empty train or test – skipping", i)
            continue

        yield train_indices, test_indices


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_walk_forward(
    features: pd.DataFrame,    # index = UTC timestamp
    target:   pd.Series,       # index = UTC timestamp, same length as features
    train_fn,                  # Callable: (X_train, y_train, X_val, y_val) → (model, scaler)
    predict_fn,                # Callable: (model, scaler, X) → np.ndarray
    window_days:   int = ROLLING_WINDOW_DAYS,
    step_days:     int = STEP_DAYS,
    n_folds:       int = N_FOLDS,
    embargo:       int = EMBARGO_PERIODS,
) -> WalkForwardResult:
    """
    Rolling Window Walk-Forward Validation.

    Architecture:
    1. Cut a 90-day rolling window.
    2. Within the window, run Purged K-Fold (5 folds, 24h embargo).
    3. Per-fold: fit StandardScaler on TRAIN, transform both TRAIN and TEST.
    4. Collect OOF predictions for the TEST slice.
    5. Step the window forward 7 days.

    Returns WalkForwardResult with:
    - oof_predictions: all out-of-fold predictions
    - metrics_per_fold: per-fold IC, MAE, RMSE
    - feature_importances: averaged importance across folds
    """
    window_hours = window_days * 24
    step_hours   = step_days   * 24

    all_preds:    dict[pd.Timestamp, float] = {}
    all_actuals:  dict[pd.Timestamp, float] = {}
    fold_metrics: list[FoldMetrics] = []
    importance_acc: list[pd.Series] = []
    best_model_ref    = None
    best_scaler_ref   = None
    best_ic           = -np.inf

    data = features.join(target.rename("target"), how="inner")
    data = data.dropna(subset=["target"])
    timestamps = data.index

    n_total = len(data)
    if n_total < window_hours:
        raise ValueError(
            f"Not enough data: need {window_hours} rows, have {n_total}"
        )

    window_start_pos = 0
    fold_counter = 0

    while window_start_pos + window_hours <= n_total:
        window_end_pos = window_start_pos + window_hours
        window_data    = data.iloc[window_start_pos:window_end_pos]
        window_ts      = timestamps[window_start_pos:window_end_pos]
        n_window       = len(window_data)

        X_window = window_data.drop(columns=["target"]).values
        y_window = window_data["target"].values
        feat_names = window_data.drop(columns=["target"]).columns.tolist()

        for train_idx, test_idx in purged_kfold_split(n_window, n_folds, embargo):
            X_train, y_train = X_window[train_idx], y_window[train_idx]
            X_test,  y_test  = X_window[test_idx],  y_window[test_idx]

            # Per-fold scaler: fit ONLY on train
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)

            # Validation split (use last 15% of train as internal val)
            val_split = max(1, int(len(X_train_sc) * 0.85))
            X_tr, y_tr = X_train_sc[:val_split], y_train[:val_split]
            X_val, y_val = X_train_sc[val_split:], y_train[val_split:]

            try:
                model, _ = train_fn(X_tr, y_tr, X_val, y_val)
            except Exception as exc:
                logger.error("Training failed on fold %d: %s", fold_counter, exc)
                fold_counter += 1
                continue

            # Predict on test
            try:
                preds = predict_fn(model, scaler, X_window[test_idx])
            except Exception as exc:
                logger.error("Prediction failed on fold %d: %s", fold_counter, exc)
                fold_counter += 1
                continue

            # Store OOF predictions (keyed by timestamp)
            test_timestamps = window_ts[test_idx]
            for ts, pred, actual in zip(test_timestamps, preds, y_test):
                all_preds[ts]   = float(pred)
                all_actuals[ts] = float(actual)

            # Compute fold metrics
            ic_val, _ = spearmanr(preds, y_test)
            mae  = float(np.mean(np.abs(preds - y_test)))
            rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))

            fold_ic = float(ic_val) if not np.isnan(ic_val) else 0.0
            fold_metrics.append(FoldMetrics(
                fold_idx=fold_counter,
                window_start=window_ts[0],
                window_end=window_ts[-1],
                ic=fold_ic,
                mae=mae,
                rmse=rmse,
                n_train=len(train_idx),
                n_test=len(test_idx),
            ))
            # Track best model by IC
            if fold_ic > best_ic:
                best_ic         = fold_ic
                best_model_ref  = model
                best_scaler_ref = scaler

            # Feature importance (XGBoost / sklearn-compatible)
            try:
                imp = getattr(model, "feature_importances_", None)
                if imp is not None:
                    importance_acc.append(pd.Series(imp, index=feat_names))
            except Exception:
                pass

            fold_counter += 1

        window_start_pos += step_hours

    if not all_preds:
        raise RuntimeError("Walk-forward produced no predictions – check data size")

    oof_preds   = pd.Series(all_preds).sort_index()
    oof_actuals = pd.Series(all_actuals).sort_index()

    avg_importance = (
        pd.concat(importance_acc, axis=1).mean(axis=1).sort_values(ascending=False)
        if importance_acc else pd.Series(dtype=float)
    )

    # Summary printout
    ics = [m.ic for m in fold_metrics]
    logger.info(
        "Walk-Forward complete: %d folds, IC mean=%.4f ± %.4f",
        len(fold_metrics), np.mean(ics), np.std(ics),
    )

    return WalkForwardResult(
        oof_predictions=oof_preds,
        oof_actuals=oof_actuals,
        metrics_per_fold=fold_metrics,
        feature_importances=avg_importance,
        best_model=best_model_ref,
        best_scaler=best_scaler_ref,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fold sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def check_signal_distribution(
    predictions: pd.Series,
    entry_threshold: float,
    hold_threshold: float,
) -> dict:
    """
    Verify that the model is not trivially conservative or aggressive.

    Returns dict with:
    - flat_pct:    fraction of predictions below hold_threshold
    - holding_pct: fraction of predictions above entry_threshold
    - valid:       True if neither extreme is > 80%

    A model that is >95% FLAT is doing nothing.
    A model that is >80% HOLDING has probably overfit.
    """
    flat_pct    = (predictions < hold_threshold).mean()
    holding_pct = (predictions > entry_threshold).mean()

    valid = flat_pct < 0.95 and holding_pct < 0.80
    if not valid:
        logger.warning(
            "Signal distribution warning: flat=%.1f%% holding=%.1f%%",
            flat_pct * 100, holding_pct * 100,
        )

    return {
        "flat_pct":    flat_pct,
        "holding_pct": holding_pct,
        "valid":       valid,
    }
