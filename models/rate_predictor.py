"""
models/rate_predictor.py – XGBoost rate predictor with Optuna hyperparameter search

Predicts: cumulative HL funding rate over next 24 hours (regression)
Loss:     Huber (robust to outlier rate spikes)
Tuning:   Optuna (50 trials, minimize validation MAE)

CRITICAL rules:
- Separate model instance per asset (different rate dynamics)
- Per-fold StandardScaler: fit on train, transform test (NEVER fit on full dataset)
- huber_slope tuned by Optuna (not fixed)
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import OPTUNA_TRIALS, EARLY_STOPPING_ROUNDS

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial output in production
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
    verbose: bool = False,
) -> tuple[XGBRegressor, StandardScaler]:
    """
    Train XGBoost regressor with Optuna hyperparameter optimisation.

    The StandardScaler was already applied by the caller (walk-forward loop).
    We receive pre-scaled arrays and return the fitted model.

    However, for the convenience of the standalone predict() function,
    we ALSO return a dummy identity scaler here (scaling already done).

    Parameters
    ----------
    X_train, y_train : scaled training arrays
    X_val,   y_val   : scaled validation arrays (for early stopping + Optuna)
    n_trials         : number of Optuna trials
    verbose          : print trial progress

    Returns
    -------
    (model, identity_scaler)
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "reg:pseudohubererror",
            "huber_slope":      trial.suggest_float("huber_slope", 0.0001, 0.005, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "tree_method":      "hist",
            "device":           "cpu",
            "verbosity":        0,
            "random_state":     42,
        }

        model = XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        mae   = float(np.mean(np.abs(preds - y_val)))
        return mae

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params.copy()
    best_params.update({
        "objective":    "reg:pseudohubererror",
        "tree_method":  "hist",
        "device":       "cpu",
        "verbosity":    0,
        "random_state": 42,
    })

    best_model = XGBRegressor(
        **best_params,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info(
        "train_model: best trial MAE=%.6f | params=%s",
        study.best_value, study.best_params,
    )

    # Return identity scaler (data already scaled by walk-forward loop)
    scaler = _IdentityScaler()
    return best_model, scaler


def train_model_with_scaling(
    X_train_raw: np.ndarray,
    y_train:     np.ndarray,
    X_val_raw:   np.ndarray,
    y_val:       np.ndarray,
    n_trials:    int = OPTUNA_TRIALS,
    verbose:     bool = False,
) -> tuple[XGBRegressor, StandardScaler]:
    """
    Full pipeline: fit StandardScaler on train, scale both sets, then train.

    USE THIS for standalone training (not inside walk-forward loop which
    handles scaling separately).

    Returns (model, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc   = scaler.transform(X_val_raw)   # ONLY transform, NOT fit_transform

    model, _ = train_model(X_train_sc, y_train, X_val_sc, y_val,
                           n_trials=n_trials, verbose=verbose)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model:  XGBRegressor,
    scaler: "StandardScaler | _IdentityScaler",
    X_new:  np.ndarray,
) -> np.ndarray:
    """
    Apply saved scaler then predict.

    If scaler is the identity (data already scaled), this is a no-op.
    """
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model:    XGBRegressor,
    scaler:   "StandardScaler | _IdentityScaler",
    symbol:   str,
    model_dir: str = "models/saved",
) -> None:
    """Save model + scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/model_{symbol}.ubj")
    with open(f"{model_dir}/scaler_{symbol}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved model + scaler for %s to %s/", symbol, model_dir)


def load_model(
    symbol:    str,
    model_dir: str = "models/saved",
) -> tuple[XGBRegressor, "StandardScaler"]:
    """Load model + scaler from disk."""
    model_path  = f"{model_dir}/model_{symbol}.ubj"
    scaler_path = f"{model_dir}/scaler_{symbol}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No scaler found at {scaler_path}")

    model = XGBRegressor()
    model.load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Loaded model + scaler for %s", symbol)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Identity scaler (for walk-forward which scales externally)
# ─────────────────────────────────────────────────────────────────────────────

class _IdentityScaler:
    """No-op scaler: returns input unchanged. Used when scaling is done upstream."""

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X
