"""
train.py
Full training pipeline with:
  - Walk-forward cross-validation (no data leakage)
  - XGBoost primary model (falls back to sklearn GBC)
  - Calibrated probabilities (Platt scaling)
  - Ensemble of XGBoost + Logistic Regression
  - Feature importance logging
  - Prediction log initialization
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, classification_report
)

from app.config import (
    MODEL_PATH, PROFILES_PATH, ELO_PATH, PRED_LOG_PATH,
    MODEL_TYPE, XGBOOST_PARAMS, LIGHTGBM_PARAMS, SKLEARN_GBC_PARAMS,
    CALIBRATION_METHOD, RANDOM_SEED, MODEL_DIR
)
from app.services.download_pbp import download_pbp_data, download_weekly_data
from app.services.feature_engineering import build_full_dataset, get_feature_columns
from app.logger import get_logger

logger = get_logger(__name__)


def _build_model():
    """Build the primary model based on MODEL_TYPE config."""
    if MODEL_TYPE == "xgboost":
        try:
            from xgboost import XGBClassifier
            logger.info("Using XGBoost classifier")
            return XGBClassifier(**XGBOOST_PARAMS)
        except ImportError:
            logger.warning("xgboost not installed — falling back to sklearn GBC")

    if MODEL_TYPE == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            logger.info("Using LightGBM classifier")
            return LGBMClassifier(**LIGHTGBM_PARAMS)
        except ImportError:
            logger.warning("lightgbm not installed — falling back to sklearn GBC")

    # Fallback
    from sklearn.ensemble import GradientBoostingClassifier
    logger.info("Using sklearn GradientBoostingClassifier")
    return GradientBoostingClassifier(**SKLEARN_GBC_PARAMS)


def _walk_forward_cv(X: np.ndarray, y: np.ndarray, seasons: np.ndarray, model_fn) -> dict:
    """
    Walk-forward cross-validation: train on seasons 1..N-1, test on season N.
    This properly simulates real-world prediction — never testing on the past.
    """
    unique_seasons = sorted(np.unique(seasons))
    if len(unique_seasons) < 2:
        logger.warning("Not enough seasons for walk-forward CV — skipping")
        return {"accuracy": np.nan, "log_loss": np.nan, "auc": np.nan}

    accs, losses, aucs = [], [], []

    for i in range(1, len(unique_seasons)):
        train_idx = seasons < unique_seasons[i]
        test_idx  = seasons == unique_seasons[i]

        if train_idx.sum() < 50 or test_idx.sum() < 10:
            continue

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        m = model_fn()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr_s, y_tr)

        proba = m.predict_proba(X_te_s)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        accs.append(accuracy_score(y_te, pred))
        losses.append(log_loss(y_te, proba))
        aucs.append(roc_auc_score(y_te, proba))

        logger.info(f"  Season {unique_seasons[i]}: "
                    f"acc={accs[-1]:.3f}  logloss={losses[-1]:.3f}  auc={aucs[-1]:.3f}  "
                    f"(train={train_idx.sum()}, test={test_idx.sum()})")

    return {
        "accuracy":  float(np.mean(accs)) if accs else np.nan,
        "log_loss":  float(np.mean(losses)) if losses else np.nan,
        "auc":       float(np.mean(aucs)) if aucs else np.nan,
        "n_folds":   len(accs),
    }


def train():
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("NFL Predictor — Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Download data ──────────────────────────────────────────────────────
    pbp = download_pbp_data()

    try:
        weekly = download_weekly_data()
    except Exception as e:
        logger.warning(f"Could not download weekly data: {e} — injury features disabled")
        weekly = None

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print(f"PBP rows loaded: {len(pbp):,}", flush=True)
    print("Starting feature engineering...", flush=True)
    import traceback as _tb
    try:
        matchups, profiles, elo, tg, h2h, injury = build_full_dataset(pbp, weekly)
        print(f"Feature engineering done: {len(matchups):,} matchups", flush=True)
    except Exception as e:
        print(f"FEATURE ENGINEERING FAILED: {e}", flush=True)
        print(_tb.format_exc(), flush=True)
        raise SystemExit(1)

    print("Checking feature columns...", flush=True)
    feature_cols = get_feature_columns()
    print(f"Expected {len(feature_cols)} features, matchup has {len(matchups.columns)} columns", flush=True)

    missing = [c for c in feature_cols if c not in matchups.columns]
    if missing:
        print(f"MISSING FEATURES: {missing}", flush=True)
        raise ValueError(f"Feature columns missing: {missing}")

    print("Building X/y arrays...", flush=True)
    try:
        X = matchups[feature_cols].values.astype(np.float32)
        y = matchups["label"].values
        seasons = matchups["season"].values if "season" in matchups.columns else np.zeros(len(y))
        print(f"X shape: {X.shape}, y shape: {y.shape}, home win rate: {y.mean():.3f}", flush=True)
    except Exception as e:
        print(f"ARRAY BUILD FAILED: {e}", flush=True)
        import traceback; print(traceback.format_exc(), flush=True)
        raise SystemExit(1)

    # ── 3. Walk-forward CV ────────────────────────────────────────────────────
    print("Starting walk-forward CV...", flush=True)
    try:
        cv_results = _walk_forward_cv(X, y, seasons, _build_model)
        print(f"CV done: acc={cv_results['accuracy']:.4f}", flush=True)
    except Exception as e:
        print(f"CV FAILED: {e}", flush=True)
        import traceback; print(traceback.format_exc(), flush=True)
        raise SystemExit(1)
    logger.info(f"CV Accuracy:  {cv_results['accuracy']:.4f}")
    logger.info(f"CV Log Loss:  {cv_results['log_loss']:.4f}")
    logger.info(f"CV AUC:       {cv_results['auc']:.4f}")

    # ── 4. Train on full dataset with calibration ─────────────────────────────
    print("Scaling features...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training base model...", flush=True)
    try:
        base_model = _build_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model.fit(X_scaled, y)
        print("Base model trained ✓", flush=True)
    except Exception as e:
        print(f"BASE MODEL FAILED: {e}", flush=True)
        import traceback; print(traceback.format_exc(), flush=True)
        raise SystemExit(1)

    print(f"Calibrating probabilities ({CALIBRATION_METHOD})...", flush=True)
    try:
        calibrated = CalibratedClassifierCV(base_model, method=CALIBRATION_METHOD, cv=3)
        calibrated.fit(X_scaled, y)
        print("Calibration done ✓", flush=True)
    except Exception as e:
        print(f"CALIBRATION FAILED: {e} — using cv=2", flush=True)
        calibrated = CalibratedClassifierCV(base_model, method=CALIBRATION_METHOD, cv=2)
        calibrated.fit(X_scaled, y)
        print("Calibration done (cv=2) ✓", flush=True)

    print("Training logistic regression ensemble...", flush=True)
    try:
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_scaled, y)
        print("LR trained ✓", flush=True)
    except Exception as e:
        print(f"LR FAILED: {e}", flush=True)
        import traceback; print(traceback.format_exc(), flush=True)
        raise SystemExit(1)

    # ── 5. Evaluate on training set ───────────────────────────────────────────
    print("Evaluating ensemble...", flush=True)
    try:
        cal_proba = calibrated.predict_proba(X_scaled)[:, 1]
        lr_proba  = lr.predict_proba(X_scaled)[:, 1]
        ens_proba = (cal_proba * 0.7 + lr_proba * 0.3)
        ens_pred  = (ens_proba >= 0.5).astype(int)

        train_acc    = accuracy_score(y, ens_pred)
        train_ll     = log_loss(y, ens_proba)
        train_brier  = brier_score_loss(y, ens_proba)
        train_auc    = roc_auc_score(y, ens_proba)
        print(f"Ensemble: acc={train_acc:.4f} auc={train_auc:.4f}", flush=True)
    except Exception as e:
        print(f"EVAL FAILED: {e}", flush=True)
        import traceback; print(traceback.format_exc(), flush=True)
        raise SystemExit(1)

    # ── 6. Feature importances ────────────────────────────────────────────────
    try:
        importances = base_model.feature_importances_
        fi = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        logger.info("\nTop 12 Feature Importances:")
        for feat, imp in fi[:12]:
            bar = "█" * int(imp * 300)
            logger.info(f"  {feat:<42} {imp:.4f}  {bar}")
        fi_dict = dict(fi)
    except AttributeError:
        fi_dict = {}

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_artifact = {
        "scaler":              scaler,
        "base_model":          base_model,
        "calibrated_model":    calibrated,
        "lr_model":            lr,
        "feature_cols":        feature_cols,
        "cv_accuracy":         cv_results["accuracy"],
        "cv_log_loss":         cv_results["log_loss"],
        "cv_auc":              cv_results["auc"],
        "cv_std":              0.0,
        "train_accuracy":      float(train_acc),
        "train_auc":           float(train_auc),
        "train_brier":         float(train_brier),
        "n_training_samples":  int(len(X)),
        "feature_importances": {k: float(v) for k, v in fi_dict.items()},
        "model_type":          MODEL_TYPE,
        "h2h":                 h2h,
        "tg_df":               tg,   # stored for SOS computation on load
    }

    print("Saving model artifacts...", flush=True)
    try:
        joblib.dump(model_artifact, MODEL_PATH)
        joblib.dump(profiles, PROFILES_PATH)
        elo.save(ELO_PATH)
        if not os.path.exists(PRED_LOG_PATH):
            joblib.dump([], PRED_LOG_PATH)
        print(f"Model saved → {MODEL_PATH}", flush=True)
        print(f"Elo saved   → {ELO_PATH}", flush=True)
        print("Training complete ✓", flush=True)
    except Exception as e:
        print(f"SAVE FAILED: {e}", flush=True)
        import traceback; print(traceback.format_exc(), flush=True)
        raise SystemExit(1)

    logger.info(f"\nModel saved      → {MODEL_PATH}")
    logger.info(f"Profiles saved   → {PROFILES_PATH}")
    logger.info(f"Elo saved        → {ELO_PATH}")
    logger.info("Training complete ✓")

    return model_artifact, profiles, elo


if __name__ == "__main__":
    train()