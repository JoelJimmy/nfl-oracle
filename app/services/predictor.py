import os
"""
predictor.py
Loads trained model artifacts and runs predictions.

Ensemble: 70% calibrated XGBoost + 30% logistic regression.
Probabilities are properly calibrated via Platt scaling.
Prediction history is logged for accuracy tracking.
"""

import time
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from app.config import (
    MODEL_PATH, PROFILES_PATH, ELO_PATH, PRED_LOG_PATH,
    CONFIDENCE_HIGH, CONFIDENCE_MODERATE
)
from app.logger import get_logger

logger = get_logger(__name__)

_model_artifact: Optional[dict] = None
_team_profiles:  Optional[dict] = None
_elo_system                     = None
_sos_data:       Optional[dict] = None
_tg_df                          = None   # team-game DataFrame for SOS computation


def load_model():
    global _model_artifact, _team_profiles, _elo_system
    raw = joblib.load(MODEL_PATH)

    # Detect old model format (pre-Phase 1) — missing 'scaler' key
    if "scaler" not in raw:
        logger.warning(
            "Old model format detected (missing 'scaler'). "
            "The model was trained with the old pipeline. "
            "Run `python -m scripts.train` to retrain with the new pipeline."
        )
        # Wrap old artifact in new format so app can still start
        from sklearn.preprocessing import StandardScaler
        raw = {
            "scaler":           StandardScaler(),
            "base_model":       raw.get("pipeline", raw.get("clf", None)),
            "calibrated_model": raw.get("pipeline", None),
            "lr_model":         None,
            "feature_cols":     raw.get("feature_cols", []),
            "cv_accuracy":      raw.get("cv_accuracy", 0),
            "cv_std":           raw.get("cv_std", 0),
            "cv_log_loss":      0.0,
            "cv_auc":           0.0,
            "train_accuracy":   raw.get("train_accuracy", 0),
            "train_auc":        0.0,
            "train_brier":      0.0,
            "n_training_samples": raw.get("n_training_samples", 0),
            "feature_importances": raw.get("feature_importances", {}),
            "model_type":       "legacy",
            "h2h":              {},
            "_legacy":          True,
        }

    _model_artifact = raw
    _team_profiles  = joblib.load(PROFILES_PATH)

    try:
        _elo_system = joblib.load(ELO_PATH)
        logger.info(f"Elo ratings loaded — {len(_elo_system.ratings)} teams")
    except Exception as e:
        logger.warning(f"Could not load Elo ratings: {e} — run scripts.train to generate")
        _elo_system = None

    legacy_note = " [LEGACY — retrain recommended]" if raw.get("_legacy") else ""
    logger.info(
        f"Model loaded — type={_model_artifact.get('model_type','?')}{legacy_note} "
        f"cv_acc={_model_artifact.get('cv_accuracy', 0):.4f} "
        f"cv_auc={_model_artifact.get('cv_auc', 0):.4f}"
    )

    # Apply off-season roster adjustment on top of base profiles
    try:
        from app.services.roster_adjustment import build_adjusted_profiles
        _team_profiles = build_adjusted_profiles(_team_profiles)
    except Exception as e:
        logger.warning(f"Roster adjustment skipped: {e}")

    # Compute SOS scores
    global _sos_data
    try:
        from app.services.sos import compute_sos
        tg = raw.get("tg_df")
        if tg is not None and _elo_system is not None:
            _sos_data = compute_sos(_elo_system, tg)
        else:
            _sos_data = {}
    except Exception as e:
        logger.warning(f"SOS computation skipped: {e}")
        _sos_data = {}


def is_loaded() -> bool:
    return _model_artifact is not None


def get_model_meta() -> dict:
    if not is_loaded():
        raise RuntimeError("Model not loaded")
    return {
        "cv_accuracy":         float(_model_artifact.get("cv_accuracy", 0)),
        "cv_std":              float(_model_artifact.get("cv_std", 0)),
        "cv_log_loss":         float(_model_artifact.get("cv_log_loss", 0)),
        "cv_auc":              float(_model_artifact.get("cv_auc", 0)),
        "train_accuracy":      float(_model_artifact.get("train_accuracy", 0)),
        "train_auc":           float(_model_artifact.get("train_auc", 0)),
        "train_brier":         float(_model_artifact.get("train_brier", 0)),
        "n_training_samples":  int(_model_artifact.get("n_training_samples", 0)),
        "feature_importances": _model_artifact.get("feature_importances", {}),
        "model_type":          _model_artifact.get("model_type", "unknown"),
        "teams_available":     sorted(_team_profiles.keys()),
    }


def _build_feature_vector(
    home_profile: dict,
    away_profile: dict,
    home_team: str = "",
    away_team: str = "",
    home_rest_days: float = 7.0,
    away_rest_days: float = 7.0,
) -> np.ndarray:
    """Build the full feature vector for a matchup."""
    from app.services.feature_engineering import STAT_COLS
    from app.config import ELO_HOME_ADVANTAGE, ELO_INITIAL

    stat_roll = [f"roll_{c}" for c in STAT_COLS]

    h = [home_profile.get(c, 0.0) for c in stat_roll]
    a = [away_profile.get(c, 0.0) for c in stat_roll]
    d = [hv - av for hv, av in zip(h, a)]

    # EPA trends
    h_epa_trend = home_profile.get("roll_epa_trend", 0.0)
    a_epa_trend = away_profile.get("roll_epa_trend", 0.0)
    h_def_trend = home_profile.get("roll_def_epa_trend", 0.0)
    a_def_trend = away_profile.get("roll_def_epa_trend", 0.0)

    # Streaks
    h_streak = home_profile.get("roll_streak", 0.0)
    a_streak = away_profile.get("roll_streak", 0.0)

    # Elo — cap at ±80 from mean (≈1 std dev in training data)
    # Keeps Elo features within the training distribution range
    ELO_CAP = 80.0
    if _elo_system and home_team and away_team:
        from app.config import ELO_HOME_ADVANTAGE
        h_elo_raw = _elo_system.get(home_team)
        a_elo_raw = _elo_system.get(away_team)
        # Soft-cap: compress values beyond ±150 from 1500
        h_elo = 1500 + max(-ELO_CAP, min(ELO_CAP, h_elo_raw - 1500))
        a_elo = 1500 + max(-ELO_CAP, min(ELO_CAP, a_elo_raw - 1500))
        elo_diff = h_elo - a_elo
    else:
        h_elo, a_elo = ELO_INITIAL, ELO_INITIAL
        elo_diff = 0.0

    # H2H removed — dominated training with only ~3 games/pair on average

    # Injury
    h_inj   = home_profile.get("injury_impact", 0.0)
    a_inj   = away_profile.get("injury_impact", 0.0)
    h_qb    = home_profile.get("qb_out", 0.0)
    a_qb    = away_profile.get("qb_out", 0.0)

    fv = (
        h + a + d +
        [h_epa_trend, a_epa_trend, h_def_trend, a_def_trend,
         h_epa_trend - a_epa_trend,
         h_streak, a_streak,
         home_rest_days, away_rest_days, home_rest_days - away_rest_days,
         float(h[STAT_COLS.index("turnovers") if "turnovers" in STAT_COLS else 6]) -
         float(a[STAT_COLS.index("turnovers") if "turnovers" in STAT_COLS else 6]),
         h_elo, a_elo, elo_diff,

         h_inj, a_inj, h_qb, a_qb,
         1.0,  # home_advantage
        ]
    )
    return np.array(fv, dtype=np.float64)


def _predict_legacy(home_team, away_team, h_prof, a_prof) -> dict:
    """Fallback prediction using old model format (pre-Phase 1)."""
    from app.services.feature_engineering import STAT_COLS
    stat_roll = [f"roll_{c}" for c in STAT_COLS]
    h = [h_prof.get(c, 0.0) for c in stat_roll]
    a = [a_prof.get(c, 0.0) for c in stat_roll]
    d = [hv - av for hv, av in zip(h, a)]
    fv = h + a + d + [1.0]

    pipeline = _model_artifact.get("calibrated_model") or _model_artifact.get("base_model")
    if pipeline is None:
        home_prob = 0.5
    else:
        import numpy as np
        X = np.array([fv[:len(_model_artifact.get("feature_cols", fv))]], dtype=np.float64)
        try:
            home_prob = float(pipeline.predict_proba(X)[0][1])
        except Exception:
            home_prob = 0.5

    away_prob = round(1 - home_prob, 4)
    home_prob = round(home_prob, 4)
    max_p = max(home_prob, away_prob)
    confidence = "high" if max_p >= 0.68 else "moderate" if max_p >= 0.56 else "low"

    return {
        "home_team": home_team, "away_team": away_team,
        "home_win_probability": home_prob, "away_win_probability": away_prob,
        "predicted_winner": home_team if home_prob >= 0.5 else away_team,
        "confidence": confidence, "elo_home": 50, "elo_away": 50,
        "home_profile": {k: round(float(v), 4) for k, v in h_prof.items()},
        "away_profile": {k: round(float(v), 4) for k, v in a_prof.items()},
        "_note": "Legacy model — retrain with scripts.train for Phase 1 features",
    }


def predict_matchup(
    home_team: str,
    away_team: str,
    home_rest_days: float = 7.0,
    away_rest_days: float = 7.0,
) -> dict:
    """Predict a single matchup. Returns calibrated ensemble probability."""
    if not is_loaded():
        raise RuntimeError("Model not loaded")
    if home_team not in _team_profiles:
        raise ValueError(f"No profile for home team: {home_team}")
    if away_team not in _team_profiles:
        raise ValueError(f"No profile for away team: {away_team}")

    h_prof = _team_profiles[home_team]
    a_prof = _team_profiles[away_team]

    # Legacy model path (old pkl, pre-Phase 1 training)
    if _model_artifact.get("_legacy"):
        return _predict_legacy(home_team, away_team, h_prof, a_prof)

    fv     = _build_feature_vector(h_prof, a_prof, home_team, away_team,
                                    home_rest_days, away_rest_days)
    scaler = _model_artifact["scaler"]
    X      = scaler.transform(fv.reshape(1, -1))

    import numpy as np
    X = np.clip(X, -2.5, 2.5)

    cal_prob = float(_model_artifact["calibrated_model"].predict_proba(X)[0][1])
    lr_model = _model_artifact.get("lr_model")
    lr_prob  = float(lr_model.predict_proba(X)[0][1]) if lr_model else cal_prob
    home_prob = round(cal_prob * 0.7 + lr_prob * 0.3, 4)
    away_prob = round(1 - home_prob, 4)

    max_p = max(home_prob, away_prob)
    if max_p >= CONFIDENCE_HIGH:
        confidence = "high"
    elif max_p >= CONFIDENCE_MODERATE:
        confidence = "moderate"
    else:
        confidence = "low"

    # Elo power scores for display
    elo_home_score = _elo_system.to_display_score(home_team) if _elo_system else 50
    elo_away_score = _elo_system.to_display_score(away_team) if _elo_system else 50

    return {
        "home_team":           home_team,
        "away_team":           away_team,
        "home_win_probability": home_prob,
        "away_win_probability": away_prob,
        "predicted_winner":    home_team if home_prob >= 0.5 else away_team,
        "confidence":          confidence,
        "elo_home":            elo_home_score,
        "elo_away":            elo_away_score,
        "home_profile":        {k: round(float(v), 4) for k, v in h_prof.items()},
        "away_profile":        {k: round(float(v), 4) for k, v in a_prof.items()},
    }


def predict_bulk(matchups: list[dict]) -> list[dict]:
    results = []
    for m in matchups:
        try:
            pred = predict_matchup(
                m["home_team"], m["away_team"],
                float(m.get("home_rest_days", 7)),
                float(m.get("away_rest_days", 7)),
            )
            pred.update({k: v for k, v in m.items() if k not in pred})
            results.append(pred)
        except ValueError as e:
            results.append({"home_team": m.get("home_team"), "away_team": m.get("away_team"),
                             "error": str(e)})
    return results


def log_prediction(prediction: dict):
    """Append a prediction to the log for accuracy tracking."""
    try:
        log = joblib.load(PRED_LOG_PATH) if os.path.exists(PRED_LOG_PATH) else []
        log.append({**prediction, "logged_at": time.time()})
        joblib.dump(log[-500:], PRED_LOG_PATH)  # keep last 500
    except Exception as e:
        logger.warning(f"Could not log prediction: {e}")


def update_team_profile(team: str, overrides: dict):
    if not is_loaded():
        raise RuntimeError("Model not loaded")
    if team not in _team_profiles:
        raise ValueError(f"Team {team} not in profiles")
    _team_profiles[team].update(overrides)
    logger.info(f"Updated profile for {team}: {overrides}")


def get_team_profile(team: str) -> dict:
    if not is_loaded():
        raise RuntimeError("Model not loaded")
    if team not in _team_profiles:
        raise ValueError(f"Team {team} not found")
    return _team_profiles[team]


def get_all_teams() -> list[str]:
    if not is_loaded():
        raise RuntimeError("Model not loaded")
    return sorted(_team_profiles.keys())