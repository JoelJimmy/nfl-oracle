"""
config.py
Single source of truth for all configuration constants.
Override any value by setting the corresponding environment variable.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
# On Render with persistent disk, store models in /data. Locally use app/models.
import os as _os
_on_render = bool(_os.environ.get("RENDER"))
MODEL_DIR  = Path("/data/models") if _on_render else BASE_DIR / "app" / "models"
LOG_DIR    = BASE_DIR / "logs"

MODEL_PATH      = str(MODEL_DIR / "model.pkl")
PROFILES_PATH   = str(MODEL_DIR / "team_profiles.pkl")
ELO_PATH        = str(MODEL_DIR / "elo_ratings.pkl")
PRED_LOG_PATH   = str(MODEL_DIR / "prediction_log.pkl")

# ── NFL Season logic ───────────────────────────────────────────────────────────
# Season starts in September; months 2-8 are off-season
SEASON_START_MONTH  = 9
OFFSEASON_MONTHS    = set(range(2, 9))
TRAINING_SEASONS    = 3        # 3 seasons balances sample size vs recency (2023-2025)

# ── Feature engineering ────────────────────────────────────────────────────────
ROLLING_WINDOW      = 8        # last N games for rolling averages
MIN_GAMES_REQUIRED  = 3        # min games before including a team in training
ELO_K_FACTOR        = 20.0     # Elo update speed (higher = faster adaptation)
ELO_HOME_ADVANTAGE  = 48.0     # Elo points added for home team
ELO_INITIAL         = 1500.0   # starting Elo for new teams
ELO_MEAN_REVERT     = 0.55     # aggressive mean reversion — a dynasty doesn't make you favored 2 years later

# ── Model ─────────────────────────────────────────────────────────────────────
# Primary: XGBoost. Falls back to sklearn GBC if xgboost not installed.
MODEL_TYPE          = os.environ.get("NFL_MODEL_TYPE", "sklearn")   # xgboost | lightgbm | sklearn
RANDOM_SEED         = 42
CV_FOLDS            = 5

XGBOOST_PARAMS = {
    "n_estimators":     200,   # reduced from 500 to avoid memory issues on Mac
    "learning_rate":    0.05,
    "max_depth":        4,
    "min_child_weight": 8,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_SEED,
}

LIGHTGBM_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.03,
    "max_depth":        5,
    "num_leaves":       31,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     RANDOM_SEED,
    "verbose":          -1,
}

SKLEARN_GBC_PARAMS = {
    "n_estimators":     200,
    "learning_rate":    0.05,
    "max_depth":        3,       # shallower = less overfitting
    "min_samples_split": 30,     # require more samples per split
    "min_samples_leaf":  15,     # minimum leaf size
    "subsample":        0.8,
    "max_features":     "sqrt",
    "random_state":     RANDOM_SEED,
}

# ── Probability calibration ───────────────────────────────────────────────────
# "sigmoid" (Platt scaling) or "isotonic" (non-parametric)
CALIBRATION_METHOD  = "sigmoid"

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH     = 0.68   # >= this win prob = high confidence
CONFIDENCE_MODERATE = 0.56   # >= this = moderate

# ── Elo display ───────────────────────────────────────────────────────────────
ELO_DISPLAY_RANGE   = (1300, 1700)   # for normalizing to 0-100 display scale

# ── Caching & scheduling ──────────────────────────────────────────────────────
ROSTER_CACHE_TTL_HOURS  = 6
ODDS_CACHE_TTL_MINUTES  = 30
SCHEDULE_CACHE_TTL_HOURS = 12

# Weekly retrain: Tuesday 6am (after Monday Night Football results are in)
RETRAIN_DAY_OF_WEEK  = "tue"
RETRAIN_HOUR         = 6

# ── External APIs ─────────────────────────────────────────────────────────────
ODDS_API_KEY         = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE        = "https://api.the-odds-api.com/v4/sports"
SLEEPER_PLAYERS_URL  = "https://api.sleeper.app/v1/players/nfl"
ESPN_ROSTER_URL      = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/roster"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL            = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT           = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT      = "%Y-%m-%d %H:%M:%S"

# ── Redis & Caching ───────────────────────────────────────────────────────────
REDIS_URL            = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# If Redis isn't available the app falls back to simple in-process cache
CACHE_TYPE           = os.environ.get("CACHE_TYPE", "SimpleCache")   # RedisCache | SimpleCache
CACHE_DEFAULT_TIMEOUT = 300   # 5 minutes default

# Cache TTLs (seconds)
CACHE_TTL_UPCOMING   = 300    # 5 min  — upcoming games + predictions
CACHE_TTL_MODEL_INFO = 600    # 10 min — model metadata
CACHE_TTL_ROSTERS    = 21600  # 6 hr   — full roster data
CACHE_TTL_ELO        = 600    # 10 min — Elo rankings

# ── Email ─────────────────────────────────────────────────────────────────────
RESEND_API_KEY       = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM           = os.environ.get("EMAIL_FROM", "NFL Oracle <noreply@yourdomain.com>")
SITE_URL             = os.environ.get("SITE_URL", "http://localhost:5000")

# ── Server ────────────────────────────────────────────────────────────────────
PORT                 = int(os.environ.get("PORT", 8080))
DEBUG                = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
HOST                 = os.environ.get("HOST", "0.0.0.0")
WORKERS              = int(os.environ.get("WEB_CONCURRENCY", 2))