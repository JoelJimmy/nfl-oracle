"""
database.py
SQLite database for prediction history, outcome tracking, and email subscribers.
Uses Python's built-in sqlite3 — no extra dependencies needed.

Tables:
  predictions   — every prediction made, with outcome once game completes
  subscribers   — email alert subscribers
  pick_outcomes — used for server-side accuracy tracking (client picks in localStorage)
"""

import sqlite3
import os
import json
import time
from pathlib import Path
from app.config import BASE_DIR
from app.logger import get_logger

logger = get_logger(__name__)

DB_PATH = str(BASE_DIR / "app" / "models" / "nfl_oracle.db")


def get_conn() -> sqlite3.Connection:
    """Get a database connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      REAL    NOT NULL DEFAULT (unixepoch()),
                season          INTEGER,
                week            INTEGER,
                home_team       TEXT    NOT NULL,
                away_team       TEXT    NOT NULL,
                home_win_prob   REAL    NOT NULL,
                away_win_prob   REAL    NOT NULL,
                predicted_winner TEXT   NOT NULL,
                confidence      TEXT,
                mode            TEXT,
                -- Outcome filled in after game completes
                home_score      INTEGER,
                away_score      INTEGER,
                actual_winner   TEXT,
                model_correct   INTEGER,   -- 1=correct, 0=wrong, NULL=pending
                -- Extra context
                is_division_game INTEGER DEFAULT 0,
                sos_home        REAL,
                sos_away        REAL
            );

            CREATE INDEX IF NOT EXISTS idx_pred_teams
                ON predictions(home_team, away_team);
            CREATE INDEX IF NOT EXISTS idx_pred_season_week
                ON predictions(season, week);

            CREATE TABLE IF NOT EXISTS subscribers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT    NOT NULL UNIQUE,
                created_at  REAL    NOT NULL DEFAULT (unixepoch()),
                active      INTEGER NOT NULL DEFAULT 1,
                team_filter TEXT    -- optional: only send alerts for this team
            );

            CREATE TABLE IF NOT EXISTS push_subscriptions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint    TEXT    NOT NULL UNIQUE,
                p256dh      TEXT    NOT NULL,
                auth        TEXT    NOT NULL,
                team_filter TEXT,
                created_at  REAL    NOT NULL DEFAULT (unixepoch())
            );
        """)
    logger.info(f"Database ready at {DB_PATH}")


# ── Predictions ───────────────────────────────────────────────────────────────

def log_prediction(pred: dict) -> int:
    """Insert a prediction and return its row ID."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO predictions
              (season, week, home_team, away_team,
               home_win_prob, away_win_prob, predicted_winner,
               confidence, mode, is_division_game, sos_home, sos_away)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pred.get("season"),
            pred.get("week"),
            pred["home_team"],
            pred["away_team"],
            float(pred["home_win_probability"]),
            float(pred["away_win_probability"]),
            pred["predicted_winner"],
            pred.get("confidence"),
            pred.get("mode"),
            int(pred.get("is_division_game", 0)),
            pred.get("sos_home"),
            pred.get("sos_away"),
        ))
        return cur.lastrowid


def record_outcome(home_team: str, away_team: str, home_score: int, away_score: int,
                   season: int = None, week: int = None):
    """
    Update the most recent pending prediction for this matchup with the actual result.
    Called by the weekly outcome-fetch job.
    """
    actual_winner = home_team if home_score > away_score else away_team
    with get_conn() as conn:
        # Find the most recent pending prediction for this matchup
        row = conn.execute("""
            SELECT id, predicted_winner FROM predictions
            WHERE home_team=? AND away_team=? AND model_correct IS NULL
            ORDER BY created_at DESC LIMIT 1
        """, (home_team, away_team)).fetchone()

        if not row:
            logger.warning(f"No pending prediction found for {home_team} vs {away_team}")
            return False

        correct = 1 if row["predicted_winner"] == actual_winner else 0
        conn.execute("""
            UPDATE predictions
            SET home_score=?, away_score=?, actual_winner=?, model_correct=?
            WHERE id=?
        """, (home_score, away_score, actual_winner, correct, row["id"]))
        logger.info(f"Outcome recorded: {home_team} {home_score}-{away_score} {away_team} "
                    f"| Predicted: {row['predicted_winner']} | Correct: {bool(correct)}")
        return True


def get_accuracy_stats() -> dict:
    """Return accuracy statistics from resolved predictions."""
    with get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_correct IS NOT NULL"
        ).fetchone()[0]

        if total == 0:
            return {"total": 0, "resolved": 0, "message": "No resolved predictions yet"}

        correct = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_correct=1"
        ).fetchone()[0]

        # By confidence
        by_conf = {}
        for level in ["high", "moderate", "low"]:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(model_correct) as correct
                FROM predictions
                WHERE confidence=? AND model_correct IS NOT NULL
            """, (level,)).fetchone()
            by_conf[level] = {
                "total":    row["total"],
                "correct":  row["correct"] or 0,
                "accuracy": round((row["correct"] or 0) / row["total"], 4) if row["total"] else 0,
            }

        # By week (last 8 weeks)
        by_week = []
        rows = conn.execute("""
            SELECT season, week,
                   COUNT(*) as total,
                   SUM(model_correct) as correct
            FROM predictions
            WHERE model_correct IS NOT NULL
            GROUP BY season, week
            ORDER BY season DESC, week DESC
            LIMIT 8
        """).fetchall()
        for row in rows:
            by_week.append({
                "season":   row["season"],
                "week":     row["week"],
                "total":    row["total"],
                "correct":  row["correct"] or 0,
                "accuracy": round((row["correct"] or 0) / row["total"], 4) if row["total"] else 0,
            })

        return {
            "total":           total,
            "resolved":        total,
            "correct":         correct,
            "overall_accuracy": round(correct / total, 4),
            "by_confidence":   by_conf,
            "by_week":         by_week[::-1],   # chronological
        }


def get_team_prediction_history(team: str, limit: int = 20) -> list[dict]:
    """Return recent predictions involving a team."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE home_team=? OR away_team=?
            ORDER BY created_at DESC LIMIT ?
        """, (team, team, limit)).fetchall()
        return [dict(r) for r in rows]


# ── Schedule for team page ────────────────────────────────────────────────────

def get_pending_predictions() -> list[dict]:
    """Return all predictions that don't have outcomes yet."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE model_correct IS NULL
            ORDER BY season, week, created_at
        """).fetchall()
        return [dict(r) for r in rows]


# ── Email subscribers ─────────────────────────────────────────────────────────

def add_subscriber(email: str, team_filter: str = None) -> bool:
    """Add or reactivate an email subscriber. Returns True if new."""
    try:
        with get_conn() as conn:
            existing = conn.execute(
                "SELECT id, active FROM subscribers WHERE email=?", (email,)
            ).fetchone()
            if existing:
                if not existing["active"]:
                    conn.execute(
                        "UPDATE subscribers SET active=1, team_filter=? WHERE email=?",
                        (team_filter, email)
                    )
                    return True
                return False   # already subscribed
            conn.execute(
                "INSERT INTO subscribers (email, team_filter) VALUES (?,?)",
                (email, team_filter)
            )
            return True
    except sqlite3.IntegrityError:
        return False


def remove_subscriber(email: str):
    with get_conn() as conn:
        conn.execute("UPDATE subscribers SET active=0 WHERE email=?", (email,))


def get_active_subscribers(team_filter: str = None) -> list[dict]:
    with get_conn() as conn:
        if team_filter:
            rows = conn.execute("""
                SELECT * FROM subscribers
                WHERE active=1 AND (team_filter IS NULL OR team_filter=?)
            """, (team_filter,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM subscribers WHERE active=1"
            ).fetchall()
        return [dict(r) for r in rows]


# ── Push subscriptions ────────────────────────────────────────────────────────

def add_push_subscription(endpoint: str, p256dh: str, auth: str, team_filter: str = None):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO push_subscriptions
              (endpoint, p256dh, auth, team_filter)
            VALUES (?,?,?,?)
        """, (endpoint, p256dh, auth, team_filter))


def get_push_subscriptions(team_filter: str = None) -> list[dict]:
    with get_conn() as conn:
        if team_filter:
            rows = conn.execute("""
                SELECT * FROM push_subscriptions
                WHERE team_filter IS NULL OR team_filter=?
            """, (team_filter,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM push_subscriptions").fetchall()
        return [dict(r) for r in rows]