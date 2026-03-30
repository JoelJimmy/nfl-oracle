"""
scheduler.py
Background job scheduler using APScheduler.

Jobs:
  - Weekly retrain: Tuesday 6am (after Monday Night Football completes)
  - Odds refresh: every 30 minutes during the season
  - Roster refresh: every 6 hours

The scheduler runs inside the Flask process (no separate worker needed).
"""

import traceback
from app.config import RETRAIN_DAY_OF_WEEK, RETRAIN_HOUR
from app.logger import get_logger

logger = get_logger(__name__)

_scheduler = None
_APSCHEDULER_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    logger.warning("apscheduler not installed — background scheduling disabled. "
                   "Run: pip install apscheduler")


def _job_retrain():
    """Weekly retrain + outcome fetch + email digest."""
    logger.info("Scheduled retrain starting...")
    try:
        from app.services.train import train
        from app.services.predictor import load_model
        train()
        load_model()
        logger.info("Scheduled retrain complete ✓")
    except Exception as e:
        logger.error(f"Scheduled retrain failed: {e}\n{traceback.format_exc()}")

    # Fetch outcomes for pending predictions
    try:
        _job_fetch_outcomes()
    except Exception as e:
        logger.error(f"Outcome fetch failed: {e}")

    # Send weekly digest (runs at 7am, 1hr after retrain)
    # We schedule it separately so it runs after retrain completes


def _job_fetch_outcomes():
    """Pull completed game scores from ESPN and mark predictions correct/wrong."""
    import requests
    from app.database import get_pending_predictions, record_outcome
    from app.services.rosters import _normalize_team

    pending = get_pending_predictions()
    if not pending:
        logger.info("No pending predictions to resolve")
        return

    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            params={"limit": 100},
            timeout=8,
        )
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as e:
        logger.warning(f"ESPN scoreboard fetch failed: {e}")
        return

    _team_map = {
        "WSH": "WAS", "LA": "LAR",
    }

    resolved = 0
    for event in events:
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("state", "")
        if status != "post":
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_c or not away_c:
            continue

        home_abbr = _team_map.get(
            home_c["team"]["abbreviation"], home_c["team"]["abbreviation"])
        away_abbr = _team_map.get(
            away_c["team"]["abbreviation"], away_c["team"]["abbreviation"])
        home_score = int(home_c.get("score", 0) or 0)
        away_score = int(away_c.get("score", 0) or 0)

        if record_outcome(home_abbr, away_abbr, home_score, away_score):
            resolved += 1

    logger.info(f"Resolved {resolved} prediction outcomes")


def _job_send_digest():
    """Send weekly email digest."""
    logger.info("Sending weekly email digest...")
    try:
        from app.services.email_alerts import send_weekly_digest
        send_weekly_digest()
    except Exception as e:
        logger.error(f"Email digest failed: {e}")


def _job_refresh_rosters():
    """Refresh roster cache and re-apply roster adjustment to team profiles."""
    logger.info("Scheduled roster refresh...")
    try:
        from app.services.rosters import load_rosters, load_full_rosters
        load_rosters()
        load_full_rosters()
        logger.info("Roster cache refreshed ✓")
    except Exception as e:
        logger.error(f"Roster refresh failed: {e}")

    # Re-apply roster adjustment so predictions reflect latest moves
    try:
        from app.services.predictor import _team_profiles, is_loaded
        from app.services.roster_adjustment import build_adjusted_profiles
        import app.services.predictor as pred_module
        if is_loaded() and pred_module._team_profiles is not None:
            pred_module._team_profiles = build_adjusted_profiles(pred_module._team_profiles)
            logger.info("Roster adjustment re-applied ✓")
    except Exception as e:
        logger.error(f"Roster adjustment refresh failed: {e}")


def start_scheduler():
    """Start the background scheduler."""
    global _scheduler
    if _scheduler is not None:
        return

    if not _APSCHEDULER_AVAILABLE:
        logger.warning("Skipping scheduler startup — apscheduler not installed.")
        return

    _scheduler = BackgroundScheduler(timezone="America/New_York", daemon=True)

    # Weekly retrain — Tuesday 6am ET
    _scheduler.add_job(
        _job_retrain,
        CronTrigger(day_of_week=RETRAIN_DAY_OF_WEEK, hour=RETRAIN_HOUR, minute=0),
        id="weekly_retrain",
        name="Weekly model retrain",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Roster refresh — every 6 hours
    _scheduler.add_job(
        _job_refresh_rosters,
        IntervalTrigger(hours=6),
        id="roster_refresh",
        name="Roster refresh",
        replace_existing=True,
    )

    # Weekly email digest — Tuesday 7am (1hr after retrain)
    _scheduler.add_job(
        _job_send_digest,
        CronTrigger(day_of_week="tue", hour=7, minute=0),
        id="weekly_digest",
        name="Weekly email digest",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Outcome fetch — every 6 hours (catches completed games)
    _scheduler.add_job(
        _job_fetch_outcomes,
        IntervalTrigger(hours=6),
        id="outcome_fetch",
        name="Outcome fetch",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        f"Scheduler started — retrain every {RETRAIN_DAY_OF_WEEK.upper()} "
        f"at {RETRAIN_HOUR:02d}:00 ET, rosters every 6h"
    )


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")


def get_next_jobs() -> list[dict]:
    """Return info about upcoming scheduled jobs (for /api/status endpoint)."""
    if not _scheduler:
        return []
    jobs = []
    for job in _scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id":       job.id,
            "name":     job.name,
            "next_run": next_run.isoformat() if next_run else None,
        })
    return jobs