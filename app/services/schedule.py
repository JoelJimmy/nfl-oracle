"""
schedule.py
Fetches the upcoming NFL game slate via nfl_data_py.

Priority order during the off-season:
  1. Real next-season schedule  — if the NFL has already released it
     (typically published in May), use those actual Week 1 matchups.
  2. Previous season's Week 1   — used as a proxy before the schedule drops.
  3. Alphabetical pairing       — last resort fallback.

During the regular season:
  Returns the earliest unplayed week from the current season's schedule.
"""

import pandas as pd
from app.services.download_pbp import (
    download_schedule_data,
    get_relevant_seasons,
    is_offseason,
    get_next_season,
)
from app.services.odds import fetch_fanduel_odds, estimate_odds_from_profiles
from app.logger import get_logger

logger = get_logger(__name__)


def get_upcoming_games(team_profiles: dict) -> list[dict]:
    """
    Return upcoming games with mode tag:
      "scheduled"  — real next-season schedule already published by NFL
      "offseason"  — proxy matchups (schedule not out yet)
      "inseason"   — current season next unplayed week
    """
    try:
        schedules = download_schedule_data()
    except Exception as e:
        logger.warning(f"Could not load schedule data: {e}")
        schedules = pd.DataFrame()

    if is_offseason():
        games, mode = _build_offseason_week1(schedules, team_profiles)
    else:
        games = _parse_next_unplayed_week(schedules)
        if not games:
            logger.info("No unplayed games found — falling back to profile matchups.")
            games = _generate_matchups_from_profiles(team_profiles)
        mode = "inseason"

    for g in games:
        g["mode"] = mode

    # Attach odds
    live_odds = fetch_fanduel_odds()
    odds_map  = {(o["home_team"], o["away_team"]): o for o in live_odds}

    enriched = []
    for g in games:
        key = (g["home_team"], g["away_team"])
        if key in odds_map:
            o = odds_map[key]
            g.update({
                "home_moneyline": o["home_moneyline"],
                "away_moneyline": o["away_moneyline"],
                "spread":         o["spread"],
                "total":          o["total"],
                "odds_source":    "fanduel",
            })
        else:
            hp = team_profiles.get(g["home_team"], {})
            ap = team_profiles.get(g["away_team"], {})
            if hp and ap:
                est = estimate_odds_from_profiles(hp, ap)
                g.update(est)
                g["odds_source"] = "estimated"
            else:
                g.update({
                    "home_moneyline": None,
                    "away_moneyline": None,
                    "spread":         None,
                    "total":          None,
                    "odds_source":    "unavailable",
                })
        enriched.append(g)

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# In-season
# ─────────────────────────────────────────────────────────────────────────────

def _parse_next_unplayed_week(schedules: pd.DataFrame) -> list[dict]:
    """Return the earliest unplayed week in the current season."""
    if schedules.empty:
        return []

    current_season = get_relevant_seasons()[-1]
    season_sched   = schedules[schedules["season"] == current_season].copy()

    if season_sched.empty:
        return []

    unplayed = season_sched[season_sched["home_score"].isna()]
    if unplayed.empty:
        logger.info(f"Season {current_season} appears complete.")
        return []

    next_week = int(unplayed["week"].min())
    upcoming  = unplayed[unplayed["week"] == next_week]

    games = []
    for _, row in upcoming.iterrows():
        home = _abbr(row.get("home_team", ""))
        away = _abbr(row.get("away_team", ""))
        if home and away:
            games.append({
                "season":    int(row["season"]),
                "week":      next_week,
                "home_team": home,
                "away_team": away,
            })

    logger.info(f"{len(games)} games for Week {next_week}, {current_season}")
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Off-season
# ─────────────────────────────────────────────────────────────────────────────

def _build_offseason_week1(
    schedules: pd.DataFrame,
    team_profiles: dict,
) -> tuple[list[dict], str]:
    """
    Build the best available Week 1 preview for the upcoming season.

    Returns (games, mode) where mode is:
      "scheduled" — real next-season schedule already in nfl_data_py
      "offseason" — proxy from last season's Week 1 (schedule not released yet)
    """
    next_season = get_next_season()
    last_season = next_season - 1

    if not schedules.empty:
        # ── Priority 1: Real next-season schedule ─────────────────────────────
        next_w1 = schedules[
            (schedules["season"] == next_season) &
            (schedules["week"]   == 1)
        ].copy()

        if not next_w1.empty:
            games = _rows_to_games(next_w1, next_season, team_profiles)
            if games:
                logger.info(
                    f"Real {next_season} schedule available — "
                    f"using actual Week 1 ({len(games)} games)."
                )
                return games, "scheduled"

        # ── Priority 2: Previous season Week 1 as proxy ───────────────────────
        last_w1 = schedules[
            (schedules["season"] == last_season) &
            (schedules["week"]   == 1)
        ].copy()

        if not last_w1.empty:
            games = _rows_to_games(last_w1, next_season, team_profiles)
            if games:
                logger.info(
                    f"Using {last_season} Week 1 as proxy for {next_season} "
                    f"({len(games)} games) — real schedule not published yet."
                )
                return games, "offseason"

    # ── Priority 3: Alphabetical fallback ─────────────────────────────────────
    logger.info("Off-season fallback: pairing teams alphabetically.")
    return _generate_matchups_from_profiles(team_profiles, season_override=next_season), "offseason"


def _rows_to_games(
    df: pd.DataFrame,
    season: int,
    team_profiles: dict,
) -> list[dict]:
    """Convert schedule rows to game dicts, filtering to known teams only."""
    games = []
    for _, row in df.iterrows():
        home = _abbr(row.get("home_team", ""))
        away = _abbr(row.get("away_team", ""))
        if home and away and home in team_profiles and away in team_profiles:
            games.append({
                "season":    season,
                "week":      1,
                "home_team": home,
                "away_team": away,
            })
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_matchups_from_profiles(
    profiles: dict,
    season_override: int = None,
) -> list[dict]:
    teams  = sorted(profiles.keys())
    season = season_override or get_relevant_seasons()[-1]
    return [
        {"season": season, "week": 1, "home_team": teams[i], "away_team": teams[i + 1]}
        for i in range(0, len(teams) - 1, 2)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _abbr(name: str) -> str:
    return name.strip().upper() if name else ""