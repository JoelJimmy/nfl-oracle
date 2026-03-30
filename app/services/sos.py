"""
sos.py
Strength of Schedule (SOS) scoring.

SOS = average Elo rating of a team's opponents over the last N games.
A higher SOS means the team has played stronger opponents.

Returns:
  sos_score     float  — raw average opponent Elo
  sos_rank      int    — rank among all 32 teams (1 = hardest schedule)
  sos_percentile float — 0-100, 100 = hardest schedule
"""

from app.logger import get_logger

logger = get_logger(__name__)

# NFL divisions for division game detection
NFL_DIVISIONS = {
    "NFC West":  ["ARI", "LAR", "LA", "SEA", "SF"],
    "NFC East":  ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB",  "MIN"],
    "NFC South": ["ATL", "CAR", "NO",  "TB"],
    "AFC West":  ["DEN", "KC",  "LAC", "LV"],
    "AFC East":  ["BUF", "MIA", "NE",  "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
}

# Flat lookup: team -> division name
_TEAM_DIVISION: dict[str, str] = {}
for div, teams in NFL_DIVISIONS.items():
    for t in teams:
        _TEAM_DIVISION[t] = div


def get_division(team: str) -> str | None:
    return _TEAM_DIVISION.get(team.upper().strip())


def are_division_rivals(team_a: str, team_b: str) -> bool:
    """Return True if two teams are in the same division."""
    da = get_division(team_a)
    db = get_division(team_b)
    return da is not None and da == db


def compute_sos(elo_system, tg_df, window: int = 8) -> dict[str, dict]:
    """
    Compute SOS for all teams using their last `window` opponents' Elo ratings.

    Args:
        elo_system: trained EloSystem object
        tg_df:      team-game DataFrame from feature_engineering (has home_team, away_team, season, week)
        window:     number of recent games to consider

    Returns:
        {team: {"sos_score": float, "sos_rank": int, "sos_percentile": float}}
    """
    if elo_system is None or tg_df is None or tg_df.empty:
        return {}

    # Build opponent list per team from most recent games
    import pandas as pd
    games = tg_df.sort_values(["season", "week"]).copy()

    # Get home/away pairs per game
    home_games = games[games["is_home"] == 1][
        ["game_id", "season", "week", "team", "home_team", "away_team"]
    ].rename(columns={"team": "this_team", "away_team": "opponent"})

    away_games = games[games["is_home"] == 0][
        ["game_id", "season", "week", "team", "home_team", "away_team"]
    ].rename(columns={"team": "this_team", "home_team": "opponent"})

    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games = all_games.sort_values(["season", "week"])

    sos_raw: dict[str, float] = {}

    for team, group in all_games.groupby("this_team"):
        recent_opponents = group.tail(window)["opponent"].tolist()
        if not recent_opponents:
            continue
        opp_elos = [elo_system.get(opp) for opp in recent_opponents
                    if opp in elo_system.ratings]
        if not opp_elos:
            continue
        sos_raw[str(team).upper()] = sum(opp_elos) / len(opp_elos)

    if not sos_raw:
        return {}

    # Rank: 1 = hardest schedule
    sorted_teams = sorted(sos_raw.items(), key=lambda x: -x[1])
    n = len(sorted_teams)

    result = {}
    for rank, (team, score) in enumerate(sorted_teams, 1):
        result[team] = {
            "sos_score":      round(score, 1),
            "sos_rank":       rank,
            "sos_percentile": round((n - rank) / (n - 1) * 100, 1) if n > 1 else 50.0,
            "sos_label":      _sos_label(rank, n),
        }

    logger.info(f"SOS computed for {len(result)} teams. "
                f"Hardest: {sorted_teams[0][0]} ({sorted_teams[0][1]:.0f}), "
                f"Easiest: {sorted_teams[-1][0]} ({sorted_teams[-1][1]:.0f})")
    return result


def _sos_label(rank: int, n: int) -> str:
    pct = (n - rank) / (n - 1) * 100 if n > 1 else 50
    if pct >= 80:
        return "Very Hard"
    if pct >= 60:
        return "Hard"
    if pct >= 40:
        return "Average"
    if pct >= 20:
        return "Easy"
    return "Very Easy"