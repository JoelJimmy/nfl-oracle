"""
roster_adjustment.py

Adjusts team rolling stat profiles based on off-season roster moves.

How it works:
  1. Pull each player's EPA share from the last completed season (nfl_data_py).
  2. Compare last season's team assignments vs current Sleeper rosters to find
     players who changed teams.
  3. For each move, transfer a fraction of that player's EPA contribution
     from their old team's profile to their new team's profile.
  4. Apply a dampening factor — we don't know how well a player will fit
     their new system, so we only transfer 60% of their historical share.

The adjusted profiles are stored separately and used for prediction without
touching the base profiles from training. They reset each time you retrain.

Stat columns adjusted:
  roll_offensive_epa, roll_pass_epa, roll_rush_epa,
  roll_points_scored, roll_turnovers
  (defensive stats are harder to attribute to individuals so left alone)
"""

import warnings
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from copy import deepcopy
from app.services.download_pbp import get_relevant_seasons, is_offseason
from app.logger import get_logger

logger = get_logger(__name__)

# How much of a player's contribution we transfer to their new team.
# 1.0 = full transfer, 0.6 = 60% (accounts for system fit uncertainty)
TRANSFER_DAMPENING = 0.60

# Minimum share to bother adjusting (ignore bench players < 5% of team production)
MIN_SHARE_THRESHOLD = 0.05

# Offensive stats we adjust — keyed to profile roll_ names
ADJUSTABLE_STATS = [
    "roll_offensive_epa",
    "roll_pass_epa",
    "roll_rush_epa",
    "roll_points_scored",
]


# Stores per-team adjustment explanations for the UI
_adjustment_explanations: dict = {}


def build_adjusted_profiles(base_profiles: dict) -> dict:
    """
    Main entry point. Returns a copy of base_profiles with team stats
    adjusted for off-season roster moves. Also populates
    _adjustment_explanations so the UI can explain each adjustment.
    """
    global _adjustment_explanations
    _adjustment_explanations = {}

    if not is_offseason():
        logger.info("[adjustment] In-season — no roster adjustment needed")
        return base_profiles

    logger.info("[adjustment] Building roster-adjusted profiles...")

    try:
        player_shares  = _build_player_shares()
        roster_changes = _detect_roster_changes(player_shares)
        adjusted, explanations = _apply_adjustments_with_explanations(
            base_profiles, player_shares, roster_changes)
        _adjustment_explanations = explanations
        logger.info(f"[adjustment] Adjusted {len(roster_changes)} roster moves "
                    f"across {len(adjusted)} teams")
        return adjusted
    except Exception as e:
        logger.warning(f"[adjustment] Roster adjustment failed: {e} — using base profiles")
        return base_profiles


def get_adjustment_explanations(team: str = None) -> dict:
    """
    Return adjustment explanations.
    If team is given, return just that team's explanations.
    Format: {team: [{player, old_team, new_team, direction, pct_change, stat}]}
    """
    if team:
        return _adjustment_explanations.get(team.upper(), [])
    return _adjustment_explanations


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build player EPA shares from last completed season
# ─────────────────────────────────────────────────────────────────────────────

def _build_player_shares() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per player showing:
      player_id, player_name, last_team, position,
      epa_share, pass_epa_share, rush_epa_share, points_share, to_share
    All share columns sum to 1.0 within each team.
    """
    seasons = get_relevant_seasons()
    last_season = seasons[-1]

    logger.info(f"[adjustment] Pulling player EPA data for {last_season}...")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weekly = nfl.import_weekly_data([last_season])
    except Exception as e:
        raise RuntimeError(f"Could not download weekly data: {e}")

    if weekly is None or weekly.empty:
        raise RuntimeError("Weekly data returned empty")

    # Columns we need
    needed = ["player_id", "player_name", "recent_team", "position",
              "fantasy_points", "passing_epa", "rushing_epa", "receiving_epa",
              "completions", "attempts", "carries", "receptions",
              "passing_yards", "rushing_yards", "receiving_yards",
              "passing_tds", "rushing_tds", "receiving_tds",
              "interceptions", "sack_fumbles_lost"]
    available = [c for c in needed if c in weekly.columns]
    weekly = weekly[available].copy()

    # Fill missing numeric cols with 0
    num_cols = [c for c in available if c not in ["player_id", "player_name",
                                                    "recent_team", "position"]]
    for col in num_cols:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").fillna(0)

    # Aggregate to season totals per player
    agg_cols = {c: "sum" for c in num_cols}
    # Use last known team for each player
    season_stats = (
        weekly.sort_values("recent_team")
        .groupby("player_id")
        .agg({**agg_cols,
              "player_name": "last",
              "recent_team": "last",
              "position":    "last"})
        .reset_index()
    )

    # Compute a combined offensive EPA column
    for epa_col in ["passing_epa", "rushing_epa", "receiving_epa"]:
        if epa_col not in season_stats.columns:
            season_stats[epa_col] = 0.0

    season_stats["total_epa"] = (
        season_stats["passing_epa"] +
        season_stats["rushing_epa"] +
        season_stats["receiving_epa"]
    )

    season_stats["turnovers"] = (
        season_stats.get("interceptions", pd.Series(0, index=season_stats.index)) +
        season_stats.get("sack_fumbles_lost", pd.Series(0, index=season_stats.index))
    )

    season_stats["points_contributed"] = (
        season_stats.get("passing_tds", 0) * 6 +
        season_stats.get("rushing_tds", 0) * 6 +
        season_stats.get("receiving_tds", 0) * 6
    )

    # Compute each player's share of their team's total production
    shares = []
    for team, group in season_stats.groupby("recent_team"):
        team_total_epa    = group["total_epa"].sum() or 1
        team_total_pass   = group["passing_epa"].sum() or 1
        team_total_rush   = (group["rushing_epa"] + group["receiving_epa"]).sum() or 1
        team_total_pts    = group["points_contributed"].sum() or 1
        team_total_to     = group["turnovers"].sum() or 1

        for _, row in group.iterrows():
            epa_share   = row["total_epa"]   / team_total_epa
            pass_share  = row["passing_epa"] / team_total_pass
            rush_share  = (row["rushing_epa"] + row["receiving_epa"]) / team_total_rush
            pts_share   = row["points_contributed"] / team_total_pts
            to_share    = row["turnovers"]    / team_total_to

            # Only include players with meaningful contribution
            if abs(epa_share) < MIN_SHARE_THRESHOLD and abs(pts_share) < MIN_SHARE_THRESHOLD:
                continue

            shares.append({
                "player_id":   str(row["player_id"]),
                "player_name": str(row["player_name"]),
                "last_team":   str(team).upper().strip(),
                "position":    str(row.get("position", "")),
                "epa_share":   float(epa_share),
                "pass_share":  float(pass_share),
                "rush_share":  float(rush_share),
                "pts_share":   float(pts_share),
                "to_share":    float(to_share),
            })

    result = pd.DataFrame(shares)
    logger.info(f"[adjustment] {len(result)} players with significant contributions")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Detect roster changes via Sleeper API
# ─────────────────────────────────────────────────────────────────────────────

def _detect_roster_changes(player_shares: pd.DataFrame) -> list[dict]:
    """
    Compare last season's team (from nfl_data_py) vs current team (from Sleeper)
    for each player in player_shares.

    Returns list of:
      {player_id, player_name, position, old_team, new_team,
       epa_share, pass_share, rush_share, pts_share, to_share}
    """
    import requests
    from app.services.rosters import _normalize_team

    logger.info("[adjustment] Fetching current rosters from Sleeper...")
    try:
        resp = requests.get(
            "https://api.sleeper.app/v1/players/nfl",
            timeout=15
        )
        resp.raise_for_status()
        sleeper_data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Sleeper API failed: {e}")

    # Build {player_id: current_team} from Sleeper
    # Sleeper uses gsis_id which matches nfl_data_py's player_id
    current_teams: dict[str, str] = {}
    for player_id, p in sleeper_data.items():
        team = p.get("team")
        if not team:
            continue
        # Sleeper's player_id isn't gsis_id — try to match via gsis_id field
        gsis = str(p.get("gsis_id") or "").strip()
        if gsis and gsis != "None":
            current_teams[gsis] = _normalize_team(team)
        # Also index by sleeper player_id as fallback
        current_teams[str(player_id)] = _normalize_team(team)

    changes = []
    for _, row in player_shares.iterrows():
        pid      = row["player_id"]
        old_team = row["last_team"]
        new_team = current_teams.get(pid, "").upper().strip()

        if not new_team:
            continue   # player not found in Sleeper (retired, unsigned)
        if new_team == old_team:
            continue   # no change

        changes.append({
            "player_id":   pid,
            "player_name": row["player_name"],
            "position":    row["position"],
            "old_team":    old_team,
            "new_team":    new_team,
            "epa_share":   row["epa_share"],
            "pass_share":  row["pass_share"],
            "rush_share":  row["rush_share"],
            "pts_share":   row["pts_share"],
            "to_share":    row["to_share"],
        })

    logger.info(f"[adjustment] Detected {len(changes)} significant roster moves")
    for c in sorted(changes, key=lambda x: -abs(x["epa_share"]))[:10]:
        logger.info(f"  {c['player_name']} ({c['position']}) "
                    f"{c['old_team']} → {c['new_team']} "
                    f"(EPA share: {c['epa_share']:+.3f})")

    return changes


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Apply adjustments to team profiles
# ─────────────────────────────────────────────────────────────────────────────

def _apply_adjustments(base_profiles, player_shares, changes):
    """Legacy wrapper — calls _apply_adjustments_with_explanations."""
    profiles, _ = _apply_adjustments_with_explanations(base_profiles, player_shares, changes)
    return profiles


def _apply_adjustments_with_explanations(
    base_profiles: dict,
    player_shares: pd.DataFrame,
    changes: list[dict],
) -> tuple[dict, dict]:
    """
    Apply adjustments and return (adjusted_profiles, explanations).
    explanations = {team: [{player, direction, pct_change, stat, old_team, new_team}]}
    """
    profiles = deepcopy(base_profiles)
    explanations: dict = {}

    share_to_stat = {
        "epa_share":  "roll_offensive_epa",
        "pass_share": "roll_pass_epa",
        "rush_share": "roll_rush_epa",
        "pts_share":  "roll_points_scored",
    }

    for move in changes:
        old = move["old_team"]
        new = move["new_team"]
        if old not in profiles or new not in profiles:
            continue

        # Track the largest single adjustment for this move
        max_pct_old = 0.0
        max_pct_new = 0.0
        primary_stat = "roll_offensive_epa"

        for share_col, stat_key in share_to_stat.items():
            share = move[share_col]
            if abs(share) < MIN_SHARE_THRESHOLD:
                continue
            old_val = profiles[old].get(stat_key, 0.0)
            new_val = profiles[new].get(stat_key, 0.0)
            transfer = old_val * share * TRANSFER_DAMPENING
            profiles[old][stat_key] = old_val - transfer
            profiles[new][stat_key] = new_val + transfer
            # Track biggest pct change for explanation
            if old_val != 0 and abs(transfer / old_val) > abs(max_pct_old):
                max_pct_old = transfer / old_val
                primary_stat = stat_key
            if new_val != 0 and abs(transfer / new_val) > abs(max_pct_new):
                max_pct_new = transfer / new_val

        # Turnover adjustment
        to_share = move["to_share"]
        if abs(to_share) >= MIN_SHARE_THRESHOLD:
            old_to = profiles[old].get("roll_turnovers", 0.0)
            new_to = profiles[new].get("roll_turnovers", 0.0)
            transfer_to = old_to * to_share * TRANSFER_DAMPENING
            profiles[old]["roll_turnovers"] = old_to - transfer_to
            profiles[new]["roll_turnovers"] = new_to + transfer_to

        # Build human-readable explanation
        stat_label = {
            "roll_offensive_epa":  "offensive EPA",
            "roll_pass_epa":       "passing EPA",
            "roll_rush_epa":       "rushing EPA",
            "roll_points_scored":  "points scored",
        }.get(primary_stat, primary_stat)

        if abs(max_pct_old) >= 0.01:
            explanations.setdefault(old, []).append({
                "player":     move["player_name"],
                "position":   move["position"],
                "direction":  "lost",
                "pct_change": round(-abs(max_pct_old) * 100, 1),
                "stat":       stat_label,
                "old_team":   old,
                "new_team":   new,
                "epa_share":  round(move["epa_share"] * 100, 1),
            })
        if abs(max_pct_new) >= 0.01:
            explanations.setdefault(new, []).append({
                "player":     move["player_name"],
                "position":   move["position"],
                "direction":  "gained",
                "pct_change": round(abs(max_pct_new) * 100, 1),
                "stat":       stat_label,
                "old_team":   old,
                "new_team":   new,
                "epa_share":  round(move["epa_share"] * 100, 1),
            })

    # Sort each team's explanations by impact
    for team in explanations:
        explanations[team].sort(key=lambda x: -abs(x["pct_change"]))

    return profiles, explanations


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper for the API
# ─────────────────────────────────────────────────────────────────────────────

def get_roster_moves_summary() -> list[dict]:
    """
    Return a list of significant off-season roster moves with their
    projected impact on team predictions. Used by /api/roster-moves.
    """
    try:
        player_shares  = _build_player_shares()
        changes        = _detect_roster_changes(player_shares)
        return sorted(changes, key=lambda x: -abs(x["epa_share"]))
    except Exception as e:
        logger.warning(f"[adjustment] Could not build moves summary: {e}")
        return []