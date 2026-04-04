"""
feature_engineering.py
Comprehensive feature pipeline for NFL game prediction.

46 features total:
  - Rolling 5-game averages (home + away, 8 stats = 16)
  - Differential features (8)
  - EPA trend / slope (5)
  - Win streak (2)
  - Rest days + advantage (3)
  - Turnover differential (1)
  - Elo ratings (4)
  - Head-to-head history (2)
  - Injury impact (4)
  - Home advantage flag (1)
"""

import warnings
import numpy as np
import pandas as pd
from app.config import ROLLING_WINDOW, MIN_GAMES_REQUIRED, ELO_HOME_ADVANTAGE
from app.logger import get_logger

logger = get_logger(__name__)

STAT_COLS = [
    "offensive_epa", "defensive_epa_allowed",
    "pass_epa", "rush_epa",
    "points_scored", "points_allowed",
    "turnovers", "won",
]
ROLL_COLS = [f"roll_{c}" for c in STAT_COLS]


def build_team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PBP into one row per (game_id, team)."""
    logger.info("Aggregating team-game stats from PBP...")

    off = (
        pbp.groupby(["game_id", "posteam"])
        .agg(
            offensive_epa=("epa", "mean"),
            pass_epa=("epa", lambda s: pbp.loc[
                s.index[pbp.loc[s.index, "pass_attempt"] == 1], "epa"
            ].mean() if "pass_attempt" in pbp.columns else np.nan),
            rush_epa=("epa", lambda s: pbp.loc[
                s.index[pbp.loc[s.index, "rush_attempt"] == 1], "epa"
            ].mean() if "rush_attempt" in pbp.columns else np.nan),
            points_scored=("posteam_score_post", "max"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_ = (
        pbp.groupby(["game_id", "defteam"])
        .agg(defensive_epa_allowed=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    to_cols = [c for c in ["interception", "fumble_lost"] if c in pbp.columns]
    if to_cols:
        turnovers = (
            pbp.groupby(["game_id", "posteam"])[to_cols]
            .sum().sum(axis=1).reset_index()
            .rename(columns={"posteam": "team", 0: "turnovers"})
        )
    else:
        turnovers = pd.DataFrame(columns=["game_id", "team", "turnovers"])

    final_scores = (
        pbp.groupby("game_id")
        .agg(home_score=("total_home_score", "max"), away_score=("total_away_score", "max"))
        .reset_index()
    )

    meta_cols = [c for c in ["game_id", "home_team", "away_team", "week", "season", "game_date"]
                 if c in pbp.columns]
    meta = pbp[meta_cols].drop_duplicates("game_id")

    tg = off.merge(def_, on=["game_id", "team"], how="left")
    if len(turnovers):
        tg = tg.merge(turnovers, on=["game_id", "team"], how="left")
    else:
        tg["turnovers"] = 0

    tg = tg.merge(final_scores, on="game_id", how="left")
    tg = tg.merge(meta, on="game_id", how="left")

    tg["is_home"] = (tg["team"] == tg["home_team"]).astype(int)
    tg["points_allowed"] = np.where(tg["is_home"] == 1, tg["away_score"], tg["home_score"])
    tg["won"] = (
        ((tg["is_home"] == 1) & (tg["home_score"] > tg["away_score"])) |
        ((tg["is_home"] == 0) & (tg["away_score"] > tg["home_score"]))
    ).astype(int)

    tg["turnovers"] = tg["turnovers"].fillna(0)
    tg["pass_epa"]  = tg["pass_epa"].fillna(tg["offensive_epa"])
    tg["rush_epa"]  = tg["rush_epa"].fillna(tg["offensive_epa"])

    if "game_date" in tg.columns:
        tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce")
    else:
        tg["game_date"] = pd.NaT

    tg = tg.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    logger.info(f"Team-game rows: {len(tg):,}")
    return tg


def add_rolling_features(tg: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling averages, EPA trends, win streaks, rest days."""
    logger.info(f"Computing rolling {window}-game features + trends...")
    tg = tg.sort_values(["season", "week", "game_id"])

    # Exponentially weighted rolling averages (EWMA)
    # Different span sizes per stat type:
    #   EPA stats: wider span (more games) — EPA is noisy, needs smoothing
    #   Points/wins: narrower span — more stable, recent form matters more
    EPA_COLS   = {"offensive_epa", "defensive_epa_allowed", "pass_epa",
                  "rush_epa", "epa_trend", "def_epa_trend"}
    EPA_SPAN   = max(window + 4, 12)   # e.g. 12 games for EPA smoothing
    OTHER_SPAN = window                 # e.g. 8 games for points/wins

    for col in STAT_COLS:
        span = EPA_SPAN if col in EPA_COLS else OTHER_SPAN
        tg[f"roll_{col}"] = (
            tg.groupby("team")[col]
            .transform(lambda s, sp=span: s.shift(1).ewm(span=sp, min_periods=MIN_GAMES_REQUIRED).mean())
        )

    # EPA trend (slope — positive = improving)
    def _slope(s: pd.Series) -> pd.Series:
        def calc(vals):
            clean = vals.dropna()
            if len(clean) < 3:
                return np.nan
            return float(np.polyfit(np.arange(len(clean)), clean, 1)[0])
        return s.shift(1).rolling(window, min_periods=3).apply(calc, raw=False)

    tg["roll_epa_trend"]     = tg.groupby("team")["offensive_epa"].transform(_slope)
    tg["roll_def_epa_trend"] = tg.groupby("team")["defensive_epa_allowed"].transform(_slope)

    # Win streak
    def _streak(s: pd.Series) -> pd.Series:
        def calc(vals):
            streak = 0
            last = None
            for v in reversed(vals.tolist()):
                if last is None:
                    last = v
                if v == last:
                    streak += (1 if v == 1 else -1)
                else:
                    break
            return float(streak)
        return s.shift(1).rolling(window, min_periods=1).apply(calc, raw=False)

    tg["roll_streak"] = tg.groupby("team")["won"].transform(_streak)

    # Rest days
    if tg["game_date"].notna().any():
        tg["prev_game_date"] = tg.groupby("team")["game_date"].shift(1)
        tg["rest_days"] = (tg["game_date"] - tg["prev_game_date"]).dt.days.clip(4, 14).fillna(7)
    else:
        tg["rest_days"] = 7.0

    return tg


def build_h2h_history(tg: pd.DataFrame) -> dict:
    """Head-to-head win rates for every team pair (last 8 meetings)."""
    logger.info("Building head-to-head history...")
    home_g = tg[tg["is_home"] == 1][["game_id", "team", "won", "season"]].rename(
        columns={"team": "home_team", "won": "home_won"})
    away_g = tg[tg["is_home"] == 0][["game_id", "team"]].rename(columns={"team": "away_team"})
    h2h_raw = home_g.merge(away_g, on="game_id").sort_values("season")

    h2h = {}
    for (ht, at), group in h2h_raw.groupby(["home_team", "away_team"]):
        recent = group.tail(8)
        h2h[(ht, at)] = {"win_rate": float(recent["home_won"].mean()), "games": len(recent)}
    return h2h


def build_injury_impact(weekly_df: pd.DataFrame) -> dict:
    """
    Per-team injury impact: fraction of fantasy production currently on IR/out.
    Returns {team: {'impact': float, 'qb_out': bool}}
    """
    if weekly_df is None or weekly_df.empty or "fantasy_points" not in weekly_df.columns:
        return {}

    logger.info("Computing injury impact scores...")
    last_week = weekly_df.sort_values(["season", "week"]).groupby("player_id").last().reset_index()
    team_totals = last_week.groupby("recent_team")["fantasy_points"].sum().to_dict()

    status_col = next((c for c in ["injury_status", "status"] if c in last_week.columns), None)
    if not status_col:
        return {}

    out_statuses = {"Out", "IR", "Doubtful", "Reserve/Injured"}
    injured = last_week[last_week[status_col].isin(out_statuses)]

    result = {}
    for team in last_week["recent_team"].dropna().unique():
        ti = injured[injured["recent_team"] == team]
        total = team_totals.get(team, 1) or 1
        impact = float(ti["fantasy_points"].sum() / total)
        qb_out = bool("QB" in ti.get("position", pd.Series()).values)
        result[str(team).upper()] = {"impact": min(impact, 1.0), "qb_out": qb_out}

    logger.info(f"Injury impact computed for {len(result)} teams")
    return result


def build_matchup_features(tg, elo, h2h, injury_impact) -> pd.DataFrame:
    """Pivot team-game rows into matchup rows with all features."""
    from app.services.elo import EloSystem
    logger.info("Building matchup feature rows...")

    roll_extra = ROLL_COLS + ["roll_epa_trend", "roll_def_epa_trend", "roll_streak", "rest_days"]

    home = tg[tg["is_home"] == 1][["game_id","season","week","team","won"] + roll_extra].copy()
    away = tg[tg["is_home"] == 0][["game_id","team"] + roll_extra].copy()

    home = home.rename(columns={**{c: f"home_{c}" for c in roll_extra}, "team":"home_team","won":"label"})
    away = away.rename(columns={**{c: f"away_{c}" for c in roll_extra}, "team":"away_team"})

    m = home.merge(away, on="game_id", how="inner")

    # Differentials
    for col in STAT_COLS:
        m[f"diff_{col}"] = m[f"home_roll_{col}"] - m[f"away_roll_{col}"]

    m["diff_epa_trend"] = m["home_roll_epa_trend"] - m["away_roll_epa_trend"]
    m["rest_advantage"] = m["home_rest_days"] - m["away_rest_days"]
    m["turnover_diff"]  = m["home_roll_turnovers"] - m["away_roll_turnovers"]

    # Elo — use pre-game ratings from history
    ELO_CAP = 80.0
    def _cap(v): return 1500 + max(-ELO_CAP, min(ELO_CAP, v - 1500))

    elo_home_map = {h["game_id"]: _cap(h["pre_home_elo"]) for h in elo.history}
    elo_away_map = {h["game_id"]: _cap(h["pre_away_elo"]) for h in elo.history}
    m["home_elo"] = m["game_id"].map(elo_home_map).fillna(m["home_team"].map(lambda t: _cap(elo.get(t))))
    m["away_elo"] = m["game_id"].map(elo_away_map).fillna(m["away_team"].map(lambda t: _cap(elo.get(t))))
    m["elo_diff"]     = m["home_elo"] - m["away_elo"]


    # Injury
    m["home_injury_impact"] = m["home_team"].map(lambda t: injury_impact.get(t,{}).get("impact",0.0))
    m["away_injury_impact"] = m["away_team"].map(lambda t: injury_impact.get(t,{}).get("impact",0.0))
    m["home_qb_out"]        = m["home_team"].map(lambda t: float(injury_impact.get(t,{}).get("qb_out",False)))
    m["away_qb_out"]        = m["away_team"].map(lambda t: float(injury_impact.get(t,{}).get("qb_out",False)))

    m["home_advantage"] = 1

    m = m.dropna(subset=get_feature_columns())
    logger.info(f"Matchup rows (clean): {len(m):,}")
    return m.reset_index(drop=True)


def build_current_team_profiles(tg, injury_impact, window=ROLLING_WINDOW) -> dict:
    """Current team rolling profiles for upcoming game prediction.

    During the off-season, uses the full most-recent season average rather than
    just the last N games — this avoids end-of-season resting/garbage-time skew.
    During the regular season, uses the last N games as a rolling window.
    """
    from app.services.download_pbp import is_offseason, get_relevant_seasons
    logger.info("Building current team profiles...")
    profiles = {}

    # In off-season, use full most-recent season (not just last 5 games)
    # This prevents "KC rested starters in Week 18" from ruining their profile
    offseason = is_offseason()
    if offseason:
        latest_season = max(tg["season"].unique())
        logger.info(f"Off-season mode: using full {latest_season} season for profiles")

    for team, group in tg.groupby("team"):
        group = group.sort_values(["season", "week"])
        if offseason:
            # Use full most-recent season for off-season profiles
            latest = group[group["season"] == group["season"].max()]
            # Exclude last 3 games (Weeks 16-18 often have resting starters
            # in meaningless games; playoff teams especially rest in Week 18)
            recent = latest.iloc[:-3] if len(latest) > 6 else latest
            if len(recent) < MIN_GAMES_REQUIRED:
                recent = latest  # fall back to full season if too short
        else:
            recent = group.tail(window)
        if len(recent) < MIN_GAMES_REQUIRED:
            continue

        EPA_COLS   = {"offensive_epa", "defensive_epa_allowed", "pass_epa",
                      "rush_epa", "epa_trend", "def_epa_trend"}
        EPA_SPAN   = max(window + 4, 12)
        OTHER_SPAN = window

        stats = {}
        for col in STAT_COLS:
            span = EPA_SPAN if col in EPA_COLS else OTHER_SPAN
            ewma_vals = recent[col].ewm(span=span, min_periods=1).mean()
            val = float(ewma_vals.iloc[-1])
            # Cap win rate to realistic mid-season ranges
            if col == "won":
                val = min(0.80, max(0.20, val))
            stats[f"roll_{col}"] = val

        epa_vals = recent["offensive_epa"].dropna().values
        stats["roll_epa_trend"] = float(np.polyfit(
            np.arange(len(epa_vals)), epa_vals, 1)[0]) if len(epa_vals) >= 2 else 0.0

        def_vals = recent["defensive_epa_allowed"].dropna().values
        stats["roll_def_epa_trend"] = float(np.polyfit(
            np.arange(len(def_vals)), def_vals, 1)[0]) if len(def_vals) >= 2 else 0.0

        streak = 0
        last = None
        for v in reversed(recent["won"].tolist()):
            if last is None: last = v
            if v == last: streak += (1 if v == 1 else -1)
            else: break
        stats["roll_streak"] = float(max(-3.0, min(3.0, streak)))
        stats["rest_days"]   = 7.0

        inj = injury_impact.get(str(team).upper(), {})
        stats["injury_impact"] = float(inj.get("impact", 0.0))
        stats["qb_out"]        = float(inj.get("qb_out", False))

        profiles[str(team).upper()] = stats

    logger.info(f"Profiles built for {len(profiles)} teams")
    return profiles


def get_feature_columns() -> list[str]:
    """Ordered feature list — must match model training exactly."""
    home = [f"home_roll_{c}" for c in STAT_COLS]
    away = [f"away_roll_{c}" for c in STAT_COLS]
    diff = [f"diff_{c}" for c in STAT_COLS]
    return (
        home + away + diff + [
            "home_roll_epa_trend", "away_roll_epa_trend",
            "home_roll_def_epa_trend", "away_roll_def_epa_trend",
            "diff_epa_trend",
            "home_roll_streak", "away_roll_streak",
            "home_rest_days", "away_rest_days", "rest_advantage",
            "turnover_diff",
            "home_elo", "away_elo", "elo_diff",
            "home_injury_impact", "away_injury_impact",
            "home_qb_out", "away_qb_out",
            "home_advantage",
        ]
    )


def build_full_dataset(pbp, weekly=None):
    """Run the complete feature engineering pipeline with progress logging."""
    from app.services.elo import build_elo_from_games
    import traceback as _tb

    print("build_full_dataset: step 1 start", flush=True)
    try:
        tg = build_team_game_stats(pbp)
        print(f"build_full_dataset: step 1 done — {len(tg):,} rows", flush=True)
    except Exception as e:
        print(f"STEP 1 FAILED: {e}", flush=True)
        print(_tb.format_exc(), flush=True)
        raise
    logger.info(f"  -> {len(tg):,} team-game rows")

    print("build_full_dataset: step 2 start", flush=True)
    try:
        tg = add_rolling_features(tg)
        print("build_full_dataset: step 2 done", flush=True)
    except Exception as e:
        print(f"STEP 2 FAILED: {e}", flush=True)
        print(_tb.format_exc(), flush=True)
        raise

    print("build_full_dataset: step 3 start", flush=True)
    try:
        h2h = build_h2h_history(tg)
        print(f"build_full_dataset: step 3 done — {len(h2h)} pairs", flush=True)
    except Exception as e:
        print(f"STEP 3 FAILED: {e}", flush=True)
        print(_tb.format_exc(), flush=True)
        raise
    logger.info(f"  -> {len(h2h):,} H2H matchup records")

    logger.info("Step 4/6: Computing injury impact...")
    injury = build_injury_impact(weekly) if weekly is not None else {}

    print("build_full_dataset: step 5 start (Elo)", flush=True)
    game_scores = tg[tg["is_home"] == 1][
        ["game_id", "season", "week", "team", "home_score", "away_score"]
    ].rename(columns={"team": "home_team"})
    away_teams = tg[tg["is_home"] == 0][["game_id", "team"]].rename(columns={"team": "away_team"})
    games_for_elo = game_scores.merge(away_teams, on="game_id").dropna(
        subset=["home_score", "away_score"])
    logger.info(f"  -> {len(games_for_elo):,} completed games")
    try:
        elo = build_elo_from_games(games_for_elo)
        print("build_full_dataset: step 5 done (Elo)", flush=True)
    except Exception as e:
        print(f"STEP 5 FAILED: {e}", flush=True)
        print(_tb.format_exc(), flush=True)
        raise

    print("build_full_dataset: step 6 start (matchups)", flush=True)
    try:
        matchups = build_matchup_features(tg, elo, h2h, injury)
        print(f"build_full_dataset: step 6 done — {len(matchups)} matchups", flush=True)
    except Exception as e:
        print(f"STEP 6 FAILED: {e}", flush=True)
        print(_tb.format_exc(), flush=True)
        raise
    logger.info(f"  -> {len(matchups):,} matchup rows, {len(get_feature_columns())} features each")

    profiles = build_current_team_profiles(tg, injury)
    return matchups, profiles, elo, tg, h2h, injury