"""
rosters.py
Three-layer roster pipeline:

  Layer 1 — Sleeper API (free, no key, updates within hours)
    https://api.sleeper.app/v1/players/nfl
    Provides: player names, team, position, jersey, depth_chart_order,
              status, injury_status, age, experience, height, weight

  Layer 2 — nfl_data_py import_depth_charts (weekly ESPN snapshots)
    Provides: official depth order when Sleeper is unavailable

  Layer 3 — nfl_data_py import_seasonal_data
    Provides: cumulative season stats (passing_yards, rushing_tds, etc.)

Merge strategy:
  - Sleeper is the authoritative source for player identity + depth order
  - If Sleeper fails, fall back to nfl_data_py depth charts
  - Stats always come from nfl_data_py seasonal data (keyed by gsis_id / espn_id)

Cache:
  - Full roster cached with a TTL of 6 hours (re-fetches automatically)
  - Slim roster (2 key players per team) cached separately for prediction chips
"""

import warnings
import time
import requests
import nfl_data_py as nfl
import pandas as pd
from app.services.download_pbp import get_relevant_seasons

# ── Cache ─────────────────────────────────────────────────────────────────────
_roster_cache       = None
_full_roster_cache  = None
_full_cache_time    = 0
_CACHE_TTL_SECONDS  = 6 * 3600   # 6 hours

# ── Sleeper API ───────────────────────────────────────────────────────────────
_SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# Positions we care about (Sleeper uses fantasy_positions list as well)
_ALL_POSITIONS = {
    "QB", "RB", "FB", "WR", "TE",
    "T", "G", "C", "OT", "OG",
    "DE", "DT", "NT",
    "OLB", "ILB", "MLB", "LB",
    "CB", "S", "SS", "FS", "DB",
    "K", "P", "LS",
}

# ── Position normalization ────────────────────────────────────────────────────
_POS_NORMALIZE = {
    "LT": "T",  "RT": "T",  "OT": "T",
    "LG": "G",  "RG": "G",  "OG": "G",
    "LE": "DE", "RE": "DE", "DL": "DE",
    "LOLB": "OLB", "ROLB": "OLB",
    "LILB": "ILB", "RILB": "ILB", "WILB": "ILB",
    "LCB": "CB",   "RCB": "CB",
    "SAF": "S",
    "KR": "K",  "PR": "P",
}

def _normalize_pos(p: str) -> str:
    return _POS_NORMALIZE.get(str(p).upper().strip(), str(p).upper().strip())

# ── NFL team abbreviation normalization ───────────────────────────────────────
# Sleeper uses some different abbreviations than nfl_data_py
_TEAM_NORMALIZE = {
    "JAC": "JAX",
    "SFO": "SF",
    "KCC": "KC",
    "GBP": "GB",
    "NOS": "NO",
    "NEP": "NE",
    "TBB": "TB",
    "LA":  "LAR",   # Sleeper uses "LA" for the Rams
    "LAR": "LAR",
    "LVR": "LV",
    "ARZ": "ARI",
    "HST": "HOU",
    "CLV": "CLE",
    "BLT": "BAL",
    "SD":  "LAC",
    "OAK": "LV",
}

def _normalize_team(t: str) -> str:
    if not t:
        return ""
    t = str(t).upper().strip()
    return _TEAM_NORMALIZE.get(t, t)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_rosters() -> dict:
    """Slim roster — 2 key players per team for the prediction page chips."""
    global _roster_cache
    season = get_relevant_seasons()[-1]
    print(f"[rosters] Building slim rosters for {season}...")

    players_by_team = _get_players_by_team(season)
    if not players_by_team:
        _roster_cache = {}
        return {}

    result = {}
    for team, players in players_by_team.items():
        chips = []
        qbs = [p for p in players if p["position"] == "QB"]
        qbs.sort(key=lambda p: p.get("depth_order", 99))
        if qbs:
            chips.append(_to_slim(qbs[0]))
        for pos in ["WR", "RB", "TE"]:
            starters = [p for p in players if p["position"] == pos]
            starters.sort(key=lambda p: p.get("depth_order", 99))
            if starters:
                chips.append(_to_slim(starters[0]))
                break
        if chips:
            result[team] = chips

    print(f"[rosters] Slim rosters built for {len(result)} teams")
    _roster_cache = result
    return result


def load_full_rosters() -> dict:
    """Full depth-chart-ordered roster with season stats for every team."""
    global _full_roster_cache, _full_cache_time

    season = get_relevant_seasons()[-1]
    now = time.time()

    # Return cached if fresh
    if _full_roster_cache and (now - _full_cache_time) < _CACHE_TTL_SECONDS:
        print(f"[rosters] Returning cached full rosters ({int((now - _full_cache_time)/60)} min old)")
        return _full_roster_cache

    print(f"[rosters] Building full rosters for {season}...")

    players_by_team = _get_players_by_team(season)
    if not players_by_team:
        _full_roster_cache = {}
        _full_cache_time = now
        return {}

    full = {}
    for team, players in players_by_team.items():
        full[team] = {
            "season":  int(season),
            "team":    team,
            "players": sorted(players, key=lambda p: (p["position"], p.get("depth_order", 99))),
        }

    print(f"[rosters] Full rosters built for {len(full)} teams")
    _full_roster_cache = full
    _full_cache_time = now
    return full


def get_rosters() -> dict:
    global _roster_cache
    if _roster_cache is None:
        return load_rosters()
    return _roster_cache


def get_full_rosters() -> dict:
    return load_full_rosters()


# ─────────────────────────────────────────────────────────────────────────────
# Core: get normalized player list per team
# ─────────────────────────────────────────────────────────────────────────────

def _get_players_by_team(season: int) -> dict:
    """
    Returns {team_abbr: [player_dict, ...]} using:
      - Sleeper API for player identity + depth order  (primary)
      - nfl_data_py depth charts                       (fallback)
    Always enriches with nfl_data_py season stats.
    """
    # Try Sleeper first
    players_by_team = _fetch_from_sleeper()

    if not players_by_team:
        print("[rosters] Sleeper failed — falling back to nfl_data_py depth charts")
        players_by_team = _fetch_from_nfl_data_py(season)

    if not players_by_team:
        print("[rosters] Both sources failed — no roster data available")
        return {}

    # Enrich with season stats regardless of source
    stats = _fetch_season_stats(season)
    for team_players in players_by_team.values():
        for p in team_players:
            pid = p.get("gsis_id") or p.get("espn_id") or ""
            p["stats"] = stats.get(str(pid), {})

    return players_by_team


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Sleeper API
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_sleeper() -> dict:
    """
    Fetch all NFL players from Sleeper API.
    Returns {team_abbr: [player_dict]} or {} on failure.

    Sleeper player fields used:
      full_name, first_name, last_name
      team                 — team abbr (may need normalization)
      position             — canonical pos (QB, RB, WR, TE, K, DEF...)
      depth_chart_position — slot (QB, WR1, WR2, WR3, etc.)
      depth_chart_order    — 1=starter, 2=backup, 3=third string
      status               — "Active", "Inactive", "IR", "PUP", "Suspended"
      injury_status        — "Questionable", "Doubtful", "Out", "IR", null
      injury_body_part     — e.g. "Knee"
      jersey_number
      years_exp            — 0 = rookie
      age
      espn_id, sportradar_id, gsis_id
    """
    print("[rosters] Fetching from Sleeper API...")
    try:
        resp = requests.get(_SLEEPER_PLAYERS_URL, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.Timeout:
        print("[rosters] Sleeper API timed out")
        return {}
    except requests.exceptions.ConnectionError:
        print("[rosters] Sleeper API unreachable")
        return {}
    except Exception as e:
        print(f"[rosters] Sleeper API error: {e}")
        return {}

    print(f"[rosters] Sleeper returned {len(raw)} total players")

    by_team: dict = {}
    skipped = 0

    for player_id, p in raw.items():
        team = _normalize_team(p.get("team") or "")
        pos  = _normalize_pos(p.get("position") or "")

        # Skip: no team (free agents, retired), no valid position, DEF/IDP
        if not team or not pos or pos in {"DEF", "IDP", ""}:
            skipped += 1
            continue

        # Skip practice squad (status == "Practice Squad" or "PS")
        status = str(p.get("status") or "").strip()
        if status in {"Practice Squad", "PS"}:
            skipped += 1
            continue

        raw_depth = p.get("depth_chart_order")
        try:
            depth_order = int(raw_depth) if raw_depth is not None else 99
        except (ValueError, TypeError):
            depth_order = 99

        injury_status = p.get("injury_status") or ""
        display_status = ""
        if status == "IR" or injury_status == "IR":
            display_status = "IR"
        elif status == "PUP":
            display_status = "PUP"
        elif status == "Suspended":
            display_status = "SUS"
        elif injury_status in ("Out", "Doubtful"):
            display_status = injury_status
        elif injury_status == "Questionable":
            display_status = "Q"

        player = {
            "name":          str(p.get("full_name") or
                                 f"{p.get('first_name','')} {p.get('last_name','')}").strip(),
            "position":      pos,
            "depth_slot":    str(p.get("depth_chart_position") or ""),
            "depth_order":   depth_order,
            "jersey":        str(p.get("jersey_number") or "").replace(".0","").strip(),
            "years_exp":     int(p.get("years_exp") or 0),
            "age":           int(p.get("age") or 0),
            "status":        display_status,
            "injury_body":   str(p.get("injury_body_part") or ""),
            "gsis_id":       str(p.get("gsis_id") or "").strip(),
            "espn_id":       str(p.get("espn_id") or "").strip(),
            "sleeper_id":    str(player_id),
        }

        if team not in by_team:
            by_team[team] = []
        by_team[team].append(player)

    print(f"[rosters] Sleeper: {sum(len(v) for v in by_team.values())} players "
          f"across {len(by_team)} teams (skipped {skipped})")
    return by_team


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: nfl_data_py depth charts (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_nfl_data_py(season: int) -> dict:
    """Fall back to nfl_data_py import_depth_charts."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dc = nfl.import_depth_charts([season])
    except Exception as e:
        print(f"[rosters] import_depth_charts failed: {e}")
        return {}

    if dc is None or dc.empty:
        return {}

    # Use most recent snapshot
    if "dt" in dc.columns:
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        latest = dc.groupby("team")["dt"].transform("max")
        dc = dc[dc["dt"] == latest].copy()

    dc["pos_rank"] = pd.to_numeric(dc.get("pos_rank", 99), errors="coerce").fillna(99).astype(int)
    dc["pos_abb"]  = dc.get("pos_abb", pd.Series("", index=dc.index)).fillna("").astype(str).str.upper()
    dc["pos_abb"]  = dc["pos_abb"].map(_normalize_pos)

    by_team: dict = {}
    for _, row in dc.iterrows():
        team = _normalize_team(str(row.get("team", "")))
        if not team:
            continue
        player = {
            "name":        str(row.get("player_name", "Unknown")),
            "position":    str(row.get("pos_abb", "")),
            "depth_slot":  str(row.get("pos_slot", "")),
            "depth_order": int(row.get("pos_rank", 99)),
            "jersey":      "",
            "years_exp":   0,
            "age":         0,
            "status":      "",
            "injury_body": "",
            "gsis_id":     str(row.get("gsis_id", "") or "").strip(),
            "espn_id":     str(row.get("espn_id", "") or "").strip(),
            "sleeper_id":  "",
        }
        if team not in by_team:
            by_team[team] = []
        by_team[team].append(player)

    print(f"[rosters] nfl_data_py fallback: {sum(len(v) for v in by_team.values())} players "
          f"across {len(by_team)} teams")
    return by_team


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Season stats (nfl_data_py)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_season_stats(season: int) -> dict:
    """
    Returns {player_id: {stat: value}} for the current season only.
    During the off-season (Feb-Aug) stats are intentionally not loaded —
    the UI will show dashes rather than stale previous-season numbers.
    """
    from app.services.download_pbp import is_offseason
    if is_offseason():
        print("[rosters] Off-season — skipping stats (will show — in UI)")
        return {}

    wanted = [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "fantasy_points",
    ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = nfl.import_seasonal_data([season])
        if s is not None and not s.empty:
            print(f"[rosters] Season stats loaded for {season}: {len(s)} rows")
            return _df_to_stats_dict(s, wanted)
    except Exception as e:
        print(f"[rosters] import_seasonal_data({season}) failed: {e}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = nfl.import_weekly_data([season])
        if w is not None and not w.empty:
            print(f"[rosters] Weekly stats loaded for {season}: {len(w)} rows")
            return _df_to_stats_dict(w, wanted, aggregate=True)
    except Exception as e:
        print(f"[rosters] import_weekly_data({season}) failed: {e}")

    print("[rosters] No stats available — depth chart only")
    return {}

def _df_to_stats_dict(df: pd.DataFrame, wanted: list, aggregate: bool = False) -> dict:
    id_col = next((c for c in ["player_id", "gsis_id"] if c in df.columns), None)
    if not id_col:
        return {}

    keep = [id_col] + [c for c in wanted if c in df.columns]
    df = df[keep].copy()
    df[id_col] = df[id_col].astype(str).str.strip()

    numeric = [c for c in keep if c != id_col]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if aggregate:
        df = df.groupby(id_col)[numeric].sum().reset_index()

    result = {}
    for _, row in df.iterrows():
        pid = str(row[id_col])
        if pid and pid != "nan":
            result[pid] = {c: _safe(row[c]) for c in numeric}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_slim(p: dict) -> dict:
    return {
        "name":      p.get("name", "Unknown"),
        "position":  p.get("position", ""),
        "jersey":    p.get("jersey", ""),
        "years_exp": p.get("years_exp", 0),
    }


def _safe(v) -> float:
    try:
        f = float(v)
        return 0.0 if f != f else round(f, 1)
    except Exception:
        return 0.0