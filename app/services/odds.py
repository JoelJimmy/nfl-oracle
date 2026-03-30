"""
odds.py
Fetches current NFL Vegas odds from The Odds API (free tier supports ~500 req/month).
FanDuel is included as a bookmaker key: "fanduel".

To use:
  1. Sign up free at https://the-odds-api.com
  2. Set the env var:  ODDS_API_KEY=your_key_here
  3. The app will automatically fetch live odds on startup.

If no API key is set, the app falls back to estimated spreads derived
from team rolling stats.
"""

import os
import requests
from typing import Optional

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORT_KEY = "americanfootball_nfl"
BOOKMAKER = "fanduel"


def fetch_fanduel_odds() -> list[dict]:
    """
    Fetch upcoming NFL game odds from FanDuel via The Odds API.
    Returns a list of dicts with keys:
        home_team, away_team, home_moneyline, away_moneyline,
        spread, spread_team, total
    Returns empty list if API key not set or request fails.
    """
    if not ODDS_API_KEY:
        print("[odds] No ODDS_API_KEY set — skipping live odds fetch.")
        return []

    url = f"{ODDS_API_BASE}/{SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "bookmakers": BOOKMAKER,
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[odds] Odds API request failed: {e}")
        return []

    results = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        entry = {
            "home_team": _normalize_team(home),
            "away_team": _normalize_team(away),
            "home_moneyline": None,
            "away_moneyline": None,
            "spread": None,
            "spread_team": None,
            "total": None,
            "commence_time": game.get("commence_time"),
        }

        for bm in game.get("bookmakers", []):
            if bm["key"] != BOOKMAKER:
                continue
            for market in bm.get("markets", []):
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if market["key"] == "h2h":
                    entry["home_moneyline"] = outcomes.get(home, {}).get("price")
                    entry["away_moneyline"] = outcomes.get(away, {}).get("price")

                elif market["key"] == "spreads":
                    ho = outcomes.get(home, {})
                    entry["spread"] = ho.get("point")
                    entry["spread_team"] = _normalize_team(home) if ho else None

                elif market["key"] == "totals":
                    over = outcomes.get("Over", {})
                    entry["total"] = over.get("point")

        results.append(entry)

    print(f"[odds] Fetched {len(results)} games from FanDuel via The Odds API.")
    return results


def estimate_odds_from_profiles(home_profile: dict, away_profile: dict,
                                home_win_prob: float = None) -> dict:
    """
    Fallback: estimate spread & moneyline from the model's win probability
    (preferred) or from rolling team stats when no probability is available.

    Uses the standard sportsbook conversion:
      spread ≈ (win_prob - 0.5) * 14   (each point of spread ≈ 3.5% win prob shift)
      moneyline from spread via standard table
    """
    h_pts = home_profile.get("roll_points_scored", 21.0)
    a_pts = away_profile.get("roll_points_scored", 21.0)
    h_def = home_profile.get("roll_points_allowed", 21.0)
    a_def = away_profile.get("roll_points_allowed", 21.0)

    if home_win_prob is not None:
        # Derive spread directly from model win probability
        # ~3.5% win prob per point of spread (standard Vegas conversion)
        # Negative spread = home favored
        expected_diff = (home_win_prob - 0.5) * 14.0
        spread = round(-expected_diff, 1)
    else:
        # Fallback: use points stats + home field advantage
        expected_diff = (h_pts - a_def) / 2 - (a_pts - h_def) / 2 + 2.5
        spread = round(-expected_diff, 1)

    # Convert spread to moneyline
    def spread_to_ml(spread_val: float) -> tuple[int, int]:
        """Convert point spread to approximate moneylines."""
        s = abs(spread_val)
        if s <= 0.5:
            return -110, -110
        # Standard book conversion table
        if s <= 1:   return -120, 100
        if s <= 1.5: return -130, 110
        if s <= 2:   return -140, 120
        if s <= 2.5: return -150, 130
        if s <= 3:   return -165, 145
        if s <= 3.5: return -180, 155
        if s <= 4:   return -190, 165
        if s <= 4.5: return -200, 170
        if s <= 5:   return -215, 180
        if s <= 5.5: return -225, 190
        if s <= 6:   return -240, 200
        if s <= 6.5: return -260, 215
        if s <= 7:   return -300, 250
        if s <= 7.5: return -330, 270
        if s <= 8:   return -350, 285
        if s <= 10:  return -400, 320
        if s <= 13:  return -500, 380
        return -600, 450

    fav_ml, dog_ml = spread_to_ml(spread)
    if spread <= 0:   # home favored
        home_ml, away_ml = fav_ml, dog_ml
    else:             # away favored
        home_ml, away_ml = dog_ml, fav_ml

    # Total from average scoring
    h_total_pts = (h_pts + a_def) / 2
    a_total_pts = (a_pts + h_def) / 2
    total = round((h_total_pts + a_total_pts), 1)

    return {
        "home_moneyline": int(home_ml),
        "away_moneyline": int(away_ml),
        "spread": float(spread),
        "spread_team": None,
        "total": float(total),
        "source": "estimated",
    }


# Map full team names (from The Odds API) to NFL abbreviations
_TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _normalize_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)