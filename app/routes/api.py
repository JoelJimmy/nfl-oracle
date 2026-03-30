"""
api.py — REST API with rate limiting, input validation, and caching.
"""

import os
import time
from functools import wraps
from flask import Blueprint, jsonify, request
from app.cache import cache
from app.config import (
    CACHE_TTL_UPCOMING, CACHE_TTL_MODEL_INFO,
    CACHE_TTL_ROSTERS, CACHE_TTL_ELO,
)
from app.services.predictor import (
    predict_matchup, predict_bulk, get_model_meta,
    get_all_teams, get_team_profile, update_team_profile, is_loaded,
)
from app.services.schedule import get_upcoming_games
from app.services.rosters import get_rosters, get_full_rosters
from app.services.scheduler import get_next_jobs
from app.services import predictor as pred_module
from app.logger import get_logger

logger = get_logger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")

# ── Rate limiting ─────────────────────────────────────────────────────────────
_rate_store: dict = {}
RATE_LIMIT = 60

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip  = request.remote_addr or "unknown"
        now = time.time()
        hits = [t for t in _rate_store.get(ip, []) if t > now - 60]
        if len(hits) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded — max 60 requests/minute"}), 429
        hits.append(now)
        _rate_store[ip] = hits
        return f(*args, **kwargs)
    return decorated


# ── Health & status ────────────────────────────────────────────────────────────

@api_bp.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": is_loaded()})






@api_bp.route("/status")
def status():
    from app.services.download_pbp import is_offseason
    return jsonify({
        "model_loaded":   is_loaded(),
        "offseason":      is_offseason(),
        "scheduled_jobs": get_next_jobs(),
    })


@api_bp.route("/model-info")
@rate_limit
@cache.cached(timeout=CACHE_TTL_MODEL_INFO, key_prefix="model_info")
def model_info():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(get_model_meta())


# ── Teams ─────────────────────────────────────────────────────────────────────

@api_bp.route("/teams")
@rate_limit
def teams():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({"teams": [
        {"abbr": a, "profile": get_team_profile(a)} for a in get_all_teams()
    ]})


@api_bp.route("/teams/<team>")
@rate_limit
def team_detail(team: str):
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        return jsonify({"team": team.upper(), "profile": get_team_profile(team.upper())})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@api_bp.route("/teams/<team>/elo")
@rate_limit
def team_elo(team: str):
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        from app.services.predictor import _elo_system
        if not _elo_system:
            return jsonify({"error": "Elo not available"}), 503
        t = team.upper()
        return jsonify({
            "team":        t,
            "elo_rating":  round(_elo_system.get(t), 1),
            "power_score": _elo_system.to_display_score(t),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Upcoming games ─────────────────────────────────────────────────────────────

@api_bp.route("/upcoming")
@rate_limit
def upcoming():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    team_filter = request.args.get("team", "").upper()
    cache_key   = f"upcoming_{team_filter or 'all'}"

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)

    profiles = pred_module._team_profiles
    games    = get_upcoming_games(profiles)
    preds    = predict_bulk([{"home_team": g["home_team"], "away_team": g["away_team"]}
                              for g in games])
    pred_map = {(p["home_team"], p["away_team"]): p for p in preds if "error" not in p}
    result   = [{**g, **pred_map.get((g["home_team"], g["away_team"]), {})} for g in games]

    if team_filter:
        result = [r for r in result
                  if r.get("home_team") == team_filter or r.get("away_team") == team_filter]

    mode     = result[0].get("mode", "inseason") if result else "inseason"
    response = {"games": result, "count": len(result), "mode": mode}

    cache.set(cache_key, response, timeout=CACHE_TTL_UPCOMING)
    return jsonify(response)


# ── Predict ────────────────────────────────────────────────────────────────────

@api_bp.route("/predict", methods=["POST"])
@rate_limit
def predict():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True) or {}
    home = str(data.get("home_team", "")).upper().strip()
    away = str(data.get("away_team", "")).upper().strip()

    if not home or not away:
        return jsonify({"error": "home_team and away_team are required"}), 400
    if home == away:
        return jsonify({"error": "Teams must be different"}), 400
    if len(home) > 10 or len(away) > 10:
        return jsonify({"error": "Invalid team abbreviation"}), 400

    cache_key = f"predict_{home}_{away}_{data.get('home_rest_days',7)}_{data.get('away_rest_days',7)}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)

    try:
        result = predict_matchup(
            home, away,
            float(data.get("home_rest_days", 7)),
            float(data.get("away_rest_days", 7)),
        )
        hp = get_team_profile(home)
        ap = get_team_profile(away)
        from app.services.odds import estimate_odds_from_profiles
        result.update(estimate_odds_from_profiles(
            hp, ap, home_win_prob=result.get("home_win_probability")))
        result["odds_source"] = "estimated"
        cache.set(cache_key, result, timeout=CACHE_TTL_UPCOMING)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ── Rosters ────────────────────────────────────────────────────────────────────

@api_bp.route("/rosters")
@rate_limit
def rosters():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    cached = cache.get("rosters_slim")
    if cached:
        return jsonify(cached)
    data = {"rosters": get_rosters(), "count": len(get_rosters())}
    cache.set("rosters_slim", data, timeout=CACHE_TTL_ROSTERS)
    return jsonify(data)


@api_bp.route("/rosters/full")
@rate_limit
def rosters_full():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    team_filter = request.args.get("team", "").upper()
    cache_key   = f"rosters_full_{team_filter or 'all'}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)
    try:
        data = get_full_rosters()
        if team_filter:
            data = {team_filter: data[team_filter]} if team_filter in data else {}
        response = {"rosters": data, "count": len(data)}
        cache.set(cache_key, response, timeout=CACHE_TTL_ROSTERS)
        return jsonify(response)
    except Exception as e:
        import traceback
        logger.error(f"/rosters/full error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/rosters/debug")
def rosters_debug():
    try:
        import warnings, nfl_data_py as nfl, numpy as np
        from app.services.download_pbp import get_relevant_seasons
        season = get_relevant_seasons()[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = nfl.import_depth_charts([season])
        def clean(v):
            if isinstance(v, np.integer): return int(v)
            if isinstance(v, np.floating): return float(v)
            if v != v: return None
            return str(v) if not isinstance(v, (int, float, bool, type(None))) else v
        sample = [{k: clean(v) for k, v in row.items()}
                  for row in raw.head(3).to_dict(orient="records")]
        club_col = "club_code" if "club_code" in raw.columns else "team"
        return jsonify({
            "season": season, "total_rows": len(raw),
            "columns": raw.columns.tolist(),
            "team_count": int(raw[club_col].nunique()) if club_col in raw.columns else 0,
            "available_functions": [f for f in dir(nfl) if not f.startswith("_")],
            "sample_rows": sample,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Roster moves ──────────────────────────────────────────────────────────────

@api_bp.route("/roster-moves")
@rate_limit
def roster_moves():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    cached = cache.get("roster_moves")
    if cached:
        return jsonify(cached)
    try:
        from app.services.roster_adjustment import get_roster_moves_summary
        from app.services.download_pbp import is_offseason
        moves    = get_roster_moves_summary()
        response = {
            "moves":     moves,
            "count":     len(moves),
            "offseason": is_offseason(),
            "note": "EPA share shows what fraction of the old team's production this player "
                    "represented. Moves sorted by impact (highest first).",
        }
        cache.set("roster_moves", response, timeout=CACHE_TTL_ROSTERS)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Elo rankings ──────────────────────────────────────────────────────────────

@api_bp.route("/elo")
@rate_limit
@cache.cached(timeout=CACHE_TTL_ELO, key_prefix="elo_rankings")
def elo_rankings():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        from app.services.predictor import _elo_system
        if not _elo_system:
            return jsonify({"error": "Elo not available"}), 503
        rankings = [
            {"team": t, "elo": round(r, 1), "power_score": _elo_system.to_display_score(t)}
            for t, r in sorted(_elo_system.ratings.items(), key=lambda x: -x[1])
        ]
        return jsonify({"rankings": rankings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Head-to-head history ──────────────────────────────────────────────────────

@api_bp.route("/h2h/<home>/<away>")
@rate_limit
def h2h(home: str, away: str):
    """Return head-to-head history between two teams."""
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    h = home.upper().strip()
    a = away.upper().strip()
    cache_key = f"h2h_{h}_{a}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)
    try:
        h2h_data  = pred_module._model_artifact.get("h2h", {})
        home_data = h2h_data.get((h, a), {})
        away_data = h2h_data.get((a, h), {})
        response  = {
            "home_team":       h,
            "away_team":       a,
            "home_win_rate":   round(float(home_data.get("win_rate", 0.5)), 4),
            "away_win_rate":   round(1 - float(home_data.get("win_rate", 0.5)), 4),
            "games_played":    int(home_data.get("games", 0)),
        }
        cache.set(cache_key, response, timeout=3600)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500





# ── Best bets ─────────────────────────────────────────────────────────────────

@api_bp.route("/best-bets")
@rate_limit
def best_bets():
    """
    Return highest-confidence predictions where ML and Vegas most agree.
    Only meaningful when live odds are available.
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    cached = cache.get("best_bets")
    if cached:
        return jsonify(cached)

    try:
        profiles = pred_module._team_profiles
        from app.services.schedule import get_upcoming_games
        games    = get_upcoming_games(profiles)
        preds    = predict_bulk([{"home_team": g["home_team"], "away_team": g["away_team"]}
                                  for g in games])
        pred_map = {(p["home_team"], p["away_team"]): p for p in preds if "error" not in p}
        merged   = [{**g, **pred_map.get((g["home_team"], g["away_team"]), {})} for g in games]

        best = []
        for g in merged:
            hp   = g.get("home_win_probability", 0.5)
            conf = max(hp, 1 - hp)
            if conf < 0.62:
                continue  # skip toss-ups

            # Vegas agreement: if ML says home wins and home ML is negative (favored), they agree
            ml   = g.get("home_moneyline")
            edge = None
            if ml is not None:
                vegas_favors_home = ml < 0
                ml_favors_home    = hp >= 0.5
                if vegas_favors_home == ml_favors_home:
                    # Convert moneyline to implied prob for edge calc
                    if ml < 0:
                        vegas_prob = abs(ml) / (abs(ml) + 100)
                    else:
                        vegas_prob = 100 / (ml + 100)
                    edge = round(hp - vegas_prob, 4) if ml_favors_home else round((1-hp) - (1-vegas_prob), 4)

            best.append({**g, "confidence_score": round(conf, 4), "edge": edge})

        best.sort(key=lambda x: (-(x.get("edge") or 0), -x["confidence_score"]))
        response = {"best_bets": best[:8], "count": len(best)}
        cache.set("best_bets", response, timeout=CACHE_TTL_UPCOMING)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Roster adjustment explanations ────────────────────────────────────────────

@api_bp.route("/adjustments")
@rate_limit
def adjustments():
    """Return roster adjustment explanations for all teams or a specific team."""
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    team = request.args.get("team", "").upper()
    try:
        from app.services.roster_adjustment import get_adjustment_explanations
        data = get_adjustment_explanations(team if team else None)
        return jsonify({"adjustments": data, "team": team or "all"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── What-if simulator ─────────────────────────────────────────────────────────

@api_bp.route("/simulate", methods=["POST"])
@rate_limit
def simulate():
    """
    Simulate a prediction with custom overrides.

    Body fields (all optional except home_team/away_team):
      home_team, away_team          required
      home_rest_days (4-14)         default 7
      away_rest_days (4-14)         default 7
      home_qb_out    bool           starting QB out
      away_qb_out    bool
      home_wr1_out   bool           WR1 out
      away_wr1_out   bool
      home_rb1_out   bool           RB1 out
      away_rb1_out   bool
      home_ol_injury bool           OL injury (pass protection)
      away_ol_injury bool
      wind_mph       int  0-50       wind speed in mph
      temp_f         int  0-110      temperature °F
      precipitation  bool           rain or snow
      dome_game      bool           game in a dome (overrides weather)

    Returns base + simulated prediction + delta + active_factors list.
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True) or {}
    home = str(data.get("home_team", "")).upper().strip()
    away = str(data.get("away_team", "")).upper().strip()
    if not home or not away:
        return jsonify({"error": "home_team and away_team required"}), 400

    try:
        base   = predict_matchup(home, away)
        from copy import deepcopy
        h_prof_raw = get_team_profile(home)
        a_prof_raw = get_team_profile(away)
        # Ensure profiles are clean Python dicts with float values
        # (numpy types from training can cause subscript errors)
        def _clean_profile(raw) -> dict:
            if not isinstance(raw, dict):
                return {}
            return {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in raw.items()}
        h_prof = _clean_profile(deepcopy(h_prof_raw))
        a_prof = _clean_profile(deepcopy(a_prof_raw))

        active_factors = []   # human-readable list of what changed

        # ── Injury: team-specific win probability adjustments ────────────────
        # Base penalties scaled by each team's actual stats — Mahomes getting
        # hurt matters more than a replacement-level QB because KC is more
        # pass-EPA-dependent. Elite defenses also lose more when their star
        # players go down than already-struggling defenses.
        #
        # Base penalties (league average team):
        #   QB out:   13%  |  WR1: 4.5%  |  RB1: 2.5%  |  OL: 3.5%
        #   EDGE out: 5.0% |  CB1: 4.0%  |  LB:  2.5%  |  S:  2.0%
        #
        # Multipliers derived from team's rolling EPA stats:
        #   QB/WR1/OL: scale with pass EPA (pass-heavy = more dependent)
        #   RB1: scale with rush EPA
        #   EDGE/CB1: scale with defensive EPA allowed (elite D = more dependent)
        #   LB/S: flat (too position-agnostic to scale reliably)

        BASE_PENALTIES = {
            "qb":   0.13,
            "wr1":  0.045,
            "rb1":  0.025,
            "ol":   0.035,
            "cb1":  0.040,
            "edge": 0.050,
            "lb":   0.025,
            "s":    0.020,
        }

        INJURY_LABELS = {
            "qb":   "Starting QB out",
            "wr1":  "WR1 out",
            "rb1":  "RB1 out",
            "ol":   "OL injury",
            "cb1":  "CB1 out",
            "edge": "Pass rusher (EDGE) out",
            "lb":   "LB out",
            "s":    "Safety out",
        }

        INJURY_FIELDS = {
            "qb":   lambda t: f"{t}_qb_out",
            "wr1":  lambda t: f"{t}_wr1_out",
            "rb1":  lambda t: f"{t}_rb1_out",
            "ol":   lambda t: f"{t}_ol_injury",
            "cb1":  lambda t: f"{t}_cb1_out",
            "edge": lambda t: f"{t}_edge_out",
            "lb":   lambda t: f"{t}_lb_out",
            "s":    lambda t: f"{t}_s_out",
        }

        def _injury_multiplier(pos: str, prof) -> float:
            """
            Scale base penalty by team's actual stats.
            Returns a multiplier in [0.55, 1.55].
            If prof is not a dict (corrupted profile), return 1.0 (base penalty).
            """
            # Guard against corrupted profiles
            if not isinstance(prof, dict):
                return 1.0

            PASS_EPA_AVG  =  0.03   # league average rolling pass EPA
            RUSH_EPA_AVG  = -0.02   # league average rolling rush EPA
            DEF_EPA_AVG   =  0.00   # league average defensive EPA allowed

            if pos in ("qb", "wr1", "ol"):
                # Scale with pass EPA — pass-heavy teams more QB/WR-dependent
                stat = float(prof.get("roll_pass_epa", PASS_EPA_AVG) or PASS_EPA_AVG)
                dev  = stat - PASS_EPA_AVG
                return max(0.55, min(1.55, 1.0 + (dev / 0.10) * 0.45))

            elif pos == "rb1":
                # Scale with rush EPA — run-heavy teams more RB-dependent
                stat = float(prof.get("roll_rush_epa", RUSH_EPA_AVG) or RUSH_EPA_AVG)
                dev  = stat - RUSH_EPA_AVG
                return max(0.55, min(1.40, 1.0 + (dev / 0.08) * 0.35))

            elif pos in ("edge", "cb1"):
                # Scale with defensive EPA allowed — elite defenses more reliant on stars
                stat = float(prof.get("roll_defensive_epa_allowed", DEF_EPA_AVG) or DEF_EPA_AVG)
                dev  = -(stat - DEF_EPA_AVG)
                return max(0.60, min(1.45, 1.0 + (dev / 0.10) * 0.35))

            else:  # lb, s — flat
                return 1.0

        # Accumulate net probability delta from all injuries
        home_prob_delta = 0.0
        injury_details  = []   # for active_factors with actual percentages

        for team_key, label_prefix, prof in [
            ("home", home, h_prof),
            ("away", away, a_prof),
        ]:
            for pos, field_fn in INJURY_FIELDS.items():
                if not data.get(field_fn(team_key)):
                    continue
                base_penalty = BASE_PENALTIES[pos]
                mult         = _injury_multiplier(pos, prof)
                delta        = round(base_penalty * mult, 4)
                # Home injury → home wins less; away injury → home wins more
                home_prob_delta += delta if team_key == "away" else -delta
                if pos == "qb":
                    if team_key == "home":
                        h_prof["qb_out"] = 1.0
                    else:
                        a_prof["qb_out"] = 1.0
                pct_str = f"{delta*100:.1f}%"
                active_factors.append(
                    f"{label_prefix} {INJURY_LABELS[pos]} (−{pct_str})"
                )

        # ── Rest days ─────────────────────────────────────────────────────────
        # Short-week teams average ~1.5 pts worse (historical Thursday splits).
        # Extra rest gives a small but real boost (bye week effect).
        REST_PENALTY_PER_DAY = 0.008
        for prof, key, label_prefix in [
            (h_prof, "home_rest_days", home),
            (a_prof, "away_rest_days", away),
        ]:
            rest = float(data.get(key, 7))
            if rest < 7:
                days_short = 7 - rest
                penalty = days_short * REST_PENALTY_PER_DAY
                prof["roll_offensive_epa"] = prof.get("roll_offensive_epa", 0) - penalty
                prof["roll_points_scored"] = prof.get("roll_points_scored", 0) * (1 - penalty * 2)
                active_factors.append(f"{label_prefix} short rest ({int(rest)}d)")
            elif rest > 7:
                bonus = min((rest - 7) * 0.003, 0.015)
                prof["roll_offensive_epa"] = prof.get("roll_offensive_epa", 0) + bonus
                active_factors.append(f"{label_prefix} extra rest ({int(rest)}d)")

        # ── Weather effects ───────────────────────────────────────────────────
        # Research basis: Burke (2014), nflfastR weather studies.
        # Wind > 15mph: passing EPA drops ~0.04 per 10mph above 15.
        # Cold (< 32°F): passing EPA drops ~0.02, rushing relatively unaffected.
        # Precipitation: similar to ~10mph wind penalty (wet ball, footing).
        # Dome: neutralises all weather effects.
        dome = bool(data.get("dome_game", False))
        wind_mph = float(data.get("wind_mph", 0))
        temp_f   = float(data.get("temp_f", 65))
        precip   = bool(data.get("precipitation", False))

        if not dome and (wind_mph > 0 or temp_f < 65 or precip):
            weather_epa_penalty  = 0.0
            weather_pass_penalty = 0.0
            weather_pts_penalty  = 0.0
            weather_rush_bonus   = 0.0
            weather_notes        = []

            # Wind
            if wind_mph > 15:
                wind_penalty = min((wind_mph - 15) / 10 * 0.04, 0.12)
                weather_pass_penalty += wind_penalty
                weather_epa_penalty  += wind_penalty * 0.6
                weather_rush_bonus   += wind_penalty * 0.3   # teams run more in wind
                weather_notes.append(f"{int(wind_mph)}mph wind")
            elif wind_mph > 25:
                weather_pts_penalty += 0.05

            # Cold
            if temp_f < 32:
                cold_penalty = min((32 - temp_f) / 20 * 0.02, 0.06)
                weather_pass_penalty += cold_penalty
                weather_epa_penalty  += cold_penalty * 0.5
                weather_notes.append(f"{int(temp_f)}°F")
            elif temp_f > 90:
                # Heat: conditioning matters, home team slight edge
                h_prof["roll_offensive_epa"] = h_prof.get("roll_offensive_epa", 0) + 0.01
                weather_epa_penalty += 0.01   # both teams slightly impaired
                weather_notes.append(f"{int(temp_f)}°F heat")

            # Precipitation
            if precip:
                weather_pass_penalty += 0.04
                weather_epa_penalty  += 0.02
                weather_pts_penalty  += 0.03
                weather_notes.append("precipitation")

            # Apply symmetrically to both teams
            for prof in [h_prof, a_prof]:
                prof["roll_offensive_epa"] = prof.get("roll_offensive_epa", 0) - weather_epa_penalty
                prof["roll_pass_epa"]      = prof.get("roll_pass_epa", 0)      - weather_pass_penalty
                prof["roll_points_scored"] = prof.get("roll_points_scored", 0) * (1 - weather_pts_penalty)
                prof["roll_rush_epa"]      = prof.get("roll_rush_epa", 0)      + weather_rush_bonus

            if weather_notes:
                active_factors.append("Weather: " + ", ".join(weather_notes))
        elif dome:
            active_factors.append("Dome game (weather neutralised)")

        home_rest = float(data.get("home_rest_days", 7))
        away_rest = float(data.get("away_rest_days", 7))

        # Run base simulation with profile changes (rest/weather)
        import app.services.predictor as pred_module
        orig_h = pred_module._team_profiles.get(home)
        orig_a = pred_module._team_profiles.get(away)
        pred_module._team_profiles[home] = h_prof
        pred_module._team_profiles[away] = a_prof

        simulated = predict_matchup(home, away, home_rest, away_rest)

        pred_module._team_profiles[home] = orig_h
        pred_module._team_profiles[away] = orig_a

        # Apply direct injury probability deltas on top of simulated result
        sim_home_prob = simulated["home_win_probability"]
        adj_home_prob = round(max(0.02, min(0.98, sim_home_prob + home_prob_delta)), 4)
        adj_away_prob = round(1 - adj_home_prob, 4)
        if home_prob_delta != 0.0:
            simulated = {
                **simulated,
                "home_win_probability": adj_home_prob,
                "away_win_probability": adj_away_prob,
                "predicted_winner":     home if adj_home_prob >= 0.5 else away,
            }

        delta = round(simulated["home_win_probability"] -
                      base["home_win_probability"], 4)

        return jsonify({
            "base":           base,
            "simulated":      simulated,
            "delta":          delta,
            "active_factors": active_factors,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"/api/simulate error:\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500


# ── Elo history ───────────────────────────────────────────────────────────────

@api_bp.route("/elo/history")
@rate_limit
def elo_history():
    """
    Return Elo trajectory for one or all teams.
    ?team=KC  returns KC's history by week.
    No team param returns season-end snapshots for all teams.
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        from app.services.predictor import _elo_system
        if not _elo_system or not _elo_system.history:
            return jsonify({"error": "No Elo history available — retrain first"}), 503

        team = request.args.get("team", "").upper()
        history = _elo_system.history

        if team:
            # Per-game trajectory for one team
            points = []
            for h in history:
                if h["home_team"] == team:
                    points.append({
                        "season": h["season"],
                        "week":   h["week"],
                        "elo":    round(h["pre_home_elo"], 1),
                        "game_id": h.get("game_id", ""),
                    })
                elif h["away_team"] == team:
                    points.append({
                        "season": h["season"],
                        "week":   h["week"],
                        "elo":    round(h["pre_away_elo"], 1),
                        "game_id": h.get("game_id", ""),
                    })
            return jsonify({"team": team, "history": points, "current_elo": round(_elo_system.get(team), 1)})
        else:
            # Season-end snapshots for all teams grouped by season
            seasons = sorted(set(h["season"] for h in history))
            snap: dict = {}
            for season in seasons:
                season_games = [h for h in history if h["season"] == season]
                if not season_games:
                    continue
                max_week = max(h["week"] for h in season_games)
                last_games = [h for h in season_games if h["week"] == max_week]
                team_elos: dict = {}
                for h in last_games:
                    # Use post-game Elo (approximate from pre + delta)
                    team_elos[h["home_team"]] = round(h["pre_home_elo"], 1)
                    team_elos[h["away_team"]] = round(h["pre_away_elo"], 1)
                snap[str(season)] = team_elos
            return jsonify({"seasons": seasons, "snapshots": snap})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Sharp vs public indicator ─────────────────────────────────────────────────

@api_bp.route("/sharp/<home>/<away>")
@rate_limit
def sharp_indicator(home: str, away: str):
    """
    Compare ML probability vs Vegas implied probability.
    Returns edge, direction, and sharp rating (1-5 stars).
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    h = home.upper().strip()
    a = away.upper().strip()
    cache_key = f"sharp_{h}_{a}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)
    try:
        pred = predict_matchup(h, a)
        hp   = get_team_profile(h)
        ap   = get_team_profile(a)
        from app.services.odds import estimate_odds_from_profiles, fetch_fanduel_odds
        # Try live odds first
        live = fetch_fanduel_odds()
        odds = next((o for o in live if o["home_team"]==h and o["away_team"]==a), None)
        if not odds:
            odds = estimate_odds_from_profiles(hp, ap)
        odds_source = "fanduel" if odds.get("home_moneyline") else "estimated"

        ml_home = pred["home_win_probability"]
        home_ml = odds.get("home_moneyline")
        away_ml = odds.get("away_moneyline")

        if home_ml is None:
            return jsonify({"error": "No odds available for this matchup"}), 404

        # Convert moneyline to implied probability
        def ml_to_prob(ml):
            if ml < 0:
                return abs(ml) / (abs(ml) + 100)
            return 100 / (ml + 100)

        vegas_home = ml_to_prob(home_ml)
        vegas_away = ml_to_prob(away_ml)
        # Remove vig (overround)
        total = vegas_home + vegas_away
        vegas_home_fair = vegas_home / total
        vegas_away_fair = vegas_away / total

        edge_home = round(ml_home - vegas_home_fair, 4)
        edge_away = round((1 - ml_home) - vegas_away_fair, 4)

        # Sharp rating: 1-5 stars based on edge magnitude
        best_edge = max(abs(edge_home), abs(edge_away))
        stars = 1 if best_edge < 0.03 else 2 if best_edge < 0.06 else 3 if best_edge < 0.10 else 4 if best_edge < 0.15 else 5

        sharp_team  = h if edge_home > edge_away else a
        sharp_edge  = edge_home if edge_home > edge_away else edge_away
        public_side = a if sharp_team == h else h

        response = {
            "home_team":        h,
            "away_team":        a,
            "ml_home_prob":     round(ml_home, 4),
            "ml_away_prob":     round(1 - ml_home, 4),
            "vegas_home_prob":  round(vegas_home_fair, 4),
            "vegas_away_prob":  round(vegas_away_fair, 4),
            "edge_home":        edge_home,
            "edge_away":        edge_away,
            "sharp_team":       sharp_team,
            "sharp_edge":       round(sharp_edge, 4),
            "public_side":      public_side,
            "sharp_stars":      stars,
            "odds_source":      odds_source,
            "home_moneyline":   home_ml,
            "away_moneyline":   away_ml,
        }
        cache.set(cache_key, response, timeout=300)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Live scores (ESPN) ────────────────────────────────────────────────────────

@api_bp.route("/live-scores")
def live_scores():
    """
    Fetch live NFL scores from ESPN's public API.
    Returns current game states with live win probabilities.
    Note: ESPN's API is unofficial and may change without notice.
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    cached = cache.get("live_scores")
    if cached:
        return jsonify(cached)

    try:
        import requests as req
        resp = req.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            timeout=8
        )
        resp.raise_for_status()
        espn = resp.json()
    except Exception as e:
        return jsonify({"error": f"ESPN API unavailable: {e}"}), 503

    games = []
    _team_map = {
        "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR",
        "CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL","DEN":"DEN",
        "DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX",
        "KC":"KC","LV":"LV","LAC":"LAC","LAR":"LAR","LA":"LAR",
        "MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG",
        "NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SF":"SF","SEA":"SEA",
        "TB":"TB","TEN":"TEN","WSH":"WAS","WAS":"WAS",
    }

    for event in espn.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_abbr = _team_map.get(home_c.get("team",{}).get("abbreviation",""), "")
        away_abbr = _team_map.get(away_c.get("team",{}).get("abbreviation",""), "")

        home_score = int(home_c.get("score", 0) or 0)
        away_score = int(away_c.get("score", 0) or 0)

        status   = event.get("status", {})
        state    = status.get("type", {}).get("state", "pre")   # pre | in | post
        period   = status.get("period", 0)
        clock    = status.get("displayClock", "0:00")
        detail   = status.get("type", {}).get("shortDetail", "")

        # Live win probability: use our model adjusted by score + time remaining
        live_wp = _live_win_probability(
            home_abbr, away_abbr,
            home_score, away_score,
            period, clock, state
        )

        games.append({
            "home_team":     home_abbr,
            "away_team":     away_abbr,
            "home_score":    home_score,
            "away_score":    away_score,
            "state":         state,
            "period":        period,
            "clock":         clock,
            "detail":        detail,
            "home_win_prob": live_wp,
            "away_win_prob": round(1 - live_wp, 4),
        })

    response = {"games": games, "count": len(games)}
    # Cache for 30 seconds during live games, longer otherwise
    has_live = any(g["state"] == "in" for g in games)
    cache.set("live_scores", response, timeout=30 if has_live else 300)
    return jsonify(response)


def _live_win_probability(
    home: str, away: str,
    home_score: int, away_score: int,
    period: int, clock: str, state: str
) -> float:
    """
    Estimate live win probability combining:
      - Pre-game ML prediction (model strength signal)
      - Current score differential
      - Time remaining (minutes)
    Uses a logistic blend: as time runs out, score dominates.
    """
    if state == "post":
        return 1.0 if home_score > away_score else (0.0 if away_score > home_score else 0.5)

    if not home or not away:
        return 0.5

    # Pre-game probability from model
    try:
        pre_game = predict_matchup(home, away)
        pre_prob = pre_game["home_win_probability"]
    except Exception:
        pre_prob = 0.5

    if state == "pre":
        return pre_prob

    # Time remaining in minutes
    try:
        mins_parts = clock.split(":")
        clock_mins = int(mins_parts[0]) + int(mins_parts[1]) / 60
    except Exception:
        clock_mins = 0

    total_mins  = 60.0
    periods_remaining = max(0, 4 - period)
    mins_remaining = periods_remaining * 15 + clock_mins
    time_elapsed_frac = 1 - (mins_remaining / total_mins)

    # Score-based probability (simple logistic on score diff)
    score_diff     = home_score - away_score
    score_prob     = 1 / (1 + 10 ** (-score_diff / 7))  # 7 pts ~= 1 score unit

    # Blend: weight shifts from pre-game to score as game progresses
    blend = time_elapsed_frac ** 1.5   # accelerates toward end
    live_prob = pre_prob * (1 - blend) + score_prob * blend

    return round(max(0.02, min(0.98, live_prob)), 4)


# ── SOS (Strength of Schedule) ───────────────────────────────────────────────

@api_bp.route("/sos")
@rate_limit
def sos():
    """Return SOS rankings for all teams."""
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    cached = cache.get("sos_rankings")
    if cached:
        return jsonify(cached)
    try:
        import app.services.predictor as pred_module
        sos_data = pred_module._sos_data or {}
        resp = {"sos": sos_data, "count": len(sos_data)}
        cache.set("sos_rankings", resp, timeout=3600)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/sos/<team>")
@rate_limit
def sos_team(team: str):
    """Return SOS for a specific team."""
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        import app.services.predictor as pred_module
        t    = team.upper().strip()
        data = (pred_module._sos_data or {}).get(t)
        if not data:
            return jsonify({"error": f"No SOS data for {t}"}), 404
        return jsonify({"team": t, **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Team detail page ──────────────────────────────────────────────────────────

# Pre-built Elo history cache (built once on first request, reused after)
_elo_history_cache: dict = {}


def _build_elo_history_cache(elo_system) -> dict:
    """Build per-team Elo history once and cache it in memory."""
    global _elo_history_cache
    if _elo_history_cache:
        return _elo_history_cache
    result: dict = {}
    for h in elo_system.history:
        for team, elo_val in [
            (h["home_team"], h["pre_home_elo"]),
            (h["away_team"], h["pre_away_elo"]),
        ]:
            if team not in result:
                result[team] = []
            result[team].append({
                "season": h["season"],
                "week":   h["week"],
                "elo":    round(elo_val, 1),
            })
    _elo_history_cache = result
    return result


@api_bp.route("/team/<team_abbr>")
@rate_limit
def team_detail_full(team_abbr: str):
    """
    Full team detail: profile, Elo history, schedule predictions,
    division H2H, SOS. Heavily cached — first load builds index, subsequent
    loads are served from cache in <50ms.
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    t = team_abbr.upper().strip()
    cache_key = f"team_detail_{t}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)

    try:
        import app.services.predictor as pred_module
        from app.services.sos import are_division_rivals, get_division, NFL_DIVISIONS

        # Profile
        try:
            profile = get_team_profile(t)
        except ValueError:
            return jsonify({"error": f"Team {t} not found"}), 404

        # Elo — use pre-built cache
        elo_system  = pred_module._elo_system
        elo_rating  = round(elo_system.get(t), 1) if elo_system else None
        power_score = elo_system.to_display_score(t) if elo_system else 50

        elo_history = []
        if elo_system and elo_system.history:
            hist_cache  = _build_elo_history_cache(elo_system)
            elo_history = hist_cache.get(t, [])

        # Division rivals — predict_matchup is fast (in-memory), no I/O
        division   = get_division(t)
        div_rivals = []
        h2h_data   = pred_module._model_artifact.get("h2h", {})
        if division:
            for rival in NFL_DIVISIONS.get(division, []):
                if rival == t:
                    continue
                h2h      = h2h_data.get((t, rival), {})
                try:
                    pred = predict_matchup(t, rival)
                except Exception:
                    pred = None
                div_rivals.append({
                    "team":         rival,
                    "h2h_win_rate": round(float(h2h.get("win_rate", 0.5)), 3),
                    "h2h_games":    int(h2h.get("games", 0)),
                    "prediction":   pred,
                    "rival_elo":    round(elo_system.get(rival), 1) if elo_system else None,
                })

        # SOS
        sos_data = (pred_module._sos_data or {}).get(t, {})

        # Upcoming games — reuse the already-cached /api/upcoming response
        # so we don't re-hit Sleeper/ESPN/odds on every team page load
        cached_upcoming = cache.get("upcoming_all")
        if cached_upcoming:
            all_games = cached_upcoming.get("games", [])
        else:
            from app.services.schedule import get_upcoming_games
            all_games = get_upcoming_games(pred_module._team_profiles)

        game_preds = []
        for g in all_games:
            if g.get("home_team") != t and g.get("away_team") != t:
                continue
            is_home  = g["home_team"] == t
            opp      = g["away_team"] if is_home else g["home_team"]
            win_prob = g.get("home_win_probability", 0.5) if is_home else g.get("away_win_probability", 0.5)
            game_preds.append({
                "home_team":      g["home_team"],
                "away_team":      g["away_team"],
                "opponent":       opp,
                "is_home":        is_home,
                "win_prob":       round(float(win_prob), 4),
                "week":           g.get("week"),
                "season":         g.get("season"),
                "is_division":    are_division_rivals(t, opp),
                "mode":           g.get("mode"),
                "home_moneyline": g.get("home_moneyline"),
                "away_moneyline": g.get("away_moneyline"),
                "spread":         g.get("spread"),
            })

        # Recent prediction history from DB
        try:
            from app.database import get_team_prediction_history
            history = get_team_prediction_history(t, limit=10)
        except Exception:
            history = []

        response = {
            "team":               t,
            "division":           division,
            "profile":            {k: round(float(v), 4) if isinstance(v, float) else v
                                   for k, v in profile.items()},
            "elo":                elo_rating,
            "power_score":        power_score,
            "elo_history":        elo_history,
            "sos":                sos_data,
            "division_rivals":    div_rivals,
            "upcoming_games":     game_preds,
            "prediction_history": history,
        }

        cache.set(cache_key, response, timeout=300)
        return jsonify(response)
    except Exception as e:
        import traceback
        logger.error(f"/api/team/{t} error: " + traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── Bye week teams ────────────────────────────────────────────────────────────

@api_bp.route("/bye-teams")
@rate_limit
def bye_teams():
    """
    Returns teams on bye this week.
    During the season: teams with no game in the current/next week vs all 32 teams.
    During the off-season: returns empty (no byes off-season).
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        from app.services.download_pbp import is_offseason
        if is_offseason():
            return jsonify({"bye_teams": [], "week": None, "season": None})

        import app.services.predictor as pred_module
        from app.services.schedule import get_upcoming_games

        games   = get_upcoming_games(pred_module._team_profiles)
        active  = set()
        for g in games:
            active.add(g["home_team"])
            active.add(g["away_team"])

        all_teams = set(get_all_teams())
        bye       = sorted(all_teams - active)

        week   = games[0].get("week") if games else None
        season = games[0].get("season") if games else None

        return jsonify({
            "bye_teams": bye,
            "week":      week,
            "season":    season,
            "count":     len(bye),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Division game flag on upcoming ───────────────────────────────────────────

@api_bp.route("/division-games")
@rate_limit
def division_games():
    """Return which upcoming games are division matchups."""
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        from app.services.sos import are_division_rivals
        import app.services.predictor as pred_module
        from app.services.schedule import get_upcoming_games
        games = get_upcoming_games(pred_module._team_profiles)
        result = [
            {**g, "is_division_game": are_division_rivals(g["home_team"], g["away_team"])}
            for g in games
        ]
        return jsonify({"games": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Email subscription ────────────────────────────────────────────────────────

@api_bp.route("/subscribe", methods=["POST"])
def subscribe():
    data  = request.get_json(force=True) or {}
    email = str(data.get("email", "")).strip().lower()
    team  = str(data.get("team", "")).upper().strip() or None

    if not email or "@" not in email or len(email) > 200:
        return jsonify({"error": "Valid email required"}), 400

    try:
        from app.database import add_subscriber
        is_new = add_subscriber(email, team)
        return jsonify({
            "success":  True,
            "is_new":   is_new,
            "message":  "Subscribed! You'll get weekly best bets every Tuesday." if is_new
                        else "Already subscribed.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/unsubscribe", methods=["POST"])
def api_unsubscribe():
    data  = request.get_json(force=True) or {}
    email = str(data.get("email", "")).strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400
    from app.database import remove_subscriber
    remove_subscriber(email)
    return jsonify({"success": True})


# ── Accuracy (now from SQLite) ────────────────────────────────────────────────

@api_bp.route("/accuracy")
@rate_limit
def accuracy():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503
    try:
        from app.database import get_accuracy_stats
        stats = get_accuracy_stats()
        # Also include CV stats from model
        meta = get_model_meta()
        stats["cv_accuracy"] = meta.get("cv_accuracy", 0)
        stats["cv_auc"]      = meta.get("cv_auc", 0)
        stats["train_brier"] = meta.get("train_brier", 0)
        stats["feature_importances"] = meta.get("feature_importances", {})
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Push notifications ───────────────────────────────────────────────────────

@api_bp.route("/push/subscribe", methods=["POST"])
def push_subscribe():
    """Store a browser push subscription."""
    data = request.get_json(force=True) or {}
    endpoint = data.get("endpoint", "")
    p256dh   = data.get("keys", {}).get("p256dh", "")
    auth     = data.get("keys", {}).get("auth", "")
    team     = str(data.get("team", "")).upper().strip() or None

    if not endpoint or not p256dh or not auth:
        return jsonify({"error": "Invalid subscription object"}), 400

    try:
        from app.database import add_push_subscription
        add_push_subscription(endpoint, p256dh, auth, team)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/push/send-test", methods=["POST"])
@rate_limit
def push_test():
    """Send a test push notification (for development/testing)."""
    data    = request.get_json(force=True) or {}
    team    = str(data.get("team", "")).upper().strip() or None
    message = data.get("message", "NFL Oracle test notification 🏈")

    try:
        from app.database import get_push_subscriptions
        subs = get_push_subscriptions(team)
        if not subs:
            return jsonify({"error": "No subscriptions found"}), 404

        sent = _send_push_notifications(subs, {
            "title": "NFL Oracle 🏈",
            "body":  message,
            "url":   "/",
        })
        return jsonify({"sent": sent, "total": len(subs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _send_push_notifications(subscriptions: list, payload: dict) -> int:
    """Send push notifications. Requires pywebpush (optional)."""
    import json
    sent = 0
    try:
        from pywebpush import webpush, WebPushException
        vapid_private = os.environ.get("VAPID_PRIVATE_KEY", "")
        vapid_email   = os.environ.get("VAPID_EMAIL", "mailto:admin@example.com")
        if not vapid_private:
            logger.warning("VAPID_PRIVATE_KEY not set — push not sent")
            return 0
        for sub in subscriptions:
            try:
                webpush(
                    subscription_info={
                        "endpoint": sub["endpoint"],
                        "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                    },
                    data=json.dumps(payload),
                    vapid_private_key=vapid_private,
                    vapid_claims={"sub": vapid_email},
                )
                sent += 1
            except WebPushException:
                pass
    except ImportError:
        logger.warning("pywebpush not installed — install with: pip install pywebpush")
    return sent


# ── Update profile ────────────────────────────────────────────────────────────

@api_bp.route("/update-profile", methods=["POST"])
@rate_limit
def update_profile():
    if not is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    data     = request.get_json(force=True) or {}
    team     = str(data.get("team", "")).upper().strip()
    overrides = data.get("stats", {})

    if not team or not overrides:
        return jsonify({"error": "team and stats are required"}), 400

    valid_keys = {
        "roll_offensive_epa", "roll_defensive_epa_allowed",
        "roll_pass_epa", "roll_rush_epa",
        "roll_points_scored", "roll_points_allowed",
        "roll_turnovers", "roll_won", "injury_impact", "qb_out",
    }
    bad = set(overrides) - valid_keys
    if bad:
        return jsonify({"error": f"Unknown keys: {bad}. Valid: {valid_keys}"}), 400

    try:
        update_team_profile(team, overrides)
        # Invalidate prediction cache for this team
        cache.delete(f"upcoming_all")
        cache.delete(f"upcoming_{team}")
        return jsonify({"success": True, "team": team, "updated": overrides})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404