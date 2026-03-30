"""
email_alerts.py
Weekly email digest of best bets using Resend (free tier: 3000 emails/month).

Setup:
  1. Sign up at https://resend.com
  2. Add RESEND_API_KEY to your .env file
  3. Verify your sending domain (or use their sandbox domain for testing)

The weekly digest is sent every Tuesday at 7am (after the model retrains at 6am).
"""

import os
from app.logger import get_logger

logger = get_logger(__name__)

RESEND_API_KEY  = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL      = os.environ.get("EMAIL_FROM", "NFL Oracle <noreply@yourdomain.com>")
SITE_URL        = os.environ.get("SITE_URL", "http://localhost:5000")


def send_weekly_digest():
    """
    Build and send the weekly best bets digest to all active subscribers.
    Called by the scheduler every Tuesday at 7am.
    """
    if not RESEND_API_KEY:
        logger.warning("[email] No RESEND_API_KEY set — skipping email digest")
        return

    try:
        from app.database import get_active_subscribers
        from app.services.schedule import get_upcoming_games
        from app.services.predictor import _team_profiles, predict_bulk, is_loaded

        if not is_loaded():
            logger.warning("[email] Model not loaded — skipping digest")
            return

        subscribers = get_active_subscribers()
        if not subscribers:
            logger.info("[email] No active subscribers — skipping digest")
            return

        # Get best bets
        games   = get_upcoming_games(_team_profiles)
        preds   = predict_bulk([{"home_team": g["home_team"], "away_team": g["away_team"]}
                                  for g in games])
        pred_map = {(p["home_team"], p["away_team"]): p for p in preds if "error" not in p}
        merged  = [{**g, **pred_map.get((g["home_team"], g["away_team"]), {})} for g in games]

        best_bets = sorted(
            [g for g in merged if max(
                g.get("home_win_probability", 0.5),
                g.get("away_win_probability", 0.5)
            ) >= 0.62],
            key=lambda g: -max(
                g.get("home_win_probability", 0.5),
                g.get("away_win_probability", 0.5)
            )
        )[:5]

        if not best_bets:
            logger.info("[email] No strong picks this week — skipping digest")
            return

        season = best_bets[0].get("season", "")
        week   = best_bets[0].get("week", "")

        sent = 0
        for sub in subscribers:
            # Team filter: only send if subscriber cares about this week's games
            if sub.get("team_filter"):
                relevant = [g for g in best_bets
                            if g["home_team"] == sub["team_filter"] or
                               g["away_team"] == sub["team_filter"]]
                if not relevant:
                    continue
                games_to_send = relevant
            else:
                games_to_send = best_bets

            html = _build_email_html(games_to_send, season, week, sub["email"])
            text = _build_email_text(games_to_send, season, week)

            success = _send_email(
                to=sub["email"],
                subject=f"NFL Oracle — Week {week} Best Bets 🏈",
                html=html,
                text=text,
            )
            if success:
                sent += 1

        logger.info(f"[email] Weekly digest sent to {sent}/{len(subscribers)} subscribers")

    except Exception as e:
        import traceback
        logger.error(f"[email] Digest failed: {traceback.format_exc()}")


def send_game_alert(email: str, home_team: str, away_team: str,
                    home_prob: float, week: int, season: int):
    """Send a single-game alert to a subscriber."""
    if not RESEND_API_KEY:
        return False

    predicted_winner = home_team if home_prob >= 0.5 else away_team
    conf = max(home_prob, 1 - home_prob)
    conf_str = f"{conf*100:.1f}%"

    html = f"""
<div style="font-family:sans-serif;max-width:480px;margin:0 auto;background:#080b10;color:#e8eef8;padding:24px;border-radius:12px">
  <div style="font-size:1.4rem;font-weight:700;color:#00e5ff;margin-bottom:8px">🏈 NFL Oracle Game Alert</div>
  <div style="color:#7a8aa0;font-size:0.85rem;margin-bottom:16px">Season {season} · Week {week}</div>
  <div style="font-size:1.1rem;font-weight:700;margin-bottom:8px">{home_team} vs {away_team}</div>
  <div style="background:#0e1420;border-radius:8px;padding:16px;margin-bottom:16px">
    <div style="color:#7a8aa0;font-size:0.75rem;margin-bottom:4px">Model Prediction</div>
    <div style="font-size:1.3rem;font-weight:700;color:#00e5ff">{predicted_winner} wins</div>
    <div style="color:#7fff6e;font-size:0.85rem">{conf_str} confidence</div>
  </div>
  <a href="{SITE_URL}" style="background:#00e5ff;color:#000;padding:10px 20px;border-radius:8px;text-decoration:none;font-weight:700;font-size:0.85rem">View Full Prediction →</a>
  <div style="margin-top:20px;font-size:0.7rem;color:#7a8aa0">
    <a href="{SITE_URL}/unsubscribe?email={email}" style="color:#7a8aa0">Unsubscribe</a>
  </div>
</div>"""

    return _send_email(
        to=email,
        subject=f"🏈 {home_team} vs {away_team} — Week {week} Prediction",
        html=html,
        text=f"NFL Oracle: {predicted_winner} predicted to win vs {away_team if predicted_winner==home_team else home_team} ({conf_str}). View at {SITE_URL}",
    )


def _send_email(to: str, subject: str, html: str, text: str) -> bool:
    """Send an email via Resend API."""
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from":    FROM_EMAIL,
            "to":      [to],
            "subject": subject,
            "html":    html,
            "text":    text,
        })
        return True
    except ImportError:
        # Resend not installed — use requests directly
        import requests
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "from":    FROM_EMAIL,
                "to":      [to],
                "subject": subject,
                "html":    html,
                "text":    text,
            },
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.error(f"[email] Send failed to {to}: {e}")
        return False


def _build_email_html(bets: list, season, week, email: str) -> str:
    rows = ""
    for g in bets:
        hp        = g.get("home_win_probability", 0.5)
        winner    = g["home_team"] if hp >= 0.5 else g["away_team"]
        conf      = max(hp, 1 - hp)
        conf_pct  = f"{conf*100:.1f}%"
        ml_home   = g.get("home_moneyline")
        ml_str    = f"+{ml_home}" if ml_home and ml_home > 0 else (str(ml_home) if ml_home else "—")
        rows += f"""
<tr>
  <td style="padding:12px 8px;border-bottom:1px solid #1d2640;font-weight:700">{g['home_team']} vs {g['away_team']}</td>
  <td style="padding:12px 8px;border-bottom:1px solid #1d2640;color:#00e5ff;font-weight:700">{winner}</td>
  <td style="padding:12px 8px;border-bottom:1px solid #1d2640;color:#7fff6e">{conf_pct}</td>
  <td style="padding:12px 8px;border-bottom:1px solid #1d2640;color:#7a8aa0">{ml_str}</td>
</tr>"""

    return f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:#080b10">
<div style="font-family:'DM Sans',Arial,sans-serif;max-width:560px;margin:0 auto;background:#080b10;color:#e8eef8;padding:32px 24px">
  <div style="font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:#00e5ff;margin-bottom:8px">NFL Oracle</div>
  <div style="font-size:1.6rem;font-weight:700;margin-bottom:4px">Week {week} Best Bets</div>
  <div style="color:#7a8aa0;font-size:0.85rem;margin-bottom:24px">{season} NFL Season · Model picks sorted by confidence</div>

  <table style="width:100%;border-collapse:collapse;background:#0e1420;border-radius:10px;overflow:hidden">
    <thead>
      <tr style="background:#161d2e">
        <th style="padding:10px 8px;text-align:left;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;color:#7a8aa0">Matchup</th>
        <th style="padding:10px 8px;text-align:left;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;color:#7a8aa0">Pick</th>
        <th style="padding:10px 8px;text-align:left;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;color:#7a8aa0">Conf</th>
        <th style="padding:10px 8px;text-align:left;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;color:#7a8aa0">ML</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <div style="margin-top:24px;text-align:center">
    <a href="{SITE_URL}" style="background:#00e5ff;color:#000;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:700">View All Predictions →</a>
  </div>

  <div style="margin-top:32px;padding-top:16px;border-top:1px solid #1d2640;font-size:0.7rem;color:#7a8aa0;text-align:center">
    NFL Oracle · Predictions powered by XGBoost + Elo ratings<br>
    <a href="{SITE_URL}/unsubscribe?email={email}" style="color:#7a8aa0">Unsubscribe</a>
  </div>
</div>
</body></html>"""


def _build_email_text(bets: list, season, week) -> str:
    lines = [f"NFL Oracle — Week {week} Best Bets ({season} Season)\n"]
    for g in bets:
        hp     = g.get("home_win_probability", 0.5)
        winner = g["home_team"] if hp >= 0.5 else g["away_team"]
        conf   = f"{max(hp,1-hp)*100:.1f}%"
        lines.append(f"  {g['home_team']} vs {g['away_team']}: {winner} ({conf})")
    lines.append(f"\nView full predictions: {SITE_URL}")
    return "\n".join(lines)