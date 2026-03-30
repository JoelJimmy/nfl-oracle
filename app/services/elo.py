"""
elo.py
Elo rating system for NFL teams.

Elo provides a single number representing team strength that:
  - Updates after every game based on margin of victory
  - Accounts for home field advantage
  - Mean-reverts between seasons (teams regress toward average)
  - Gives recent games more weight naturally through sequential updating

Based on FiveThirtyEight's NFL Elo methodology.
"""

import math
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from app.config import (
    ELO_K_FACTOR, ELO_HOME_ADVANTAGE, ELO_INITIAL,
    ELO_MEAN_REVERT, ELO_PATH
)
from app.logger import get_logger

logger = get_logger(__name__)


class EloSystem:
    """
    Maintains Elo ratings for all NFL teams.
    Ratings are updated sequentially through games in chronological order.
    """

    def __init__(self):
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []   # {game_id, season, week, home, away, pre_home, pre_away, result}

    def get(self, team: str) -> float:
        return self.ratings.get(team, ELO_INITIAL)

    def _expected(self, elo_a: float, elo_b: float) -> float:
        """Expected win probability for team A given ratings."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def _mov_multiplier(self, margin: float, elo_diff: float) -> float:
        """
        Margin of victory multiplier (FiveThirtyEight method).
        Larger wins update Elo more, but with diminishing returns.
        Autocorrelation correction prevents inflating Elo for large favorites.
        """
        abs_margin = abs(margin)
        mov = math.log(abs_margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
        return mov

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        game_id: str = "",
        season: int = 0,
        week: int = 0,
    ):
        """Update Elo ratings after a completed game."""
        home_elo = self.get(home_team)
        away_elo = self.get(away_team)

        # Home team gets advantage
        home_elo_adj = home_elo + ELO_HOME_ADVANTAGE
        expected_home = self._expected(home_elo_adj, away_elo)

        margin = home_score - away_score
        if margin > 0:
            result_home = 1.0
        elif margin < 0:
            result_home = 0.0
        else:
            result_home = 0.5

        elo_diff = home_elo_adj - away_elo
        mov_mult = self._mov_multiplier(margin, elo_diff * (1 if margin > 0 else -1))

        delta = ELO_K_FACTOR * mov_mult * (result_home - expected_home)

        self.ratings[home_team] = home_elo + delta
        self.ratings[away_team] = away_elo - delta

        self.history.append({
            "game_id":    game_id,
            "season":     season,
            "week":       week,
            "home_team":  home_team,
            "away_team":  away_team,
            "pre_home_elo": home_elo,
            "pre_away_elo": away_elo,
            "home_win_prob": expected_home,
            "result_home": result_home,
        })

    def season_revert(self):
        """
        Between seasons, regress all ratings toward 1500.
        Prevents a team's historical dominance from compounding forever.
        """
        for team in list(self.ratings.keys()):
            self.ratings[team] = (
                self.ratings[team] * (1 - ELO_MEAN_REVERT) +
                ELO_INITIAL * ELO_MEAN_REVERT
            )
        logger.info(f"Elo season revert applied. Range: "
                    f"{min(self.ratings.values()):.0f} – {max(self.ratings.values()):.0f}")

    def win_probability(self, home_team: str, away_team: str) -> float:
        """Return ML-ready home win probability from current Elo ratings."""
        home_elo = self.get(home_team) + ELO_HOME_ADVANTAGE
        away_elo = self.get(away_team)
        return self._expected(home_elo, away_elo)

    def get_all_ratings(self) -> dict[str, float]:
        return dict(self.ratings)

    def to_display_score(self, team: str) -> int:
        """Convert Elo rating to a 0-100 power ranking display score."""
        elo = self.get(team)
        lo, hi = 1300, 1700
        return max(0, min(100, int((elo - lo) / (hi - lo) * 100)))

    def save(self, path: str = ELO_PATH):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Elo ratings saved → {path}")

    @classmethod
    def load(cls, path: str = ELO_PATH) -> "EloSystem":
        elo = joblib.load(path)
        logger.info(f"Elo ratings loaded from {path}")
        return elo


def build_elo_from_games(games_df: pd.DataFrame) -> EloSystem:
    """
    Build Elo ratings by replaying all historical games chronologically.
    games_df must have: season, week, home_team, away_team, home_score, away_score, game_id
    """
    elo = EloSystem()
    games = games_df.sort_values(["season", "week"]).copy()

    prev_season = None
    for _, row in games.iterrows():
        season = int(row["season"])

        # Revert between seasons
        if prev_season is not None and season != prev_season:
            elo.season_revert()
        prev_season = season

        elo.update(
            home_team  = str(row["home_team"]),
            away_team  = str(row["away_team"]),
            home_score = float(row["home_score"]),
            away_score = float(row["away_score"]),
            game_id    = str(row.get("game_id", "")),
            season     = season,
            week       = int(row.get("week", 0)),
        )

    logger.info(f"Elo built from {len(games)} games. "
                f"Top 5: {sorted(elo.ratings.items(), key=lambda x:-x[1])[:5]}")
    return elo
