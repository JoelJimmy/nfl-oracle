"""
download_pbp.py
Downloads play-by-play and schedule data via nfl_data_py.
Season window auto-updates every year.
"""

import warnings
import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
from app.config import TRAINING_SEASONS, SEASON_START_MONTH, OFFSEASON_MONTHS
from app.logger import get_logger

logger = get_logger(__name__)


def get_relevant_seasons() -> list[int]:
    """Return the N most relevant NFL seasons based on today's date."""
    today = datetime.today()
    current = today.year if today.month >= SEASON_START_MONTH else today.year - 1
    return [current - i for i in range(TRAINING_SEASONS - 1, -1, -1)]


def is_offseason() -> bool:
    return datetime.today().month in OFFSEASON_MONTHS


def get_next_season() -> int:
    today = datetime.today()
    return today.year if today.month in OFFSEASON_MONTHS else today.year + 1


def download_pbp_data() -> pd.DataFrame:
    """Download and filter play-by-play data for all relevant seasons."""
    seasons = get_relevant_seasons()
    logger.info(f"Downloading PBP for seasons: {seasons}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pbp = nfl.import_pbp_data(seasons)

    logger.info(f"Raw PBP rows: {len(pbp):,}")

    pbp = pbp[pbp["season_type"] == "REG"]
    pbp = pbp[pbp["total_home_score"].notna() & pbp["total_away_score"].notna()]

    logger.info(f"Filtered PBP rows: {len(pbp):,}")
    return pbp


def download_schedule_data() -> pd.DataFrame:
    """Download schedule/game-level data including scores and game info."""
    seasons = get_relevant_seasons()
    logger.info(f"Downloading schedules for: {seasons}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nfl.import_schedules(seasons)


def download_weekly_data() -> pd.DataFrame:
    """Download weekly player stats — used for injury impact scoring."""
    seasons = get_relevant_seasons()
    logger.info(f"Downloading weekly player data for: {seasons}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nfl.import_weekly_data(seasons)


if __name__ == "__main__":
    df = download_pbp_data()
    print(df[["season", "week", "game_id"]].drop_duplicates().tail(10))
