from abc import ABC, abstractmethod

import polars as pl

from fpl_predictor.player_stats import get_player_gameweek_stats


class _BasePrediction(ABC):
    @abstractmethod
    def predict_gw_scores(self) -> pl.DataFrame:
        pass


class MedianPastScore(_BasePrediction):
    def __init__(
        self,
        upcoming_gameweek: int,
        n_previous_weeks: int = 5,
        min_required_weeks: int = 3,
    ) -> None:
        """
        Predicts player gameweek points as the median gameweek points recorded in past
        weeks.

        Args:
            upcoming_gameweek (int): Which gameweek to predict?
            n_previous_weeks (int): How many previous gameweeks to consider?
            min_required_weeks (int): How many weeks must a player have participated in
                to be considered?
        """
        self.upcoming_gameweek = upcoming_gameweek
        self.n_previous_weeks = n_previous_weeks
        self.min_required_weeks = min_required_weeks

        self.data = self._get_data()

    def _get_data(self) -> pl.DataFrame:
        df = pl.concat(
            [
                get_player_gameweek_stats(
                    i, cols=["player_id", "gameweek", "gameweek_points"]
                ).drop("gameweek")
                for i in range(
                    self.upcoming_gameweek - self.n_previous_weeks,
                    self.upcoming_gameweek,
                )
            ]
        )
        players_to_consider = (
            df["player_id"]
            .value_counts()
            .filter(pl.col("count") > self.min_required_weeks)["player_id"]
        )
        return df.filter(pl.col("player_id").is_in(players_to_consider))

    def predict_gw_scores(self) -> pl.DataFrame:
        return self.data.group_by("player_id").median()


SCORE_PREDICTOR_FACTORY = {"median_past_score": MedianPastScore}
