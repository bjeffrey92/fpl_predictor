from abc import ABC, abstractmethod
from functools import reduce

import joblib
import polars as pl
import s3fs

from fpl_predictor.model_training.xgboost import XGBoostPredictor
from fpl_predictor.player_stats import (
    get_fixtures,
    get_player_data,
    get_player_gameweek_stats,
)


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


def _process_home_away_teams(gw_fixture_stats: dict[str, object]) -> pl.DataFrame:
    gw_fixture_stats_df = pl.DataFrame(
        {k: v for k, v in gw_fixture_stats.items() if k.startswith("team_")}
    )
    home_team = gw_fixture_stats_df.select(
        [i for i in gw_fixture_stats_df.columns if i.startswith("team_h")]
    )
    away_team = gw_fixture_stats_df.select(
        [i for i in gw_fixture_stats_df.columns if i.startswith("team_a")]
    )
    return pl.DataFrame(
        {
            "team_id": [home_team["team_h"].item(), away_team["team_a"].item()],
            "team_score": [
                home_team["team_h_score"].item(),
                away_team["team_a_score"].item(),
            ],
            "team_difficulty": [
                home_team["team_h_difficulty"].item(),
                away_team["team_a_difficulty"].item(),
            ],
            "home_team": [True, False],
            "opposition_team_score": [
                away_team["team_a_score"].item(),
                home_team["team_h_score"].item(),
            ],
            "opposition_team_difficulty": [
                away_team["team_a_difficulty"].item(),
                home_team["team_h_difficulty"].item(),
            ],
        }
    )


class XGBoost(_BasePrediction):
    _bucket = "fpl-prediction-models"
    _key_pattern = "xgboost/xgboost_{}_prediction_week.joblib"
    _player_stats_cols = (
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "yellow_cards",
        "saves",
        "bonus",
        "influence",
        "creativity",
        "threat",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "gameweek_points",
        "player_id",
    )

    def __init__(self, n_prediction_weeks: int, gameweek: int) -> None:
        if gameweek - n_prediction_weeks < 1:
            raise ValueError("Not enough data to predict gameweek scores")
        self.n_prediction_weeks = n_prediction_weeks
        self.gameweek = gameweek
        self.data = self._load_data()
        self.model = self._load_model()

    def _load_data(self) -> pl.DataFrame:
        gw_player_stats = {
            f"gw_-{i}": get_player_gameweek_stats(
                self.gameweek - i, self._player_stats_cols
            )
            for i in range(1, self.n_prediction_weeks + 1)
        }
        player_data = get_player_data().select("player_id", "team_id")
        gw_player_stats = {
            k: v.join(player_data, on="player_id") for k, v in gw_player_stats.items()
        }
        gw_fixture_stats = self._load_fixture_data()
        all_stats = {
            k: v.join(gw_fixture_stats[k], on="team_id").drop("team_id")
            for k, v in gw_player_stats.items()
        }
        all_stats_with_gw = [
            v.rename({i: f"{k}_{i}" for i in v.columns if i != "player_id"})
            for k, v in all_stats.items()
        ]
        if len(all_stats_with_gw) < 1:
            all_stats_df = reduce(
                lambda x, y: x.join(y, on="player_id"), all_stats.values()
            )
        else:
            all_stats_df = all_stats_with_gw[0]
        return all_stats_df

    def _load_fixture_data(self) -> dict[str, pl.DataFrame]:
        all_fixtures = get_fixtures()
        relevant_fixtures = {
            f"gw_-{i}": [x for x in all_fixtures if x["event"] == self.gameweek - i]
            for i in range(1, self.n_prediction_weeks + 1)
        }
        if not all([[i["finished"] for i in v] for v in relevant_fixtures.values()]):
            raise ValueError("Not all relevant fixtures have finished")
        return {
            k: pl.concat([_process_home_away_teams(i) for i in v])
            for k, v in relevant_fixtures.items()
        }

    def _load_model(self) -> XGBoostPredictor:
        fs = s3fs.S3FileSystem()
        key = self._key_pattern.format(self.n_prediction_weeks)
        filename = f"s3://{self._bucket}/{key}"
        with fs.open(filename, encoding="utf8") as fh:
            model = joblib.load(fh)
        return model

    def predict_gw_scores(self) -> pl.DataFrame:
        if not "player_id" in self.data.columns:
            raise ValueError("Data must contain player_id column")
        data = self.data.select(self.model.prediction_columns)
        preds = self.model.model.predict(data)
        return pl.DataFrame(
            {"player_id": self.data["player_id"], "gameweek_points": preds}
        )


SCORE_PREDICTOR_FACTORY = {"median_past_score": MedianPastScore, "xgboost": XGBoost}
