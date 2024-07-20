import json
from functools import cache, lru_cache, reduce
from typing import Literal, NamedTuple

import boto3
import polars as pl


class TrainTestValData(NamedTuple):
    train_X: pl.DataFrame
    train_y: pl.Series
    test_X: pl.DataFrame
    test_y: pl.Series
    val_X: pl.DataFrame
    val_y: pl.Series


@cache
def _s3_client() -> boto3.client:
    return boto3.client("s3")


def _load_data(key: str) -> str:
    s3 = _s3_client()
    data = s3.get_object(Bucket="fpl-data", Key=key)
    return data["Body"].read().decode("utf-8")


def player_gameweek_stats(columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_csv("s3://fpl-data/player_gameweek_stats_23-24.csv", columns=columns)


def _fixtures() -> list[dict[str, object]]:
    return json.loads(_load_data("fixtures_23-24.json"))


def _get_gw_player_stats(gw_id: int, gw_stats: pl.DataFrame) -> pl.DataFrame:
    selection_cols = [
        "player_id",
        "team_id",
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
    ]
    gw_stats = gw_stats.filter(pl.col("gameweek") == gw_id).select(selection_cols)
    return gw_stats.rename({i: f"gw_{gw_id}_{i}" for i in selection_cols[2:]})


def _process_home_away_teams(
    gw_fixture_stats_df: pl.DataFrame, home_away: Literal["team_h", "team_a"]
) -> pl.DataFrame:
    prefixes = ["team_a", "team_h"]
    if home_away not in prefixes:
        raise ValueError("home_away must be one of 'team_h' or 'team_a'")

    teams = gw_fixture_stats_df.select(
        [i for i in gw_fixture_stats_df.columns if i.startswith(home_away)]
    )
    teams = teams.with_columns(
        pl.Series("home_team", [bool(prefixes.index(home_away))])
    )
    return teams.rename({i: i.replace(home_away[-2:], "") for i in teams.columns})


def _get_gw_fixture_stats(gw_id: int, fixtures: list[dict[str, object]]):
    gw_fixtures = filter(lambda x: x["event"] == gw_id, fixtures)
    gw_fixture_stats = [
        {
            k: f[k]
            for k in [
                "team_a",
                "team_a_score",
                "team_a_difficulty",
                "team_h",
                "team_h_score",
                "team_h_difficulty",
            ]
        }
        for f in gw_fixtures
    ]
    gw_fixture_stats_df = pl.concat([pl.DataFrame(f) for f in gw_fixture_stats])

    home_teams = _process_home_away_teams(gw_fixture_stats_df, "team_h")
    away_teams = _process_home_away_teams(gw_fixture_stats_df, "team_a")
    df = pl.concat((home_teams, away_teams))

    # concat in opposite order to get opposition data
    opposition_data = pl.concat((away_teams, home_teams)).select(
        ["team_score", "team_difficulty"]
    )
    opposition_data = opposition_data.rename(
        {i: f"opposition_{i}" for i in opposition_data.columns}
    )
    df = pl.concat((df, opposition_data), how="horizontal")

    return df.rename({i: f"gw_{gw_id}_{i}" for i in df.columns if i != "team"})


def _train_test_val_split(data: pl.DataFrame, test_frac: float, val_frac: float):
    if test_frac + val_frac >= 1:
        raise ValueError("test_frac + val_frac must be less than 1")

    n = len(data)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)

    test_data = data.sample(n=test_n, seed=42, with_replacement=False)
    data = data.join(
        test_data, on=["player_id", "team_id", "prediction_gw"], how="anti"
    )
    val_data = data.sample(n=val_n, seed=42, with_replacement=False)
    train_data = data.join(
        val_data, on=["player_id", "team_id", "prediction_gw"], how="anti"
    )

    return TrainTestValData(
        train_X=train_data.drop(
            "gameweek_points", "player_id", "team_id", "prediction_gw"
        ),
        train_y=train_data["gameweek_points"],
        test_X=test_data.drop(
            "gameweek_points", "player_id", "team_id", "prediction_gw"
        ),
        test_y=test_data["gameweek_points"],
        val_X=val_data.drop("gameweek_points", "player_id", "team_id", "prediction_gw"),
        val_y=val_data["gameweek_points"],
    )


def load_data(
    n_prediction_weeks: int, test_frac: float = 0.2, val_frac: float = 0.2
) -> TrainTestValData:
    gw_stats = player_gameweek_stats()
    fixtures = _fixtures()

    @lru_cache
    def get_gw_predictors(gw_id: int, prediction_gw_id: int):
        player_stats = _get_gw_player_stats(gw_id, gw_stats)
        fixture_stats = _get_gw_fixture_stats(gw_id, fixtures)
        df = player_stats.join(fixture_stats, left_on="team_id", right_on="team")
        gw_delta = gw_id - prediction_gw_id
        return df.rename(
            {
                i: i.replace(f"_{gw_id}_", f"_{gw_delta}_")
                for i in df.columns
                if i.startswith("gw_")
            }
        )

    n = 1
    all_data = []
    while n + n_prediction_weeks + 1 < 39:
        prediction_gw = n + n_prediction_weeks
        gw_predictors_ = [
            get_gw_predictors(gw, prediction_gw)
            for gw in range(n, n_prediction_weeks + n)
        ]
        predictors = reduce(
            lambda x, y: x.join(y, on=["player_id", "team_id"]), gw_predictors_
        )
        response = gw_stats.filter(pl.col("gameweek") == prediction_gw)[
            "gameweek_points", "player_id", "team_id"
        ]
        df = predictors.join(
            response, on=["player_id", "team_id"], how="left", coalesce=True
        )
        df = df.with_columns(pl.Series("prediction_gw", [prediction_gw]))
        all_data.append(df)
        n += n_prediction_weeks + 1

    data = pl.concat(all_data)
    return _train_test_val_split(data, test_frac, val_frac)
