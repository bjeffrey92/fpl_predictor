import json
from pathlib import Path
from typing import Literal
from unittest import mock

import polars as pl
import pytest

from fpl_predictor.model_training import load_23_24_season_data


def test_s3_client() -> None:
    load_23_24_season_data._s3_client.cache_clear()
    with mock.patch.object(load_23_24_season_data, "boto3") as mock_boto3:
        assert load_23_24_season_data._s3_client() == mock_boto3.client.return_value


def test__load_data() -> None:
    s3_client = mock.Mock()
    mock_body = mock.Mock()
    mock_body.read.return_value = b"data"
    s3_client.get_object.return_value = {"Body": mock_body}
    with mock.patch.object(
        load_23_24_season_data, "_s3_client", return_value=s3_client
    ) as mock_s3_client:
        response = load_23_24_season_data._load_data("key")
        mock_s3_client.assert_called_once()
        mock_s3_client.return_value.get_object.assert_called_once_with(
            Bucket="fpl-data", Key="key"
        )
        assert response == "data"


def test_player_gameweek_stats() -> None:
    with mock.patch.object(load_23_24_season_data, "pl") as mock_pl:
        response = load_23_24_season_data._player_gameweek_stats()
        mock_pl.read_csv.assert_called_once_with(
            "s3://fpl-data/player_gameweek_stats_23-24.csv"
        )
        assert response == mock_pl.read_csv.return_value


def test_fixtures() -> None:
    with mock.patch.object(
        load_23_24_season_data, "_load_data", return_value='{"foo": "bar"}'
    ) as mock_load_data:
        response = load_23_24_season_data._fixtures()
        mock_load_data.assert_called_once_with("fixtures_23-24.json")
        assert response == {"foo": "bar"}


def test_get_gw_player_stats() -> None:
    gw_stats = pl.DataFrame(
        {
            "player_id": ["foo", "bar"],
            "team_id": ["foo", "bar"],
            "minutes": ["foo", "bar"],
            "goals_scored": ["foo", "bar"],
            "assists": ["foo", "bar"],
            "clean_sheets": ["foo", "bar"],
            "goals_conceded": ["foo", "bar"],
            "yellow_cards": ["foo", "bar"],
            "saves": ["foo", "bar"],
            "bonus": ["foo", "bar"],
            "influence": ["foo", "bar"],
            "creativity": ["foo", "bar"],
            "threat": ["foo", "bar"],
            "expected_goals": ["foo", "bar"],
            "expected_assists": ["foo", "bar"],
            "expected_goal_involvements": ["foo", "bar"],
            "expected_goals_conceded": ["foo", "bar"],
            "gameweek_points": ["foo", "bar"],
            "gameweek": [1, 2],
        }
    )
    response = load_23_24_season_data._get_gw_player_stats(1, gw_stats)
    assert response.to_dict(as_series=False) == {
        "gw_1_assists": ["foo"],
        "gw_1_bonus": ["foo"],
        "gw_1_clean_sheets": ["foo"],
        "gw_1_creativity": ["foo"],
        "gw_1_expected_assists": ["foo"],
        "gw_1_expected_goal_involvements": ["foo"],
        "gw_1_expected_goals": ["foo"],
        "gw_1_expected_goals_conceded": ["foo"],
        "gw_1_gameweek_points": ["foo"],
        "gw_1_goals_conceded": ["foo"],
        "gw_1_goals_scored": ["foo"],
        "gw_1_influence": ["foo"],
        "gw_1_minutes": ["foo"],
        "gw_1_saves": ["foo"],
        "gw_1_threat": ["foo"],
        "gw_1_yellow_cards": ["foo"],
        "player_id": ["foo"],
        "team_id": ["foo"],
    }


@pytest.mark.parametrize(
    "team,expected_response",
    (
        (
            "team_h",
            {
                "home_team": [True, True],
                "team": [6, 1],
                "team_difficulty": [5, 2],
                "team_score": [0, 2],
            },
        ),
        (
            "team_a",
            {
                "home_team": [False, False],
                "team": [13, 16],
                "team_difficulty": [2, 5],
                "team_score": [3, 1],
            },
        ),
    ),
)
def test_process_home_away_teams(
    team: Literal["team_h", "team_a"], expected_response: dict[str, list[int | bool]]
) -> None:
    gw_fixture_stats_df = pl.DataFrame(
        {
            "team_a": [13, 16],
            "team_a_score": [3, 1],
            "team_a_difficulty": [2, 5],
            "team_h": [6, 1],
            "team_h_score": [0, 2],
            "team_h_difficulty": [5, 2],
        }
    )
    assert (
        load_23_24_season_data._process_home_away_teams(
            gw_fixture_stats_df, team
        ).to_dict(as_series=False)
        == expected_response
    )


def test_get_gw_fixture_stats() -> None:
    gw_1_fixture = {
        "event": 1,
        "team_a": 13,
        "team_a_difficulty": 2,
        "team_a_score": 3,
        "team_h": 6,
        "team_h_difficulty": 5,
        "team_h_score": 0,
    }
    response = load_23_24_season_data._get_gw_fixture_stats(1, [gw_1_fixture])  # type: ignore[list-item]
    assert response.to_dict(as_series=False) == {
        "team": [6, 13],
        "gw_1_team_score": [0, 3],
        "gw_1_team_difficulty": [5, 2],
        "gw_1_home_team": [True, False],
        "gw_1_opposition_team_score": [3, 0],
        "gw_1_opposition_team_difficulty": [2, 5],
    }


def test_train_test_val_split() -> None:
    data = pl.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "team_id": [1, 1, 1, 1, 1],
            "gw_-1_minutes": [0, 0, 0, 0, 4],
            "gw_-1_goals_scored": [0, 0, 0, 0, 0],
            "gw_-1_assists": [0, 0, 0, 0, 0],
            "gw_-1_clean_sheets": [0, 0, 0, 0, 0],
            "gw_-1_goals_conceded": [0, 0, 0, 0, 0],
            "gw_-1_team_score": [2, 2, 2, 2, 2],
            "gw_-1_team_difficulty": [2, 2, 2, 2, 2],
            "gw_-1_home_team": [True, True, True, True, True],
            "gw_-1_opposition_team_score": [1, 1, 1, 1, 1],
            "gw_-1_opposition_team_difficulty": [5, 5, 5, 5, 5],
            "gameweek_points": [0, 0, 0, 0, 1],
            "prediction_gw": [2, 2, 2, 2, 2],
        }
    )
    response = load_23_24_season_data._train_test_val_split(data, 0.2, 0.2)
    assert isinstance(response, load_23_24_season_data.TrainTestValData)
    for i in [response.train_X, response.test_X, response.val_X]:
        assert isinstance(i, pl.DataFrame)
        assert i.shape[1] == 10
    for j in [response.train_y, response.test_y, response.val_y]:
        assert isinstance(j, pl.Series)


@pytest.fixture
def gw_stats(fixtures_dir: Path) -> pl.DataFrame:
    fpath = fixtures_dir.joinpath("gw_stats.parquet")
    return pl.read_parquet(fpath)


@pytest.fixture
def gw_fixtures(fixtures_dir: Path) -> pl.DataFrame:
    fpath = fixtures_dir.joinpath("gw_fixtures.json")
    return json.loads(fpath.read_text())


def test_load_data(gw_stats: pl.DataFrame, gw_fixtures: pl.DataFrame) -> None:
    with mock.patch.object(
        load_23_24_season_data, "_player_gameweek_stats", return_value=gw_stats
    ), mock.patch.object(load_23_24_season_data, "_fixtures", return_value=gw_fixtures):
        response = load_23_24_season_data.load_data(1)
        assert isinstance(response, load_23_24_season_data.TrainTestValData)
