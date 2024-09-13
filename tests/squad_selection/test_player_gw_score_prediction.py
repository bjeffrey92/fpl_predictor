import os
from unittest import mock
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fpl_predictor.squad_selection import player_gw_score_prediction
from fpl_predictor.squad_selection.player_gw_score_prediction import MedianPastScore


def test_process_home_away_teams() -> None:
    gw_fixture_stats: dict[str, object] = {
        "team_a": 5,
        "team_a_score": 1,
        "team_h": 1,
        "team_h_score": 1,
        "team_h_difficulty": 2,
        "team_a_difficulty": 5,
    }
    result = player_gw_score_prediction._process_home_away_teams(gw_fixture_stats)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "team_id": [1, 5],
                "team_score": [1, 1],
                "team_difficulty": [2, 5],
                "home_team": [True, False],
                "opposition_team_score": [1, 1],
                "opposition_team_difficulty": [5, 2],
            }
        ),
    )


def test_player_gw_score_prediction() -> None:
    df = pl.DataFrame(
        {
            "position": ["MID", "FWD", "DEF", "FWD", "GKP"],
            "player_id": [1, 2, 3, 4, 5],
            "team_id": [1, 1, 1, 1, 1],
        }
    )
    result = player_gw_score_prediction._append_position_encodings(df)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "team_id": [1, 1, 1, 1, 1],
                "DEF": [0.0, 0.0, 1.0, 0.0, 0.0],
                "FWD": [0.0, 1.0, 0.0, 1.0, 0.0],
                "GKP": [0.0, 0.0, 0.0, 0.0, 1.0],
                "MID": [1.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
    )


def test_append_prediction_gameweek_team_stats() -> None:
    prediction_gameweek = 1
    mock_fixtures = [
        {
            "team_a": i,
            "team_h": i + 10,
            "team_h_difficulty": 1,
            "team_a_difficulty": 2,
            "event": prediction_gameweek,
        }
        for i in range(1, 11)
    ]
    mock_fixtures.extend(
        [i | {"event": prediction_gameweek + 1} for i in mock_fixtures]
    )
    data = pl.DataFrame({"player_id": [1, 2, 3, 4, 5], "team_id": [1, 2, 3, 4, 1]})
    with patch(
        f"{player_gw_score_prediction.__name__}.get_fixtures",
        return_value=mock_fixtures,
    ):
        result = player_gw_score_prediction._append_prediction_gameweek_team_stats(
            data, prediction_gameweek
        )
        assert_frame_equal(
            result,
            pl.DataFrame(
                {
                    "home_team": [False] * 5,
                    "opposition_team_difficulty": [1] * 5,
                    "player_id": [1, 5, 2, 3, 4],
                    "team_difficulty": [2] * 5,
                }
            ),
            check_column_order=False,
        )


def test_median_past_score() -> None:
    def mock_get_player_gameweek_stats(
        upcoming_gameweek: int, cols: list[str]
    ) -> pl.DataFrame:
        if upcoming_gameweek == 6:
            return pl.DataFrame({"player_id": [1, 2, 3], "points": [2, 6, 7]})
        else:
            return pl.DataFrame(
                {
                    "player_id": [1],
                    "points": [5],
                }
            )

    n_previous_weeks = 5
    with patch(
        "fpl_predictor.squad_selection.player_gw_score_prediction.get_player_gameweek_stats",
        side_effect=mock_get_player_gameweek_stats,
    ) as mock_get_stats:
        median_past_score = MedianPastScore(
            upcoming_gameweek=6, n_previous_weeks=n_previous_weeks, min_required_weeks=2
        )

        result = median_past_score.predict_gw_scores()
        expected_result = pl.DataFrame({"player_id": [1], "points": [5.0]})
        assert result.equals(expected_result)

        assert mock_get_stats.call_count == n_previous_weeks


def test_xgboost_init() -> None:
    with patch(
        f"{player_gw_score_prediction.__name__}.XGBoost._load_model"
    ) as mock_load_model, patch(
        f"{player_gw_score_prediction.__name__}.XGBoost._load_data"
    ) as mock_load_data:
        xgboost = player_gw_score_prediction.XGBoost(3, 2)
        mock_load_model.assert_called_once()
        mock_load_data.assert_called_once()
        assert xgboost.n_prediction_weeks == 2
        assert xgboost.gameweek == 3
        assert xgboost.model == mock_load_model.return_value
        assert xgboost.data == mock_load_data.return_value


@pytest.mark.parametrize("raise_exception", (True, False))
def test_xgboost_load_fixture_data(raise_exception: bool) -> None:
    gameweek = 3
    mock_fixtures = [
        {
            "team_a": i,
            "team_h": i + 10,
            "team_h_difficulty": 1,
            "team_a_difficulty": 2,
            "team_a_score": 1,
            "team_h_score": 1,
            "finished": True,
            "event": gameweek - 1,
        }
        for i in range(1, 11)
    ]
    mock_fixtures.extend([i | {"event": gameweek - 2} for i in mock_fixtures])
    if raise_exception:
        mock_fixtures[-1]["finished"] = False
    with patch(f"{player_gw_score_prediction.__name__}.XGBoost._load_model"), patch(
        f"{player_gw_score_prediction.__name__}.XGBoost._load_data"
    ), patch(
        f"{player_gw_score_prediction.__name__}.get_fixtures",
        return_value=mock_fixtures,
    ):
        xgboost = player_gw_score_prediction.XGBoost(gameweek, 2)
        if raise_exception:
            with pytest.raises(ValueError):
                xgboost._load_fixture_data()
        else:
            response = xgboost._load_fixture_data()
            assert isinstance(response, dict)
            assert sorted(response.keys()) == ["gw_-1", "gw_-2"]
            for v in response.values():
                assert isinstance(v, pl.DataFrame)
                assert v.shape == (20, 6)
                assert v.columns == [
                    "team_id",
                    "team_score",
                    "team_difficulty",
                    "home_team",
                    "opposition_team_score",
                    "opposition_team_difficulty",
                ]


def test_xgboost_load_model() -> None:
    mock_fs = mock.MagicMock()
    n_prediction_weeks = 2
    with patch(
        f"{player_gw_score_prediction.__name__}.s3fs.S3FileSystem", return_value=mock_fs
    ), patch(f"{player_gw_score_prediction.__name__}.XGBoost._load_data"), patch(
        f"{player_gw_score_prediction.__name__}.joblib.load"
    ) as mock_joblib_load:
        xgboost = player_gw_score_prediction.XGBoost(3, n_prediction_weeks)
        assert xgboost.model == mock_joblib_load.return_value
        mock_fs.open.assert_called_once_with(
            os.path.join(
                "s3://",
                xgboost._bucket,
                xgboost._key_pattern.format(n_prediction_weeks),
            ),
            encoding="utf8",
        )


@pytest.mark.parametrize("raise_exception", (True, False))
def test_xgboost_predict_gw_scores(raise_exception: bool) -> None:
    mock_model = mock.Mock()
    mock_model.prediction_columns = ["player_id", "foo", "bar"]
    mock_model.model.predict.return_value = [1, 2, 3]
    mock_data = pl.DataFrame(
        {"player_id": [1, 2, 3], "foo": [1, 2, 3], "bar": [4, 5, 6]}
    )
    if raise_exception:
        mock_data = mock_data.drop("player_id")
    with patch(
        f"{player_gw_score_prediction.__name__}.XGBoost._load_model",
        return_value=mock_model,
    ), patch(
        f"{player_gw_score_prediction.__name__}.XGBoost._load_data",
        return_value=mock_data,
    ):
        xgboost = player_gw_score_prediction.XGBoost(3, 2)
        if raise_exception:
            with pytest.raises(ValueError):
                xgboost.predict_gw_scores()
        else:
            response = xgboost.predict_gw_scores()
            assert_frame_equal(
                response,
                pl.DataFrame({"player_id": [1, 2, 3], "gameweek_points": [1, 2, 3]}),
            )
