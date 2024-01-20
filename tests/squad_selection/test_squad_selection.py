from unittest.mock import MagicMock, call, patch

import polars as pl
import pytest

from fpl_predictor.squad_selection import squad_selection


@patch("fpl_predictor.squad_selection.squad_selection.SCORE_PREDICTOR_FACTORY")
@patch("fpl_predictor.squad_selection.squad_selection.get_player_data")
def test_get_player_data(
    mock_get_player_data: MagicMock, mock_score_predictor_factory: MagicMock
) -> None:
    mock_gw_predictions = pl.DataFrame(
        {"player_id": [1, 2, 3], "gameweek_points": [5, 6, 7]}
    )
    mock_player_data = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "team_id": [1, 2, 3],
            "position": ["FWD", "MID", "DEF"],
            "cost_times_ten": [100, 200, 300],
        }
    )
    mock_predictor = MagicMock()
    mock_predictor.predict_gw_scores.return_value = mock_gw_predictions
    mock_score_predictor_factory.__getitem__.return_value = mock_predictor
    mock_get_player_data.return_value = mock_player_data

    result = squad_selection._get_player_data(1, "method")
    expected_result = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "team_id": [1, 2, 3],
            "position": ["FWD", "MID", "DEF"],
            "cost": [10.0, 20.0, 30.0],
            "gameweek_points": [5, 6, 7],
        }
    )
    assert result.frame_equal(expected_result)


@pytest.mark.parametrize(
    "n_transfers, n_free_transfers, expected_points",
    [(None, 1, 25), (2, 1, 21), (3, 2, 21)],
)
def test_annotate_squad_and_compute_points(
    n_transfers: int | None, n_free_transfers: int, expected_points: int
) -> None:
    player_ids = [1, 2, 3]
    mock_squad = pl.DataFrame(
        {
            "player_id": player_ids,
            "position": ["FWD", "MID", "DEF"],
            "team_id": [1, 2, 3],
            "cost": [10.0, 20.0, 30.0],
            "gameweek_points": [5, 6, 7],
        }
    )
    player_data = pl.DataFrame(
        {
            "player_id": player_ids,
            "team_id": [1, 2, 3],
            "position": ["FWD", "MID", "DEF"],
            "cost": [10.0, 20.0, 30.0],
            "gameweek_points": [5, 6, 7],
        }
    )

    squad, total_points = squad_selection._annotate_squad_and_compute_points(
        mock_squad, player_data, n_transfers, n_free_transfers
    )
    expected_squad = pl.DataFrame(
        {
            "player_id": player_ids,
            "position": ["FWD", "MID", "DEF"],
            "team_id": [1, 2, 3],
            "cost": [10.0, 20.0, 30.0],
            "gameweek_points": [5, 6, 7],
            "starting": [True, True, True],
            "player_annotation": ["", "vice_captain", "captain"],
        }
    )
    assert squad.sort("player_id").equals(expected_squad.sort("player_id"))
    assert total_points == expected_points


@patch("fpl_predictor.squad_selection.squad_selection._squad_and_predicted_score")
def test_select_squad(mock_squad_and_predicted_score: MagicMock) -> None:
    mock_squad_and_predicted_score.return_value = (pl.DataFrame(), 0)

    gameweek = 1
    current_squad = pl.DataFrame()
    n_free_transfers = 1
    prediction_method = "median_past_score"

    squad_selection.select_squad(
        gameweek,
        current_squad=current_squad,
        n_free_transfers=n_free_transfers,
        prediction_method=prediction_method,
    )

    # Check that _squad_and_predicted_score was called the expected number of times with the expected arguments
    expected_calls = [
        call(
            gameweek,
            current_squad,
            n,
            n_free_transfers,
            prediction_method=prediction_method,
        )
        for n in range(1, 15)
    ]
    mock_squad_and_predicted_score.assert_has_calls(expected_calls, any_order=True)
