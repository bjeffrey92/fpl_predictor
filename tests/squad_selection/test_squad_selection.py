from unittest import mock

import polars as pl
import pytest

from fpl_predictor.squad_selection import linear_optimisation, squad_selection


@mock.patch("fpl_predictor.squad_selection.squad_selection.SCORE_PREDICTOR_FACTORY")
@mock.patch("fpl_predictor.squad_selection.squad_selection.get_player_data")
def test_get_player_data(
    mock_get_player_data: mock.MagicMock, mock_score_predictor_factory: mock.MagicMock
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
    mock_predictor = mock.MagicMock()
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


@pytest.mark.parametrize(
    "squad_selection_method", ("preselect_cheapest_players", "naive", "invalid")
)
def test_squad_and_predicted_score(squad_selection_method: str) -> None:
    mock_pscp_squad_optimiser = mock.Mock(linear_optimisation.PSCPSquadOptimiser)
    mock_starting_team_optimiser = mock.Mock(linear_optimisation.StartingTeamOptimiser)
    mock_squad_optimiser = mock.Mock(linear_optimisation.SquadOptimiser)
    with mock.patch.object(
        squad_selection,
        "_get_player_data",
        return_value=pl.DataFrame({"team_id": [], "gameweek_points": []}),
    ) as mock_get_player_data, mock.patch.object(
        squad_selection, "_annotate_squad_and_compute_points"
    ) as mock_annotate_squad_and_compute_points, mock.patch(
        "fpl_predictor.squad_selection.squad_selection.PSCPSquadOptimiser",
        mock_pscp_squad_optimiser,
    ), mock.patch(
        "fpl_predictor.squad_selection.squad_selection.StartingTeamOptimiser",
        mock_starting_team_optimiser,
    ), mock.patch(
        "fpl_predictor.squad_selection.squad_selection.SquadOptimiser",
        mock_squad_optimiser,
    ), mock.patch(
        "fpl_predictor.squad_selection.squad_selection.SQUAD_SELECTION_METHOD",
        squad_selection_method,
    ):
        if squad_selection_method == "invalid":
            with pytest.raises(ValueError):
                squad_selection._squad_and_predicted_score(
                    1, prediction_method="pred_method"
                )
        else:
            squad_selection._squad_and_predicted_score(
                1, prediction_method="pred_method"
            )
            mock_annotate_squad_and_compute_points.assert_called_once()
            mock_starting_team_optimiser.assert_called_once()
            mock_starting_team_optimiser.return_value.optimise.assert_called_once()

        mock_get_player_data.assert_called_once_with(1, "pred_method")
        if squad_selection_method == "preselect_cheapest_players":
            mock_pscp_squad_optimiser.assert_called_once()
            mock_pscp_squad_optimiser.return_value.optimise.assert_called_once()
        elif squad_selection_method == "naive":
            mock_squad_optimiser.assert_called_once()
            mock_squad_optimiser.return_value.optimise.assert_called_once()


@mock.patch("fpl_predictor.squad_selection.squad_selection._squad_and_predicted_score")
def test_select_squad(mock_squad_and_predicted_score: mock.MagicMock) -> None:
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
        mock.call(
            gameweek,
            current_squad,
            n,
            n_free_transfers,
            prediction_method=prediction_method,
        )
        for n in range(1, 15)
    ]
    mock_squad_and_predicted_score.assert_has_calls(expected_calls, any_order=True)
