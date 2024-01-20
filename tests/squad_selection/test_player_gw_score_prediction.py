from unittest.mock import patch

import polars as pl

from fpl_predictor.squad_selection.player_gw_score_prediction import MedianPastScore


def test_median_past_score() -> None:
    def mock_get_player_gameweek_stats(upcoming_gameweek: int) -> pl.DataFrame:
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
