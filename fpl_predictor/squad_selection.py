from functools import cache

import polars as pl

from .linear_optimisation import SquadOptimiser, StartingTeamOptimiser
from .player_gw_score_prediction import MedianPastScore


@cache
def _get_player_data(gameweek: int) -> pl.DataFrame:
    MedianPastScore(gameweek, 5, 3).predict_gw_scores()
    # TODO: append with other columns needed for squad optimiser, maybe do this inside the class


def _annotate_squad_and_compute_points(  # type: ignore
    squad: pl.DataFrame, starting_team: pl.DataFrame
) -> tuple[pl.DataFrame, int]:
    ...


def _squad_and_predicted_score(
    gameweek: int,
    current_squad: pl.DataFrame | None = None,
    n_substitutions: int | None = None,
) -> tuple[pl.DataFrame, int]:
    player_data = _get_player_data(gameweek)

    squad = SquadOptimiser(player_data, current_squad, n_substitutions).optimise()
    starting_team = StartingTeamOptimiser(squad).optimise()
    return _annotate_squad_and_compute_points(squad, starting_team)


def select_squad(
    gameweek: int, current_squad: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, int]:
    if current_squad is None:
        return _squad_and_predicted_score(gameweek)

    predicted_points_and_squad = {
        points: squad
        for squad, points in (
            _squad_and_predicted_score(gameweek, current_squad, n) for n in range(1, 15)
        )
    }
    max_points = max(predicted_points_and_squad)
    return predicted_points_and_squad[max_points], max_points
