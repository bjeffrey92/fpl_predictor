from functools import cache

import polars as pl

from fpl_predictor.linear_optimisation import SquadOptimiser, StartingTeamOptimiser
from fpl_predictor.player_gw_score_prediction import SCORE_PREDICTOR_FACTORY
from fpl_predictor.player_stats import get_player_data


@cache
def _get_player_data(gameweek: int, prediction_method: str, **kwargs) -> pl.DataFrame:
    predictor = SCORE_PREDICTOR_FACTORY[prediction_method]
    gw_predictions = predictor(gameweek, **kwargs).predict_gw_scores()
    player_data = get_player_data()
    df = gw_predictions.join(player_data, on="player_id")
    return df.select(
        pl.col("player_id"),
        pl.col("team_id"),
        pl.col("position"),
        (pl.col("cost_times_ten") / 10).alias("cost"),
        pl.col("gameweek_points"),
    )


def _annotate_squad_and_compute_points(
    squad: pl.DataFrame,
    starting_team: pl.DataFrame,
    n_transfers: int | None,
    n_free_transfers: int,
) -> tuple[pl.DataFrame, int]:
    starting_team = starting_team.with_columns(pl.lit(True).alias("starting")).select(
        pl.col("player_id"), pl.col("starting")
    )
    squad = squad.join(starting_team, on="player_id", how="left").fill_null(False)
    squad = squad.sort(["starting", "gameweek_points"], descending=[True, True])
    squad = squad.with_columns(
        pl.Series(
            name="player_annotation",
            values=["captain", "vice_captain"] + [""] * (len(squad) - 2),
        )
    )

    # captain's points are doubled
    total_points = (
        squad.filter(squad["starting"])["gameweek_points"].sum()
        + squad["gameweek_points"][0]
    )
    if n_transfers is not None and n_transfers > n_free_transfers:
        total_points -= (
            n_transfers - n_free_transfers
        ) * 4  # -4 points for each transfer which isn't a free transfer

    return squad, total_points


def _squad_and_predicted_score(
    gameweek: int,
    current_squad: pl.DataFrame | None = None,
    n_transfers: int | None = None,
    n_free_transfers: int = 1,
    prediction_method: str = "median_past_score",
    **kwargs
) -> tuple[pl.DataFrame, int]:
    player_data = _get_player_data(gameweek, prediction_method, **kwargs)

    squad = SquadOptimiser(player_data, current_squad, n_transfers).optimise()
    starting_team = StartingTeamOptimiser(squad).optimise()
    return _annotate_squad_and_compute_points(
        squad, starting_team, n_transfers, n_free_transfers
    )


def select_squad(
    gameweek: int,
    *,
    current_squad: pl.DataFrame | None = None,
    n_free_transfers: int = 1,
    prediction_method: str = "median_past_score",
    **kwargs
) -> tuple[pl.DataFrame, int]:
    if current_squad is None:
        return _squad_and_predicted_score(
            gameweek, prediction_method=prediction_method, **kwargs
        )

    predicted_points_and_squad = {
        points: squad
        for squad, points in (
            _squad_and_predicted_score(
                gameweek,
                current_squad,
                n,
                n_free_transfers,
                prediction_method=prediction_method,
                **kwargs
            )
            for n in range(1, 15)
        )
    }
    max_points = max(predicted_points_and_squad)
    return predicted_points_and_squad[max_points], max_points
