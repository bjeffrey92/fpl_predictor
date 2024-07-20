"""
Select the first squad of the season using each player's median gameweek points from the
previous season
"""

import polars as pl

from fpl_predictor import player_stats
from fpl_predictor.model_training import load_23_24_season_data
from fpl_predictor.settings import N_WORST_TEAMS
from fpl_predictor.squad_selection import linear_optimisation, squad_selection


def _prev_season_median_gw_points() -> pl.DataFrame:
    player_data = player_stats.get_player_data()
    previous_season_player_stats = load_23_24_season_data.player_gameweek_stats(
        columns=["name", "position_id", "team_id", "gameweek_points"]
    )
    median_points_gw_points = previous_season_player_stats.group_by(
        ["name", "position_id", "team_id"]
    ).median()
    return player_data.join(
        median_points_gw_points, on=["name", "position_id", "team_id"]
    )


def _select_optimal_squad(gw_points: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    points_per_team = gw_points.group_by("team_id").agg(pl.col("gameweek_points").sum())
    worst_teams = points_per_team.sort("gameweek_points").head(N_WORST_TEAMS)["team_id"]
    squad = linear_optimisation.PSCPSquadOptimiser(
        gw_points, teams_to_exclude_from_preselection=tuple(worst_teams)
    ).optimise()
    starting_team = linear_optimisation.StartingTeamOptimiser(squad).optimise()
    return squad_selection.annotate_squad_and_compute_points(
        squad, starting_team, None, 0
    )


def main() -> None:
    gw_points = _prev_season_median_gw_points()
    gw_points = gw_points.with_columns((pl.col("cost_times_ten") / 10).alias("cost"))
    annotated_squad, expected_points = _select_optimal_squad(gw_points)
    print(f"{expected_points=}")
    annotated_squad.write_csv("starting_squad.csv")
