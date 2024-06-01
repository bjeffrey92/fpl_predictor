import polars as pl
import pytest
from polars.datatypes import Float64, Int32, Int64, Utf8

from fpl_predictor.squad_selection.linear_optimisation import (
    POSITION_MAX_SELECTIONS,
    PSCPSquadOptimiser,
    SquadOptimiser,
    StartingTeamOptimiser,
)


@pytest.fixture(scope="module")
def player_data() -> pl.DataFrame:
    return pl.read_csv(
        "tests/sample_data/player_data.csv",
        schema={
            "player_id": Int64,
            "team_id": Int64,
            "position_id": Int64,
            "name": Utf8,
            "cost_times_ten": Int64,
            "position": Utf8,
            "gameweek_points": Int64,
            "gameweek": Int32,
            "cost": Float64,
        },
    )


@pytest.fixture(scope="module")
def current_squad() -> pl.DataFrame:
    return pl.read_csv(
        "tests/sample_data/sample_squad.csv",
        schema={
            "player_id": Int64,
            "team_id": Int64,
            "position_id": Int64,
            "name": Utf8,
            "cost_times_ten": Int64,
            "position": Utf8,
            "gameweek_points": Int64,
            "gameweek": Int32,
            "cost": Float64,
        },
    )


def test_squad_optimiser(
    player_data: pl.DataFrame, current_squad: pl.DataFrame
) -> None:
    position_max_selections = {
        "GKP": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    squad_optimiser = SquadOptimiser(player_data)
    assert squad_optimiser.n_selections == 15
    assert squad_optimiser.total_cost == 100
    assert squad_optimiser.n_substitutions is None
    assert squad_optimiser.current_squad is None
    assert squad_optimiser.current_team_constraint is None
    assert len(squad_optimiser.constraints) == 4

    squad = squad_optimiser.optimise()
    assert squad.shape[0] == 15
    assert {
        position: counts
        for position, counts in squad["position"].value_counts().iter_rows()
    } == position_max_selections

    for n_substitutions in range(1, 10):
        squad_optimiser = SquadOptimiser(
            player_data, current_squad, n_substitutions=n_substitutions
        )
        assert squad_optimiser.n_selections == 15
        assert squad_optimiser.total_cost == 100
        assert squad_optimiser.n_substitutions == n_substitutions
        assert squad_optimiser.current_squad is not None
        assert len(squad_optimiser.constraints) == 5

        new_squad = squad_optimiser.optimise()
        assert new_squad.shape[0] == 15
        assert (
            new_squad["player_id"].is_in(current_squad["player_id"]).sum()
            >= squad_optimiser.n_selections - squad_optimiser.n_substitutions
        )
        assert {
            position: counts
            for position, counts in new_squad["position"].value_counts().iter_rows()
        } == position_max_selections


def test_starting_team_optimiser(current_squad: pl.DataFrame) -> None:
    starting_team_optimiser = StartingTeamOptimiser(current_squad)
    assert starting_team_optimiser.n_selections == 11
    assert len(starting_team_optimiser.constraints) == 2

    starting_team = starting_team_optimiser.optimise()
    assert starting_team.shape[0] == 11
    assert (
        starting_team["player_id"].is_in(current_squad["player_id"]).sum()
        == starting_team_optimiser.n_selections
    )

    player_position_counts = {
        position: counts
        for position, counts in starting_team["position"].value_counts().iter_rows()
    }
    position_min_selections = {
        "GKP": 1,
        "DEF": 3,
        "MID": 0,
        "FWD": 1,
    }
    position_max_selections = {
        "GKP": 1,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }
    for position in position_min_selections:
        assert (
            player_position_counts[position] >= position_min_selections[position]
        ), f"Too few {position}s selected"
        assert (
            player_position_counts[position] <= position_max_selections[position]
        ), f"Too many {position}s selected"


def test_pscp_squad_optimiser(player_data: pl.DataFrame) -> None:
    pscp_squad_optimiser = PSCPSquadOptimiser(
        player_data,
        players_to_preselect=4,
        teams_to_exclude_from_preselection=tuple(range(1, 19)),
    )
    assert pscp_squad_optimiser.selected_players["team_id"].is_in((19, 20)).all()
    assert (
        pscp_squad_optimiser.total_cost
        + pscp_squad_optimiser.selected_players["cost"].sum()
        == 100
    )
    assert pscp_squad_optimiser.n_selections == 11
    for pos, df in pscp_squad_optimiser.selected_players.group_by(["position"]):
        position: str = pos[0]  # type: ignore[index]
        assert (
            pscp_squad_optimiser.position_max_selections[position] + df.shape[0]
            == POSITION_MAX_SELECTIONS[position]
        )
    df = pscp_squad_optimiser.optimise()
    assert df.shape[0] == 15
