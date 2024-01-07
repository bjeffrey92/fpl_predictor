from abc import ABC, abstractproperty

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.preprocessing import OneHotEncoder

from fpl_predictor.player_stats import load_player_gameweek_data


class _BaseOptimiser(ABC):
    def __init__(self, player_data: pl.DataFrame) -> None:
        self.player_data = player_data
        self.n_players = player_data.shape[0]

        self.constraints = self.get_constraints()

    @abstractproperty
    def n_selections(self) -> int:
        pass

    @abstractproperty
    def position_constraint(self) -> LinearConstraint:
        pass

    @property
    def total_selections_constraint(self) -> LinearConstraint:
        return LinearConstraint(
            np.ones(self.n_players), self.n_selections, self.n_selections
        )

    def get_position_encoder(self) -> tuple[OneHotEncoder, np.ndarray]:
        position_encoder = OneHotEncoder(sparse_output=False).fit(
            self.player_data.select("position")
        )
        position_encoded_data = position_encoder.transform(
            self.player_data.select("position")
        )
        return position_encoder, position_encoded_data

    def get_constraints(self) -> list[LinearConstraint]:
        return [
            self.position_constraint,
            self.total_selections_constraint,
        ]

    def optimise(self) -> pl.DataFrame:
        gw_points = self.player_data.select("gameweek_points").to_numpy().squeeze()
        res = milp(
            c=-gw_points,  # minimise negative gameweek points
            constraints=self.constraints,
            integrality=np.ones(self.n_players),  # all decision variables are integers
            bounds=Bounds(0, 1),  # decision variable can be only one or zero
        )
        assert res.success, "Optimisation failed"
        selections = np.round(res.x).astype(bool)
        return self.player_data.filter(selections)


class SquadOptimiser(_BaseOptimiser):
    n_selections = 15
    total_cost = 100

    def __init__(
        self,
        player_data: pl.DataFrame,
        current_squad: pl.DataFrame | None = None,
        n_substitutions: int | None = None,
    ) -> None:
        self.current_squad = current_squad
        self.n_substitutions = n_substitutions
        super().__init__(player_data)

    @property
    def cost_constraint(self) -> LinearConstraint | None:
        cost_matrix = self.player_data.select("cost").to_numpy().squeeze()
        return LinearConstraint(cost_matrix, 0, self.total_cost)

    @property
    def team_constraint(self) -> LinearConstraint | None:
        team_encoder = OneHotEncoder(sparse_output=False).fit(
            self.player_data.select("team_id")
        )
        team_encoding = team_encoder.transform(self.player_data.select("team_id"))
        n_teams = len(team_encoder.categories_[0])
        return LinearConstraint(
            team_encoding.transpose(), np.zeros(n_teams), np.ones(n_teams) * 3
        )

    @property
    def position_constraint(self) -> LinearConstraint:
        position_max_selections = {
            "GKP": 2,
            "DEF": 5,
            "MID": 5,
            "FWD": 3,
        }
        position_encoder, position_encoded_data = self.get_position_encoder()
        pos_requirements = np.array(
            [position_max_selections[pos] for pos in position_encoder.categories_[0]]
        )
        return LinearConstraint(
            position_encoded_data.transpose(), pos_requirements, pos_requirements
        )

    @property
    def current_team_constraint(self) -> LinearConstraint | None:
        if self.current_squad is None:
            return None
        players_in_current_team = self.player_data["player_id"].is_in(
            self.current_squad["player_id"]
        )
        current_team_encoded_data = players_in_current_team.cast(int).to_numpy()
        return LinearConstraint(
            current_team_encoded_data,
            len(self.current_squad) - self.n_substitutions,  # type: ignore[operator]
            len(self.current_squad),
        )

    def get_constraints(self) -> list[LinearConstraint]:
        required_constraints = super().get_constraints()
        return (
            required_constraints
            + [self.cost_constraint, self.team_constraint]
            + ([self.current_team_constraint] if self.current_team_constraint else [])
        )


class StartingTeamOptimiser(_BaseOptimiser):
    n_selections = 11

    @property
    def position_constraint(self) -> LinearConstraint:
        position_encoder, position_encoded_data = self.get_position_encoder()
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
        min_pos_requirements = np.array(
            [position_min_selections[pos] for pos in position_encoder.categories_[0]]
        )
        max_pos_requirements = np.array(
            [position_max_selections[pos] for pos in position_encoder.categories_[0]]
        )
        return LinearConstraint(
            position_encoded_data.transpose(),
            min_pos_requirements,
            max_pos_requirements,
        )


if __name__ == "__main__":
    player_data = load_player_gameweek_data(1)
    squad_optimiser = SquadOptimiser(player_data)
    best_squad = squad_optimiser.optimise()
    starting_team_optimiser = StartingTeamOptimiser(best_squad)
    starting_team = starting_team_optimiser.optimise()

    current_squad = best_squad
    player_data = load_player_gameweek_data(2)
    squad_optimiser = SquadOptimiser(player_data, current_squad, n_substitutions=1)
    new_squad = squad_optimiser.optimise()
    starting_team_optimiser = StartingTeamOptimiser(new_squad)
    new_starting_team = starting_team_optimiser.optimise()
