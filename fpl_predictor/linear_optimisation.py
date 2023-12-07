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
    def position_max_selections(self) -> dict[str, int | None]:
        pass

    @abstractproperty
    def position_constraint(self) -> LinearConstraint:
        pass

    @abstractproperty
    def cost_constraint(self) -> LinearConstraint | None:
        pass

    @abstractproperty
    def team_constraint(self) -> LinearConstraint | None:
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
        required_constraints = [
            self.position_constraint,
            self.total_selections_constraint,
        ]
        optional_constraints = [
            i for i in [self.cost_constraint, self.team_constraint] if i is not None
        ]
        return required_constraints + optional_constraints

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
    position_max_selections = {
        "GKP": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }
    n_selections = 15
    total_cost = 100

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
        position_encoder, position_encoded_data = self.get_position_encoder()
        pos_requirements = np.array(
            [
                self.position_max_selections[pos]
                for pos in position_encoder.categories_[0]
            ]
        )
        return LinearConstraint(
            position_encoded_data.transpose(), pos_requirements, pos_requirements
        )


class StartingTeamOptimiser(_BaseOptimiser):
    n_selections = 11
    position_max_selections = {
        "GKP": 1,
        "DEF": 3,
        "MID": n_selections - 5,
        "FWD": 1,
    }
    cost_constraint = None
    team_constraint = None

    @property
    def position_constraint(self) -> LinearConstraint:
        position_encoder, position_encoded_data = self.get_position_encoder()
        pos_requirements = np.array(
            [
                self.position_max_selections[pos]
                for pos in position_encoder.categories_[0]
            ]
        )
        return LinearConstraint(
            position_encoded_data.transpose(),
            np.zeros(len(pos_requirements)),
            pos_requirements,
        )


if __name__ == "__main__":
    player_data = load_player_gameweek_data(1)
    squad_optimiser = SquadOptimiser(player_data)
    best_squad = squad_optimiser.optimise()
    starting_team_optimiser = StartingTeamOptimiser(best_squad)
