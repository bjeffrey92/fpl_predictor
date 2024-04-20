import copy
from abc import ABC, abstractmethod
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.preprocessing import OneHotEncoder

N_SELECTIONS: Final[int] = 15
TOTAL_COST: Final[float] = 100.0
POSITION_MAX_SELECTIONS: Final[dict[str, int]] = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


class _BaseOptimiser(ABC):
    def __init__(self, player_data: pl.DataFrame) -> None:
        self.player_data = player_data

    @property
    def n_players(self) -> int:
        return self.player_data.shape[0]

    @property
    @abstractmethod
    def n_selections(self) -> int:
        pass

    @property
    @abstractmethod
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

    def _constraints(self) -> list[LinearConstraint]:
        return [
            self.position_constraint,
            self.total_selections_constraint,
        ]

    @property
    def constraints(self) -> list[LinearConstraint]:
        return self._constraints()

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
    _n_selections = N_SELECTIONS
    _total_cost = TOTAL_COST
    _position_max_selections = copy.deepcopy(POSITION_MAX_SELECTIONS)

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
    def n_selections(self) -> int:
        return self._n_selections

    @n_selections.setter
    def n_selections(self, value: int) -> None:
        self._n_selections = value

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @total_cost.setter
    def total_cost(self, value: float) -> None:
        self._total_cost = value

    @property
    def position_max_selections(self) -> dict[str, int]:
        return self._position_max_selections.copy()

    @position_max_selections.setter
    def position_max_selections(self, value: dict[str, int]) -> None:
        self._position_max_selections = value

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

    def _constraints(self) -> list[LinearConstraint]:
        required_constraints = super()._constraints()
        return (
            required_constraints
            + [
                self.cost_constraint,
                self.team_constraint,
            ]
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


class PSCPSquadOptimiser(SquadOptimiser):
    def __init__(
        self,
        player_data: pl.DataFrame,
        players_to_preselect: int = 4,
        teams_to_exclude_from_preselection: tuple[int, ...] = (),
        current_squad: pl.DataFrame | None = None,
        n_substitutions: int | None = None,
    ) -> None:
        super().__init__(player_data, current_squad, n_substitutions)
        self.players_to_preselect = players_to_preselect
        self.teams_to_exclude_from_preselection = teams_to_exclude_from_preselection
        self._preselect_cheapest_players()

    def _preselect_cheapest_players(self) -> None:
        """
        Selects 4 cheapest players to be named in the squad but not the starting team.
        Must be within the position constraints.
        """
        max_selections = {
            "GKP": 1,
            "DEF": 2,
            "MID": 5,
            "FWD": 2,
        }  # max number of players in each position that can be selected but not named in the starting team

        selected_player_positions = {pos: 0 for pos in max_selections}
        df = self.player_data.filter(
            ~self.player_data["team_id"].is_in(self.teams_to_exclude_from_preselection)
        )
        players_by_cost = df.sort("cost")
        selected_players = [None for _ in range(self.players_to_preselect)]
        i = 0
        for row in players_by_cost.iter_rows(named=True):
            position = row["position"]
            if selected_player_positions[position] < max_selections[position]:
                selected_player_positions[position] += 1
                selected_players[i] = row  # type: ignore[call-overload]
                i += 1
            if i == self.players_to_preselect:
                break

        self.selected_players = pl.DataFrame(selected_players, self.player_data.schema)
        self._update_constraints()

    def _update_constraints(self) -> None:
        self.total_cost = self.total_cost - self.selected_players["cost"].sum()
        self.n_selections -= self.players_to_preselect
        selected_positions = self.selected_players["position"].value_counts()
        for pos, count in selected_positions.iter_rows():
            self.position_max_selections[pos] -= count
        self.player_data = self.player_data.filter(
            ~self.player_data["player_id"].is_in(self.selected_players["player_id"])
        )

    def optimise(self) -> pl.DataFrame:
        df = super().optimise()
        return pl.concat([df, self.selected_players])
