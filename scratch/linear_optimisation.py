from fpl_predictor.player_stats import load_player_gameweek_data
from fpl_predictor.squad_selection.linear_optimisation import (
    SquadOptimiser,
    StartingTeamOptimiser,
)

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
