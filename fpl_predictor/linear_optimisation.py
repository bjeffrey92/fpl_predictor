import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.preprocessing import OneHotEncoder

from fpl_predictor.player_stats import get_player_data, get_player_gameweek_stats

n_selections = 15
total_cost = 100
gameweek = 1

player_data = get_player_data()
gw_stats = get_player_gameweek_stats(gameweek)


player_data = player_data.join(gw_stats, on="player_id")
player_data = player_data.with_columns((pl.col("cost_times_ten") / 10).alias("cost"))

team_encoder = OneHotEncoder(sparse_output=False).fit(player_data.select("team_id"))
position_encoder = OneHotEncoder(sparse_output=False).fit(
    player_data.select("position")
)

df = player_data
vector_length = df.shape[0]

team_encoding = team_encoder.transform(df.select("team_id"))
position_encoding = position_encoder.transform(player_data.select("position"))

position_constraints = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}
pos_requirements = np.array(
    [position_constraints[pos] for pos in position_encoder.categories_[0]]
)

cost_matrix = df.select("cost").to_numpy().squeeze()
gw_points = df.select("gameweek_points").to_numpy().squeeze()

constraints = [
    LinearConstraint(cost_matrix, 0, total_cost),
    LinearConstraint(team_encoding.transpose(), np.zeros(20), np.ones(20) * 3),
    LinearConstraint(position_encoding.transpose(), pos_requirements, pos_requirements),
    LinearConstraint(
        np.ones(vector_length), n_selections, n_selections
    ),  # ensure that n_selections are made
]


res = milp(
    c=-gw_points,  # minimise negative gameweek points
    constraints=constraints,
    integrality=np.ones(vector_length),  # all decision variables are integers
    bounds=Bounds(0, 1),  # decision variable can be only one or zero
)
selections = np.round(res.x).astype(bool)
df.filter(selections)
