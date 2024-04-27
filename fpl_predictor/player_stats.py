from functools import cache

import jmespath
import polars as pl

from fpl_predictor.utils import get


def get_player_gameweek_stats(gameweek: int) -> pl.DataFrame:
    url = f"https://fantasy.premierleague.com/api/event/{gameweek}/live/"
    data = get(url)
    player_gw_stats = jmespath.search(
        "elements[*].{player_id: id, gameweek_points: stats.total_points}", data
    )
    player_gw_stats_df = pl.DataFrame(player_gw_stats)
    return player_gw_stats_df.with_columns(pl.lit(gameweek).alias("gameweek"))


@cache
def get_player_data() -> pl.DataFrame:
    data = get("https://fantasy.premierleague.com/api/bootstrap-static/")
    player_data = jmespath.search(
        "elements[*].{player_id: id, team_id: team, position_id: element_type, name: web_name, cost_times_ten: now_cost}",
        data,
    )
    position_data = jmespath.search(
        "element_types[*].{position_id: id, position: singular_name_short}", data
    )
    player_data_df = pl.DataFrame(player_data)
    position_data_df = pl.DataFrame(position_data)
    return player_data_df.join(position_data_df, on="position_id")


def load_player_gameweek_data(gameweek: int) -> pl.DataFrame:
    gw_stats = get_player_gameweek_stats(gameweek)
    raw_player_data = get_player_data()
    player_data = raw_player_data.join(gw_stats, on="player_id")
    return player_data.with_columns((pl.col("cost_times_ten") / 10).alias("cost"))
