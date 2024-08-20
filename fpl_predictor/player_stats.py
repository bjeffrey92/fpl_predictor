from functools import cache
from typing import Iterable, cast

import jmespath
import polars as pl

from fpl_predictor.utils import get


def get_player_gameweek_stats(
    gameweek: int, cols: Iterable[str] | None = None
) -> pl.DataFrame:
    url = f"https://fantasy.premierleague.com/api/event/{gameweek}/live/"
    data = get(url)
    search_pattern = "elements[*].{player_id: id, minutes: stats.minutes, goals_scored: stats.goals_scored, \
        assists: stats.assists, clean_sheets: stats.clean_sheets, goals_conceded: stats.goals_conceded, \
        own_goals: stats.own_goals, penalties_saved: stats.penalties_saved, penalties_missed: stats.penalties_missed, \
        yellow_cards: stats.yellow_cards, red_cards: stats.red_cards, saves: stats.saves, bonus: stats.bonus, \
        bps: stats.bps, influence: stats.influence, creativity: stats.creativity, threat: stats.threat, \
        ict_index: stats.ict_index, starts: stats.starts, expected_goals: stats.expected_goals, \
        expected_assists: stats.expected_assists, expected_goal_involvements: stats.expected_goal_involvements, \
        expected_goals_conceded: stats.expected_goals_conceded, total_points: stats.total_points}"
    player_gw_stats = jmespath.search(
        search_pattern,
        data,
    )
    player_gw_stats_df = pl.DataFrame(player_gw_stats)
    player_gw_stats_df = player_gw_stats_df.rename({"total_points": "gameweek_points"})
    player_gw_stats_df = player_gw_stats_df.with_columns(
        pl.lit(gameweek).alias("gameweek")
    )
    if cols:
        return player_gw_stats_df.select(cols)
    return player_gw_stats_df


@cache
def get_player_data() -> pl.DataFrame:
    data = get("https://fantasy.premierleague.com/api/bootstrap-static/")

    player_data = jmespath.search(
        "elements[*].{player_id: id, team_id: team, position_id: element_type, name: web_name, first_name: first_name, second_name: second_name, cost_times_ten: now_cost}",
        data,
    )
    team_data = jmespath.search(
        "teams[*].{team_id: id, team: name, team_short_name: short_name}", data
    )
    position_data = jmespath.search(
        "element_types[*].{position_id: id, position: singular_name_short}", data
    )
    player_data_df = pl.DataFrame(player_data)
    position_data_df = pl.DataFrame(position_data)
    team_data_df = pl.DataFrame(team_data)

    df = player_data_df.join(position_data_df, on="position_id")
    return df.join(team_data_df, on="team_id")


@cache
def get_fixtures() -> list[dict]:
    data = cast(list[dict], get("https://fantasy.premierleague.com/api/fixtures/"))
    return data


def load_player_gameweek_data(gameweek: int) -> pl.DataFrame:
    gw_stats = get_player_gameweek_stats(gameweek)
    raw_player_data = get_player_data()
    player_data = raw_player_data.join(gw_stats, on="player_id")
    return player_data.with_columns((pl.col("cost_times_ten") / 10).alias("cost"))
