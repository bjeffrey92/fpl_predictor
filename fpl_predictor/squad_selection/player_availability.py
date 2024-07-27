import re

import polars as pl
import requests
from bs4 import BeautifulSoup

URL = "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries/"


def _parse_table_row(table_row) -> list[str]:
    """
    Returns a list of the player's name and the team they play for
    """
    return [j.text for i, j in enumerate(table_row.find_all("td")) if i in [0, 1]]


def _extract_player_first_name(full_name: str) -> str:
    match = re.search(r"\((.*?)\)", full_name)
    if match:
        return match.group(1)
    else:
        raise RuntimeError(f"Could not extract player first name from {full_name=}")


def _extract_player_second_name(full_name: str) -> str:
    match = re.search(r"\s(.*?)\s\(", full_name)
    if match:
        return match.group(1).strip()
    else:
        raise RuntimeError(f"Could not extract player last name from {full_name=}")


def get_unavailable_players():
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, features="html.parser")
    table = soup.find("table")
    players_list = [_parse_table_row(table_row) for table_row in table.find_all("tr")][
        1:  # Skip the header row
    ]
    df = pl.DataFrame(players_list, orient="row", schema=["player_full_name", "team"])
    return df.with_columns(
        first_name=pl.col("player_full_name").map_elements(
            _extract_player_first_name, return_dtype=pl.String
        ),
        second_name=pl.col("player_full_name").map_elements(
            _extract_player_second_name, return_dtype=pl.String
        ),
    ).select("first_name", "second_name", "team")
