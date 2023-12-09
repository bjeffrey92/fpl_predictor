from unittest.mock import patch

import polars as pl
from polars.datatypes import Float64, Int32, Int64, Utf8

from fpl_predictor.player_stats import (
    get_player_data,
    get_player_gameweek_stats,
    load_player_gameweek_data,
)


def test_player_gameweek_stats() -> None:
    with patch("fpl_predictor.player_stats.get") as mock_get:
        mock_get.return_value = {
            "elements": [
                {"id": 1, "stats": {"total_points": 2}},
                {"id": 2, "stats": {"total_points": 3}},
            ]
        }
        player_gw_stats = get_player_gameweek_stats(1)
        assert player_gw_stats.columns == ["player_id", "gameweek_points", "gameweek"]
        assert player_gw_stats.dtypes == [Int64, Int64, Int32]
        assert player_gw_stats.shape == (2, 3)
        assert player_gw_stats["player_id"].to_list() == [1, 2]
        assert player_gw_stats["gameweek_points"].to_list() == [2, 3]


def test_get_player_data() -> None:
    with patch("fpl_predictor.player_stats.get") as mock_get:
        mock_get.return_value = {
            "elements": [
                {
                    "id": 1,
                    "team": 1,
                    "element_type": 1,
                    "web_name": "Player 1",
                    "now_cost": 50,
                },
                {
                    "id": 2,
                    "team": 2,
                    "element_type": 2,
                    "web_name": "Player 2",
                    "now_cost": 60,
                },
            ],
            "element_types": [
                {"id": 1, "singular_name_short": "GKP"},
                {"id": 2, "singular_name_short": "DEF"},
            ],
        }
        player_data = get_player_data()
        assert player_data.columns == [
            "player_id",
            "team_id",
            "position_id",
            "name",
            "cost_times_ten",
            "position",
        ]
        assert player_data.dtypes == [Int64, Int64, Int64, Utf8, Int64, Utf8]
        assert player_data.shape == (2, 6)
        assert player_data["player_id"].to_list() == [1, 2]
        assert player_data["team_id"].to_list() == [1, 2]
        assert player_data["position_id"].to_list() == [1, 2]
        assert player_data["name"].to_list() == ["Player 1", "Player 2"]
        assert player_data["cost_times_ten"].to_list() == [50, 60]
        assert player_data["position"].to_list() == ["GKP", "DEF"]


def test_load_player_gameweek_data() -> None:
    with patch(
        "fpl_predictor.player_stats.get_player_gameweek_stats"
    ) as mock_get_player_gameweek_stats, patch(
        "fpl_predictor.player_stats.get_player_data"
    ) as mock_get_player_data:
        mock_get_player_gameweek_stats.return_value = pl.DataFrame(
            {
                "player_id": [1, 2],
                "gameweek_points": [2, 3],
                "gameweek": [1, 1],
            }
        )
        mock_get_player_data.return_value = pl.DataFrame(
            {
                "player_id": [1, 2],
                "team_id": [1, 2],
                "position_id": [1, 2],
                "name": ["Player 1", "Player 2"],
                "cost_times_ten": [50, 60],
                "position": ["GKP", "DEF"],
            }
        )
        player_data = load_player_gameweek_data(1)
        assert player_data.columns == [
            "player_id",
            "team_id",
            "position_id",
            "name",
            "cost_times_ten",
            "position",
            "gameweek_points",
            "gameweek",
            "cost",
        ]
        assert player_data.dtypes == [
            Int64,
            Int64,
            Int64,
            Utf8,
            Int64,
            Utf8,
            Int64,
            Int64,
            Float64,
        ]
        assert player_data.shape == (2, 9)
        assert player_data["player_id"].to_list() == [1, 2]
        assert player_data["team_id"].to_list() == [1, 2]
        assert player_data["position_id"].to_list() == [1, 2]
        assert player_data["name"].to_list() == ["Player 1", "Player 2"]
        assert player_data["cost_times_ten"].to_list() == [50, 60]
        assert player_data["position"].to_list() == ["GKP", "DEF"]
        assert player_data["gameweek_points"].to_list() == [2, 3]
