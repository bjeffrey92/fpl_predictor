import argparse
from pathlib import Path

import polars as pl

from fpl_predictor.player_stats import get_player_data
from fpl_predictor.squad_selection import squad_selection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
        help="The gameweek for which to select the squad",
    )
    parser.add_argument(
        "--prediction-method",
        type=str,
        default="xgboost",
        help="The method to use for predicting gameweek scores",
    )
    parser.add_argument(
        "--n-free-transfers",
        type=int,
        default=1,
        help="The number of free transfers to make",
    )
    parser.add_argument(
        "--n-prediction-weeks",
        type=int,
        required=False,
        help="Only used with the xgboost prediction method",
    )
    parser.add_argument(
        "--current-squad",
        type=Path,
        required=False,
        help="The number of transfers to make",
    )
    args = parser.parse_args()
    if args.prediction_method == "xgboost" and not args.n_prediction_weeks:
        parser.error(
            "The --n_prediction_weeks argument is required when using the xgboost prediction method"
        )
    return args


def _read_in_current_squad(current_squad_path: Path) -> pl.DataFrame:
    return pl.read_csv(current_squad_path)


def main() -> None:
    args = _parse_args()
    current_squad = (
        _read_in_current_squad(args.current_squad) if args.current_squad else None
    )
    squad, expected_points = squad_selection.select_squad(
        args.gameweek,
        current_squad=current_squad,
        n_free_transfers=args.n_free_transfers,
        prediction_method=args.prediction_method,
        n_prediction_weeks=args.n_prediction_weeks,
    )
    print(f"{expected_points=}")
    player_data = get_player_data()
    df = squad.join(
        player_data.select("player_id", "name", "team", "position"), on="player_id"
    )
    df.write_csv("selected_squad.csv")
