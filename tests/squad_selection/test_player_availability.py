from typing import Callable
from unittest import mock

import pytest

from fpl_predictor.squad_selection import player_availability


def test_parse_table_row() -> None:
    mock_table_row = mock.Mock()
    mock_table_row.find_all.return_value = [
        mock.Mock(text="td_1"),
        mock.Mock(text="td_2"),
        mock.Mock(text="td_3"),
    ]
    response = player_availability._parse_table_row(mock_table_row)
    assert response == ["td_1", "td_2"]


def _extract_player_names_test(
    raise_on_error: bool,
    player_name: str,
    expected_output: str,
    extract_fn: Callable[[str], str],
) -> None:
    with mock.patch.object(player_availability, "RAISE_ON_ERROR", raise_on_error):
        if raise_on_error and expected_output == "":
            with pytest.raises(RuntimeError):
                extract_fn(player_name)
            return
        response = extract_fn(player_name)
        assert response == expected_output


@pytest.mark.parametrize(
    "player_name,expected_output",
    (
        ("  Tierney (Kieran)", "Kieran"),
        ("  Tierney Kieran", ""),
    ),
)
@pytest.mark.parametrize("raise_on_error", (True, False))
def test_extract_player_first_name(
    raise_on_error: bool, player_name: str, expected_output: str
) -> None:
    _extract_player_names_test(
        raise_on_error,
        player_name,
        expected_output,
        player_availability._extract_player_first_name,
    )


@pytest.mark.parametrize(
    "player_name,expected_output",
    (
        ("  Tierney (Kieran)", "Tierney"),
        ("  Tierney Kieran", ""),
    ),
)
@pytest.mark.parametrize("raise_on_error", (True, False))
def test_extract_player_second_name(
    raise_on_error: bool, player_name: str, expected_output: str
) -> None:
    _extract_player_names_test(
        raise_on_error,
        player_name,
        expected_output,
        player_availability._extract_player_second_name,
    )


def test_get_unavailable_players() -> None:
    with mock.patch.object(
        player_availability.requests, "get"
    ) as mock_get, mock.patch.object(
        player_availability.BeautifulSoup, "find"
    ) as mock_find, mock.patch.object(
        player_availability, "_parse_table_row"
    ), mock.patch.object(
        player_availability.pl, "DataFrame"
    ) as mock_df:
        mock_get.return_value.text = "html"
        mock_find.return_value.find_all.return_value = [
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
        ]
        player_availability.get_unavailable_players()
        mock_get.assert_called_once_with(player_availability.URL)
        mock_find.assert_called_once_with("table")
        mock_find.return_value.find_all.assert_called_once_with("tr")
        mock_df.assert_called_once_with(
            mock.ANY,
            orient="row",
            schema=["player_full_name", "team"],
        )
