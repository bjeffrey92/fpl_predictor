from unittest import mock

from fpl_predictor import utils


def test_get() -> None:
    url = "https://example.com"
    with mock.patch.object(utils.requests, "get") as mock_get:
        output = utils.get(url)
        mock_get.assert_called_once_with(url)
        mock_get.return_value.raise_for_status.assert_called_once()
        assert output == mock_get.return_value.json.return_value
