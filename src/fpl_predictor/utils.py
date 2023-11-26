import requests  # type: ignore[import-untyped]


def get(url: str) -> dict | str:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
