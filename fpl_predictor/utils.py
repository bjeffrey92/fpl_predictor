import requests


def get(url: str) -> dict | list:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
