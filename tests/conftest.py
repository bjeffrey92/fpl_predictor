from importlib.resources import files
from pathlib import Path
from typing import cast

import pytest

from tests import fixtures


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return cast(Path, files(fixtures))
