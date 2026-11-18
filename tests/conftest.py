import pytest

from iad.scoring import Scorer


@pytest.fixture(scope="session")
def scorer() -> Scorer:
    return Scorer()
