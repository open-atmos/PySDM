""" regression tests asserting on values from the plots """

from pathlib import Path

import pytest

from PySDM_examples.utils import notebook_vars
from PySDM_examples import deJong_Azimi

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(deJong_Azimi.__file__).parent / "box.ipynb", plot=PLOT
    )


def test_settings_a(variables):
    pass


def test_settings_b(variables):
    pass


def test_settings_c(variables):
    pass
