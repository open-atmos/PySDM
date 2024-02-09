""" tests ensuring values on plots match those in the paper """

from pathlib import Path

import pytest

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Graf_et_al_2019

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Graf_et_al_2019.__file__).parent / "figure_4.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "level_name, expected_height", (("CB", 1 * si.km), ("0C", 2.24 * si.km))
)
def test_fig_4(variables, level_name, expected_height):
    # arrange
    tolerance = 50 * si.m

    # act
    actual = variables["levels"][level_name] + variables["alt_initial"]

    # assert
    assert abs(expected_height - actual) < tolerance
