from pathlib import Path

import pytest
from PySDM_examples import Graf_2019
from PySDM_examples.utils import notebook_vars

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="vars")
def vars_fixture():
    return notebook_vars(
        file=Path(Graf_2019.__file__).parent / "figure_4.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "level_name, expected_height", (("CB", 1 * si.km), ("0C", 2.24 * si.km))
)
def test_fig_4(vars, level_name, expected_height):
    # arrange
    tolerance = 50 * si.m

    # act
    actual = vars["levels"][level_name] + vars["alt_initial"]

    # assert
    assert abs(expected_height - actual) < tolerance
