""" """

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Gonfiantini_1986

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Gonfiantini_1986.__file__).parent / "fig_3_1.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "humidity",
    (0, 0.25, 0.5, 0.75, 0.95),
)
def test_slope_bottom_fig(notebook_local_variables, humidity):
    # arrange
    delta_18O = notebook_local_variables["plot_y"]["18O"][humidity]
    delta_2H = notebook_local_variables["plot_y"]["2H"][humidity]

    # act
    slope = np.mean(delta_2H[1:] - delta_2H[:-1]) / np.mean(
        delta_18O[1:] - delta_18O[:-1]
    )

    # assert
    np.testing.assert_equal(actual=slope < 8, desired=True)
