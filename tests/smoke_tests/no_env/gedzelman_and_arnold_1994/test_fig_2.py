"""
regression tests checking values plotted in Fig 2
"""

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils.notebook_vars import notebook_vars
from PySDM_examples import Gedzelman_and_Arnold_1994

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Gedzelman_and_Arnold_1994.__file__).parent / "fig_2.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "x, y, var",
    (
        (0.99, 0.3, "eq_22"),
        (0.898, 2 / 3, "eq_22"),
        (0.875, 1, "eq_22"),
        (0.8925, 0, "eq_23"),
        (0.88, 2 / 3, "eq_23"),
        (0.875, 1, "eq_23"),
    ),
)
def test_fig_2(notebook_local_variables, x, y, var):
    """given that the plot depends on a number of constants that are likely to cause
    discrepancies, the comparison is rough and effectively just a regression test
    (expected values roughly correspond to the paper plot, but are based on PySDM output)
    """
    plot_x = notebook_local_variables["x"]
    plot_y = notebook_local_variables[f"y_{var}"]
    eps = (plot_x[1] - plot_x[0]) / 2
    index = np.where(abs(plot_x - x) < eps)
    np.testing.assert_allclose(actual=plot_y[index], desired=y, atol=0.01)
