"""
regression tests checking values plotted in Fig 2
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Gedzelman_and_Arnold_1994

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        Path(Gedzelman_and_Arnold_1994.__file__).parent / "fig_2.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "x, y, var",
    (
        (0.99, 0.27, "eq_22"),
        (0.898, 0.62, "eq_22"),
        (0.875, 0.95, "eq_22"),
        (0.8875, 0, "eq_23"),
        (0.88, 0.32, "eq_23"),
        (0.85, 1, "eq_23"),
    ),
)
def test_fig_2(notebook_variables, x, y, var):  # TODO, fix variable names
    """given that the plot depends on a number of constants that are likely to cause
    discrepancies, the comparison is rough and effectively just a regression test
    (expected values roughly correspond to the paper plot, but are based on PySDM output)
    """
    plot_x = notebook_variables["fig2_x"]
    plot_y = notebook_variables["fig2_y"][var]
    eps = (plot_x[1] - plot_x[0]) / 2
    index = np.where(abs(plot_x - x) < eps)
    np.testing.assert_allclose(actual=plot_y[index], desired=y, atol=0.01)


def test_isotope_ratio_change(notebook_variables):
    # arange
    saturation_for_zero_dR_condition = notebook_variables["S_eq"]
    rel_diff_vap, rel_diff_liq = (
        notebook_variables["rel_diff_vap"],
        notebook_variables["rel_diff_liq"],
    )

    mesh_grid = np.meshgrid(
        saturation_for_zero_dR_condition["liquid"], notebook_variables["rh_percent"]
    )
    growth_liq = np.where(
        mesh_grid[1] > saturation_for_zero_dR_condition["liquid"], 1, -1
    )
    mesh_grid = np.meshgrid(
        saturation_for_zero_dR_condition["vapour"], notebook_variables["rh_percent"]
    )
    growth_vap = np.where(
        mesh_grid[1] > saturation_for_zero_dR_condition["vapour"], 1, -1
    )

    # act

    # assert
    np.testing.assert_allclose(actual=growth_liq, desired=True)
    np.testing.assert_allclose(actual=growth_vap, desired=True)
