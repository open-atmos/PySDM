"""
regression tests checking values plotted in Fig 2
"""

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot
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


@pytest.mark.parametrize(
    "phase, eps",
    (
        ("liquid", 1e-10),
        ("liquid", 1e-5),
        ("liquid", 1e-1),
        ("vapour", 1e-10),
        ("vapour", 1e-5),
        ("vapour", 1e-1),
    ),
)
def test_isotope_ratio_change(notebook_variables, phase, eps):
    # arange
    rh = notebook_variables["RH"]
    mR_liq = notebook_variables["molecular_R_liq"]
    rh_tile = np.tile(rh[::-1], (len(rh), 1)).T
    s_eq = notebook_variables["s_eq"]
    COMMONS = notebook_variables["COMMONS"]
    heavier_liq = mR_liq > COMMONS.iso_ratio_liq_eq

    rel_diff_vap, rel_diff_liq = (
        notebook_variables["rel_diff_vap"],
        notebook_variables["rel_diff_liq"],
    )

    above_liq_eq_line = heavier_liq * (rh_tile > s_eq["liquid"])
    above_vap_eq_line = heavier_liq * (rh_tile > s_eq["vapour"])
    # act
    sut = {
        "liquid": np.where(above_liq_eq_line, -1, 1) * rel_diff_liq[::-1, :],
        "vapour": np.where(above_vap_eq_line, 1, -1) * rel_diff_vap[::-1, :],
    }
    # assert
    np.testing.assert_array_less(-sut["liquid"][~np.isnan(sut["liquid"])], eps)
    np.testing.assert_array_less(-sut["vapour"][~np.isnan(sut["vapour"])], eps)
