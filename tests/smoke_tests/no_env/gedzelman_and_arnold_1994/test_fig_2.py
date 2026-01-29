"""
regression tests checking values plotted in Fig 2
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Gedzelman_and_Arnold_1994

from PySDM.physics.constants import PER_CENT

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    """notebook variables fixture"""
    return notebook_vars(
        Path(Gedzelman_and_Arnold_1994.__file__).parent / "fig_2.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "x, y, phase",
    (
        (0.99, 0.27, "liquid"),
        (0.898, 0.62, "liquid"),
        (0.875, 0.95, "liquid"),
        (0.8875, 0, "vapour"),
        (0.88, 0.32, "vapour"),
        (0.85, 1, "vapour"),
    ),
)
def test_fig_2(notebook_variables, x, y, phase):
    """given that the plot depends on a number of constants that are likely to cause
    discrepancies, the comparison is rough and effectively just a regression test
    (expected values roughly correspond to the paper plot, but are based on PySDM output)
    """
    plot_x = notebook_variables["molecular_R_liq"]
    plot_y = notebook_variables["S_eq"][phase]
    eps = (plot_x[1] - plot_x[0]) / 2
    index = np.where(abs(plot_x - x) < eps)
    np.testing.assert_allclose(actual=plot_y[index], desired=y, atol=0.01)


@pytest.mark.parametrize(
    "phase, eps_percent",
    (
        pytest.param("liquid", 1.0, marks=pytest.mark.xfail(strict=True)),
        ("liquid", 1.1),
        ("liquid", 1.2),
        pytest.param("vapour", 0.05, marks=pytest.mark.xfail(strict=True)),
        ("vapour", 0.09),
        ("vapour", 0.1),
    ),
)
def test_isotope_ratio_change(notebook_variables, phase, eps_percent):
    """test sign of the changes above theoretical equilibrium line"""
    # arrange
    cmn = notebook_variables["COMMONS"]
    rh = notebook_variables["RH"]
    s_eq = notebook_variables["S_eq"][phase]
    x_axis_size1 = notebook_variables["x"]
    x_axis_size2 = notebook_variables["molecular_R_liq"]

    eq_line_x_idx = np.abs(x_axis_size1[:, None] - x_axis_size2[None, :]).argmin(axis=0)
    s_eq_iso = s_eq[eq_line_x_idx]

    above_eq_line = (rh[:, None] > s_eq_iso[None, :]) & (
        x_axis_size2[None, :] > cmn.ratios.iso_ratio_liq_eq
    )

    # act
    rel_diff = notebook_variables[f"rel_diff_{phase[:3]}"][::-1, :]
    expected_sign = np.where(above_eq_line, -1.0, 1.0)
    if phase == "vapour":
        expected_sign *= -1
    sut = expected_sign * rel_diff * PER_CENT

    # assert
    np.testing.assert_array_less(-sut[~np.isnan(sut)], eps_percent)
