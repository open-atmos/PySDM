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
        pytest.param("liquid", 0.01, marks=pytest.mark.xfail(strict=True)),
        ("liquid", 0.02),
        ("liquid", 0.06),
        pytest.param("vapour", 0.05, marks=pytest.mark.xfail(strict=True)),
        ("vapour", 0.09),
        ("vapour", 0.1),
    ),
)
def test_isotope_ratio_change(notebook_variables, phase, eps_percent):
    """
    Check that the sign of isotope ratio change (dR) matches
    the theoretical saturation lines (dR = 0) from
    Gedzelman & Arnold (1994), within tolerance eps_percent.
    """
    cmn = notebook_variables["COMMONS"]
    rh = notebook_variables["RH"]
    x = notebook_variables["x"]
    s_eq = notebook_variables["S_eq"][phase]

    rel_diff = notebook_variables[f"rel_diff_{phase[:3]}"]
    molecular_R_liq = notebook_variables["molecular_R_liq"]

    # Map isotope ratios onto equilibrium curve
    idx = np.abs(x[:, None] - molecular_R_liq[None, :]).argmin(axis=0)
    s_eq_iso = s_eq[idx]  # (n_iso,)

    # Broadcast into RH × isotope space
    rh_2d = rh[:, None]  # (n_rh, 1)
    s_eq_2d = s_eq_iso[None, :]  # (1, n_iso)

    heavier_liq = molecular_R_liq > cmn.ratios.iso_ratio_liq_eq
    heavier_liq_2d = heavier_liq[None, :]
    above_eq_line = heavier_liq_2d & (rh_2d > s_eq_2d)

    if phase == "liquid":
        expected_sign = np.where(above_eq_line, -1.0, 1.0)
    else:  # vapour
        expected_sign = np.where(above_eq_line, 1.0, -1.0)
    sut = expected_sign * rel_diff * PER_CENT

    valid = ~np.isnan(sut)
    np.testing.assert_array_less(-sut[valid], eps_percent)
