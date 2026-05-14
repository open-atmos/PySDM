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
    """notebook variables fixture"""
    return notebook_vars(
        Path(Gedzelman_and_Arnold_1994.__file__).parent / "fig_2.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "x, expected_y, phase",
    (
        (1.0, 0.30, "liquid"),
        (0.93, 0.5, "liquid"),
        (0.8776, 1.0, "liquid"),
        (0.9, 0.0, "vapour"),
        (0.8776, 1.0, "vapour"),
    ),
)
def test_fig_2(notebook_variables, x, expected_y, phase):
    """
    Verify values plotted in Figure 2.

    The test selects the data point whose x-value is closest to ``x`` and
    checks that the corresponding y-value matches ``expected_y``.
    The (x, expected_y) pairs are approximated from Fig 2 in paper.
    """

    # arrange
    plot_x = notebook_variables["plots"][phase]["x"]
    plot_y = notebook_variables["plots"][phase]["y"]
    plot_x_eps = (plot_x[1] - plot_x[0]) / 2
    plot_y_eps = np.max(abs(np.diff(plot_y))) / 2

    # act
    idx = np.where(abs(plot_x - x) < plot_x_eps)
    sut = max(0, plot_y[idx])

    # assert
    np.testing.assert_allclose(
        actual=sut,
        desired=expected_y,
        rtol=plot_y_eps,
    )


@pytest.mark.parametrize(
    "phase, condition, rtol, eps",
    (
        ("vapour", 0.0, 0.1, 1e-3),
        ("liquid", 1.0, 0.1, 1e-2),
    ),
)
def test_dR_zero_condition(notebook_variables, phase, condition, rtol, eps):
    """Test values plotted with color in Fig 1.
    Points (x, y) for which z equals condition should match theoretical lines."""
    # arrange
    cmn = notebook_variables["CMN_FOR_TEST"]
    iso_ratio_v = notebook_variables["ISO_RATIO_V"]

    Y = notebook_variables["YY"]
    X = notebook_variables["XX"]

    pcm_data = notebook_variables["cases"][phase]["pcolormesh"].get_array()
    within = (
        (condition - eps < pcm_data)
        & (pcm_data < condition + eps)
        & (X > notebook_variables["X_eq"])
    )
    assert np.sum(within) > 0

    # act
    iso_ratio_r = X[within] * cmn.params.vsmow
    expected_y = cmn.f.isotope_ratio_evolution.saturation_for_zero_dR_condition(
        iso_ratio_x=iso_ratio_r if phase == "liquid" else iso_ratio_v,
        diff_rat_light_to_heavy=(cmn.params.f_ratio / cmn.params.D_ratio),
        b=cmn.params.b,
        alpha_w=cmn.params.alpha_w,
        iso_ratio_r=iso_ratio_r,
        iso_ratio_v=iso_ratio_v,
    )

    # assert
    np.testing.assert_allclose(Y[within], expected_y, rtol=rtol)
