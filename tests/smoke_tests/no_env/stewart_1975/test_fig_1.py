"""
test checking values on Fig 1
"""

from pathlib import Path
import pytest
import numpy as np

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Stewart_1975

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    """returns variables from the notebook Stewart_1975/fig_1.ipynb"""
    return notebook_vars(
        file=Path(Stewart_1975.__file__).parent / "fig_1.ipynb", plot=PLOT
    )


def test_fig_ventilation_coefficient(notebook_variables):
    """test if ventilation coefficients (f) have the same line slope"""
    # Arrange
    vent_coeff_K_G = notebook_variables["plot_K_G_coeff"][0].get_data()[1]
    vent_coeff_B_P = notebook_variables["plot_B_P_coeff"][0].get_data()[1]

    # Act
    eps = vent_coeff_K_G / vent_coeff_B_P

    # Assert
    np.testing.assert_allclose(actual=eps, desired=1, atol=0.4)


@pytest.mark.parametrize("paper", ("Kinzer & Gunn", "Beard & Pruppacher"))
def test_fig_1(notebook_variables, paper):
    """test coefficient factors (F) from Kinzer & Gunn and Beard & Pruppacher"""
    # Arrange
    radii_mm, ventilation_factor = notebook_variables["plot_factor"][paper][
        0
    ].get_data()
    idx_to_check = radii_mm >= 0.25
    ventilation_factor_to_check = ventilation_factor[idx_to_check]
    vent_factor_high = 1.4
    vent_factor_low = 0.8

    # Act
    eps = (vent_factor_high - vent_factor_low) / 2
    avg_value = vent_factor_high - eps

    # Assert
    np.testing.assert_allclose(
        actual=ventilation_factor_to_check, desired=avg_value, atol=eps
    )
