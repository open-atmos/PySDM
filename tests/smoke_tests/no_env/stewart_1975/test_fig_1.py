"""
test checking values
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


# @pytest.mark.parametrize(
#     "x, y, paper",
#     (
#         (1.0e-04, 0),
#         (3.0e-04, 0),
#         (5.0e-04, 0),
#         (9.0e-04, 0),
#         (1.3e-03, 0),
#         (1.7e-03, 0)
#     )
# )
# def test_fig_1(
#         notebook_variables, x, y, paper
# ):
#     np.testing.assert_allclose(
#         actual=notebook_variables,
#         desired=1,
#         rtol=1,
#     )


def test_fig_ventilation_coefficient(notebook_variables):
    """ventilation coefficients should have same  the same line scope"""
    #
    ventilation_coefficient_K_G = notebook_variables["plot_K_G_coeff"][0].get_data()[1]
    ventilation_coefficient_B_P = notebook_variables["plot_B_P_coeff"][0].get_data()[1]

    # act
    eps = ventilation_coefficient_K_G / ventilation_coefficient_B_P

    # assert
    np.testing.assert_allclose(actual=eps, desired=1, atol=0.4)
