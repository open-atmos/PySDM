"""
regression tests checking values against paper Fig 1 from [Pruppacher and Rasmussen
1979](https://doi.org/10.1175/1520-0469%281979%29036%3C1255:AWTIOT%3E2.0.CO;2)
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Pruppacher_and_Rasmussen_1979

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Pruppacher_and_Rasmussen_1979.__file__).parent / "fig_1.ipynb", plot=PLOT
    )


class TestFig1:
    @staticmethod
    @pytest.mark.parametrize(
        "sqrt_re_times_cbrt_sc, vent_coeff",
        (
            (3, 1.7),
            (20, 7),
            (44, 14),
        ),
    )
    def test_values_match(notebook_local_variables, sqrt_re_times_cbrt_sc, vent_coeff):
        plot_x = notebook_local_variables["sqrt_re_times_cbrt_sc"]
        plot_y = notebook_local_variables["vent_coef"]
        eps = 0.1
        ((index,),) = np.where(abs(plot_x - sqrt_re_times_cbrt_sc) < eps)
        np.testing.assert_approx_equal(
            actual=plot_y[index], desired=vent_coeff, significant=2
        )

    @staticmethod
    def test_monotonic_x(notebook_local_variables):
        plot_x = notebook_local_variables["sqrt_re_times_cbrt_sc"]
        assert (np.diff(plot_x) > 0).all()

    @staticmethod
    def test_monotonic_y(notebook_local_variables):
        plot_y = notebook_local_variables["vent_coef"]
        assert (np.diff(plot_y) > 0).all()
