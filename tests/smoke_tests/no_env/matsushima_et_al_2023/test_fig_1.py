# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from pathlib import Path
from scipy.interpolate import interp1d
import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Matsushima_et_al_2023

from PySDM.physics import si


PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Matsushima_et_al_2023.__file__).parent / "figure_1.ipynb", plot=PLOT
    )


class TestFig1:
    @staticmethod
    @pytest.mark.parametrize(
        "alpha, x, y",
        (
            pytest.param(
                0,
                11 * si.nm,
                1.6e3,
                marks=pytest.mark.xfail(strict=True, raises=AssertionError),
            ),
            (0, 20 * si.nm, 1.6e3),
            (0, 100 * si.nm, 1.6e3),
            (0, 500 * si.nm, 1.6e3),
            pytest.param(
                0,
                1000 * si.nm,
                1.6e3,
                marks=pytest.mark.xfail(strict=True, raises=AssertionError),
            ),
            (0.5, 11 * si.nm, 230),
            (0.5, 20 * si.nm, 2.1e3),
            (0.5, 100 * si.nm, 730),
            (0.5, 500 * si.nm, 74),
            (0.5, 1000 * si.nm, 2.1),
            (1, 10.1 * si.nm, 0.84),
            (1, 20 * si.nm, 3.6e3),
            (1, 100 * si.nm, 850),
            (1, 500 * si.nm, 76),
            (1, 1000 * si.nm, 2.1),
        ),
    )
    def test_panel_b(notebook_local_variables, alpha, x, y):
        f = interp1d(
            notebook_local_variables["xas"][alpha],
            notebook_local_variables["yas"][alpha],
        )
        np.testing.assert_approx_equal(actual=f(x), desired=y, significant=2)

    @staticmethod
    @pytest.mark.parametrize(
        "alpha, first_value, last_value",
        (
            (0.0, 120, np.nan),
            (0.1, 25, 4.3e3),
            (0.2, 1.1, 5.7e3),
            (0.4, 0.011, 7.9e3),
            (0.5, 0.0018, 9e3),
            (0.8, 2e-5, 1.2e4),
            (1.0, 1.4e-6, 1.4e4),
        ),
    )
    def test_panel_c(notebook_local_variables, alpha, first_value, last_value):
        plot_data = np.sort(notebook_local_variables["yas"][alpha])
        np.testing.assert_approx_equal(
            actual=plot_data[0], desired=first_value, significant=2
        )
        np.testing.assert_approx_equal(
            actual=plot_data[-1], desired=last_value, significant=2
        )
