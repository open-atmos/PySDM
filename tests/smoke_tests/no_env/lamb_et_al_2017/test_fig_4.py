"""
regression tests checking values from the plots in Fig 4
"""

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils.notebook_vars import notebook_vars
from PySDM_examples import Lamb_et_al_2017

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Lamb_et_al_2017.__file__).parent / "fig_4.ipynb", plot=PLOT
    )


class TestFig4:
    @staticmethod
    @pytest.mark.parametrize(
        "T, alpha_i_2H, paper",
        (
            (180, 1.50, "MerlivatAndNief1967"),
            (220, 1.27, "MerlivatAndNief1967"),
            (273, 1.13, "MerlivatAndNief1967"),
            (193, 1.60, "EllehojEtAl2013"),
            (220, 1.35, "EllehojEtAl2013"),
            (273, 1.13, "EllehojEtAl2013"),
            (180, 1.44, "LambEtAl2017"),
            (220, 1.25, "LambEtAl2017"),
            (273, 1.13, "LambEtAl2017"),
        ),
    )
    def test_values_match(notebook_local_variables, T, alpha_i_2H, paper):
        plot_x = notebook_local_variables["T"]
        plot_y = notebook_local_variables["alphas"][paper]
        eps = 5e-1
        index = np.where(abs(plot_x - T) < eps)
        np.testing.assert_approx_equal(
            actual=plot_y[index], desired=alpha_i_2H, significant=3
        )

    @staticmethod
    def test_monotonic(notebook_local_variables):
        assert (np.diff(notebook_local_variables["T"]) > 0).all()
        for paper in notebook_local_variables["PAPERS"]:
            assert (np.diff(notebook_local_variables["alphas"][paper]) < 0).all()
