"""tests ensuring values on plots match those in the paper"""

from pathlib import Path

import pytest
import numpy as np

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Graf_et_al_2019

from PySDM.physics import si
from PySDM.physics.constants import PER_MILLE

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Graf_et_al_2019.__file__).parent / "figure_4.ipynb", plot=PLOT
    )


class TestFig4:
    @staticmethod
    @pytest.mark.parametrize(
        "level_name, expected_height", (("CB", 1 * si.km), ("0C", 2.24 * si.km))
    )
    def test_fig_4(variables, level_name, expected_height):
        # arrange
        tolerance = 50 * si.m

        # act
        actual = variables["levels"][level_name] + variables["alt_initial"]

        # assert
        assert abs(expected_height - actual) < tolerance

    @staticmethod
    @pytest.mark.parametrize("R_name", ("R_Rayleigh_eql", "R_Rayleigh_kin"))
    def test_fig_4c(variables, R_name):
        # arrange
        below_cloud = (
            variables["z"] < variables["levels"]["CB"] + variables["alt_initial"]
        )

        # act
        actual = variables[R_name][below_cloud]
        desired = variables["R0_2H"]

        # assert
        np.testing.assert_array_almost_equal(
            actual,
            desired,
        )

    @staticmethod
    def test_fig_4d(variables):
        # arrange
        desired = 10

        # act
        actual = variables["d_excess"] / PER_MILLE

        # assert
        np.testing.assert_allclose(actual, desired, rtol=4e-2)
