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
    @pytest.mark.parametrize("fractionation_type", ("eql", "kin"))
    @pytest.mark.parametrize("isotope", ("2H", "18O"))
    def test_fig_4c(variables, fractionation_type, isotope):
        # arrange
        below_cloud = (
            variables["z"] < variables["levels"]["CB"] + variables["alt_initial"]
        )
        deltas = variables["deltas_Rayleigh"]

        # act
        actual = deltas[isotope][fractionation_type][below_cloud]
        desired = variables[f"delta0_{isotope}"]

        # assert
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=2e-5,
        )

    @staticmethod
    def test_fig_4d(variables):
        # arrange
        desired = 10

        # act
        actual = variables["d_excess"] / PER_MILLE

        # assert
        np.testing.assert_allclose(actual, desired, rtol=4e-2)
