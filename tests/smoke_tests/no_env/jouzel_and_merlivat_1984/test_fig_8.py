"""
tests for fig. 8 from [Jouzel & Merlivat 1984 (J. Geophys. Res. Atmos. 89)](https://doi.org/10.1029/JD089iD07p11749)
"""  # pylint: disable=line-too-long

from pathlib import Path
import pytest
import numpy as np
from open_atmos_jupyter_utils import notebook_vars
from PySDM.physics.constants import T0
from PySDM_examples import Jouzel_and_Merlivat_1984

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Jouzel_and_Merlivat_1984.__file__).parent / "fig_8_9.ipynb",
        plot=PLOT,
    )


class TestFig8:
    @staticmethod
    @pytest.mark.parametrize(
        "if_effective, temp_C, value",
        (
            (False, 0, 1),
            (False, -10, 1.1),
            (False, -20, 1.22),
            (False, -30, 1.34),
            (False, -45, 1.54),
            (True, 0, 1),
            (True, -10, 1.08),
            (True, -20, 1.18),
            (True, -30, 1.30),
            (True, -45, 1.52),
        ),
    )
    def test_fig8_values_against_the_paper(
        notebook_variables, if_effective, temp_C, value
    ):
        # arrange
        temperature_C = notebook_variables["T_0_50"] - T0
        if if_effective:
            Si = notebook_variables["eff_saturation_wrt_ice"]
        else:
            Si = notebook_variables["saturation_wrt_ice"]

        # act
        sut = Si[np.argmin(np.abs(temperature_C - temp_C))]

        # assert
        np.testing.assert_approx_equal(sut, value, significant=1)

    @staticmethod
    @pytest.mark.parametrize("temperature_C, expected", ((-20, 0.05),))
    def test_fig_8_max_difference(notebook_variables, temperature_C, expected):
        """
        Test maximum difference between saturation and effective saturation over ice.
        Based on comment the on eq. (13)
        in [Jouzel & Merlivat 1984 (J. Geophys. Res. Atmos. 89)](https://doi.org/10.1029/JD089iD07p11749)
        """  # pylint: disable=line-too-long
        # arrange
        saturation_difference = notebook_variables["saturation_difference"]
        temp = notebook_variables["T_0_50"]
        temp_C = temp - T0
        temp_C_tolerance = 0.01

        # act
        max_difference = np.max(saturation_difference)
        temp_C_range_max = temp_C[
            np.isclose(saturation_difference, max_difference, rtol=0.1)
        ]

        # assert
        assert max_difference <= expected
        assert (
            np.min(temp_C_range_max) - temp_C_tolerance
            <= temperature_C
            <= np.max(temp_C_range_max) + temp_C_tolerance
        )

    @staticmethod
    def test_alpha_less_then_eff_alpha(notebook_variables):
        # arrange
        alpha_eff = notebook_variables["eff_alpha_kinetic"]
        alpha = notebook_variables["alpha_kinetic"]

        # act
        sut = alpha_eff

        # assert
        np.testing.assert_array_less(sut, 1)
        np.testing.assert_array_less(alpha, alpha_eff)
