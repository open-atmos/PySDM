""" """

from pathlib import Path
import pytest
import numpy as np
from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Jouzel_and_Merlivat_1984

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Jouzel_and_Merlivat_1984.__file__).parent
        / "Si_vs_effective_Si.ipynb",
        plot=PLOT,
    )


class TestSiVsEffectiveSi:
    @staticmethod
    def test_alpha_kinetic_of_temperature_values(notebook_variables):
        # arrange
        temperature = notebook_variables["temperature"]
        alpha_eff = notebook_variables["eff_alpha_kinetic"]
        alpha = notebook_variables["alpha_kinetic"]
        temp_below_0C = temperature <= 273.15

        # act
        sut = alpha_eff[temp_below_0C]

        # assert
        np.testing.assert_array_less(sut, 1)
        np.testing.assert_array_less(alpha[temp_below_0C], alpha_eff[temp_below_0C])

    @staticmethod
    @pytest.mark.parametrize("temperature_C, expected", ((-20, 0.05),))
    def test_max_difference(notebook_variables, temperature_C, expected):
        # arrange
        diff = notebook_variables["diff"]
        temp = notebook_variables["temperature"]
        temp_C = temp - 273.15
        temp_C_tolerance = 1

        # act
        diff_max = np.max(diff)
        temp_C_range_max = temp_C[np.isclose(diff, diff_max, rtol=0.1)]

        # assert
        assert diff_max <= expected
        assert (
            np.min(temp_C_range_max) - temp_C_tolerance
            <= temperature_C
            <= np.max(temp_C_range_max) + temp_C_tolerance
        )
