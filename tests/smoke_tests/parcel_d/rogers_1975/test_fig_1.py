"""test values on the plot against paper"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Rogers_1975
from PySDM.physics.constants import PER_CENT


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Rogers_1975.__file__).parent / "fig_1.ipynb", plot=False
    )


class TestFig1:
    @staticmethod
    def test_fig1_saturation_peak_against_paper(variables):
        # arrange
        SS = variables["solution"].S - 1
        time = variables["tsteps"]
        expected_peak_time = 7
        expected_peak_value = 0.97 * PER_CENT

        # act
        peak_value = np.max(SS)
        peak_time = time[np.argmax(SS)]

        # assert
        np.testing.assert_allclose(
            actual=peak_value.magnitude, desired=expected_peak_value, rtol=1e-3
        )
        np.testing.assert_allclose(
            actual=peak_time.magnitude, desired=expected_peak_time, atol=0.5
        )

    @staticmethod
    def test_fig1_radius_scope(variables):
        """
        Check if for first 2.5 seconds slope is smaller than after this time.
        It represents slower droplets growth with small supersaturation.
        """
        # arrange
        radius = variables["solution"].r
        time_less_than = np.sum(variables["tsteps"].to_base_units().magnitude <= 2.5)
        dr_before = np.diff(radius[:time_less_than]).to_base_units().magnitude
        dr_after = np.diff(radius[time_less_than:]).to_base_units().magnitude

        # act
        sut = np.mean(dr_before) / np.mean(dr_after)

        # assert
        assert sut < 1
        assert sut > 0
        assert (dr_before > 0).all()
