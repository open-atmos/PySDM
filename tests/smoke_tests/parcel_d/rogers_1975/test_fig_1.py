"""test values on the plot against paper"""

import numpy as np
import pytest
from pathlib import Path

from numpy.ma.core import argmax

from PySDM.physics import si
from PySDM.physics.constants import PER_CENT
from PySDM_examples import Rogers_1975
from open_atmos_jupyter_utils import notebook_vars


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
        peak_value = max(SS)
        peak_time = time[argmax(SS)]

        # assert
        np.testing.assert_allclose(
            actual=peak_value, desired=expected_peak_value, rtol=1e-3
        )
        np.testing.assert_allclose(
            actual=peak_time, desired=expected_peak_time, atol=0.5
        )


@staticmethod
def test_fig1_radius_scope(variables):
    """
    Check if for first 2.5 seconds slope is smaller than after this time.
    It represents slower droplets growth with small supersaturation.
    """
    # arrange
    radius = variables["solution"].r
    time_less_than = np.sum(variables["tsteps"].magnitude <= 2.5)
    dr_before = np.diff(radius[:time_less_than]).magnitude
    dr_after = np.diff(radius[time_less_than:]).magnitude

    # act
    sut = np.mean(dr_before) / np.mean(dr_after)

    # assert
    assert sut < 1
    assert sut > 0
    assert (dr_before > 0).all()
