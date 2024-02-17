# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Jensen_and_Nugent_2017

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Jensen_and_Nugent_2017.__file__).parent / "Fig_1.ipynb", plot=PLOT
    )


class TestFig1:
    @staticmethod
    @pytest.mark.parametrize(
        "idx, value",
        (
            (0, 3.7e-1 / si.cm**3),
            (-1, 2e-2 / si.cm**3),
        ),
    )
    def test_dn_dlogr_values(variables, idx, value):
        np.testing.assert_approx_equal(
            actual=variables["dN_dlogr"][idx], desired=value, significant=2
        )

    @staticmethod
    def test_two_maxima(variables):
        assert (
            signal.argrelextrema(np.asarray(variables["dN_dlogr"]), np.greater)[
                0
            ].shape[0]
            == 2
        )

    @staticmethod
    def test_maximal_value(variables):
        np.testing.assert_approx_equal(
            actual=max(variables["dN_dlogr"]), desired=2.3e2 / si.cm**3, significant=2
        )
