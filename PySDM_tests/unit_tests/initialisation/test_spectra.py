"""
Created at 2019
"""

from PySDM.initialisation.spectra import Lognormal, Exponential

import numpy as np
from numpy.testing import assert_approx_equal
import pytest


class TestLognormal:
    def test_size_distribution_n_part(self):
        # Arrange
        s = 1.5
        n_part = 256
        sut = Lognormal(n_part, .5e-5, s)

        # Act
        m, dm = np.linspace(.1e-6, 100e-6, 100, retstep=True)
        sd = sut.size_distribution(m)

        # Assert
        assert_approx_equal(np.sum(sd) * dm, n_part, 4)

    def test_size_distribution_r_mode(self):
        # Arrange
        s = 1.001
        r_mode = 1e-6
        sut = Lognormal(1, r_mode, s)

        # Act
        m, dm = np.linspace(.01e-6, 100e-6, 10000, retstep=True)
        sd = sut.size_distribution(m)

        # Assert
        assert_approx_equal(
            m[sd == np.amax(sd)],
            r_mode,
            2
        )


class TestExponential:
    @pytest.mark.parametrize("scale", [
        pytest.param(0.5), pytest.param(1), pytest.param(1.5),
    ])
    def test_size_distribution_n_part(self, scale):
        # Arrange
        scale = 1
        n_part = 256
        sut = Exponential(n_part, scale)

        # Act
        m, dm = np.linspace(0, 5, 10000, retstep=True)
        sd = sut.size_distribution(m)

        # Assert
        assert_approx_equal(np.sum(sd) * dm, n_part, 2)
