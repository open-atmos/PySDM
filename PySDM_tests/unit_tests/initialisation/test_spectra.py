from PySDM.physics.spectra import Lognormal, Exponential, Sum, default_interpolation_grid

import numpy as np
from numpy.testing import assert_approx_equal
import pytest


class TestLognormal:

    @staticmethod
    def test_size_distribution_n_part():
        # Arrange
        s = 1.5
        n_part = 256
        sut = Lognormal(n_part, .5e-5, s)

        # Act
        m, dm = np.linspace(.1e-6, 100e-6, 100, retstep=True)
        sd = sut.size_distribution(m)

        # Assert
        assert_approx_equal(np.sum(sd) * dm, n_part, 4)

    @staticmethod
    def test_size_distribution_r_mode():
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

    @staticmethod
    @pytest.mark.parametrize("scale", [
        pytest.param(0.5), pytest.param(1), pytest.param(1.5),
    ])
    def test_size_distribution_n_part(scale):
        # Arrange
        scale = 1
        n_part = 256
        sut = Exponential(n_part, scale)

        # Act
        m, dm = np.linspace(0, 5, 10000, retstep=True)
        sd = sut.size_distribution(m)

        # Assert
        assert_approx_equal(np.sum(sd) * dm, n_part, 2)


class TestSum:
    scale = 1
    n_part = 256
    exponential = Exponential(n_part, scale)

    s = 1.001
    r_mode = 1e-6
    lognormal = Lognormal(1, r_mode, s)

    @staticmethod
    def test_size_distribution():
        # Arrange
        sut = Sum((TestSum.exponential,))

        # Act
        x = np.linspace(0, 1)
        sut_sd = sut.size_distribution(x)
        exp_sd = TestSum.exponential.size_distribution(x)

        # Assert
        np.testing.assert_array_equal(sut_sd, exp_sd)

    @staticmethod
    def test_cumulative():
        # Arrange
        sut = Sum((TestSum.exponential,))

        # Act
        x = np.linspace(0, 1)
        sut_c = sut.cumulative(x)
        exp_c = TestSum.exponential.cumulative(x)

        # Assert
        np.testing.assert_array_equal(sut_c, exp_c)

    @staticmethod
    @pytest.mark.parametrize("distributions", [
        pytest.param((exponential,), id="single exponential"),
        pytest.param((lognormal,), id="single lognormal"),
        pytest.param((exponential, exponential), id="2 exponentials")
    ])
    def test_percentiles(distributions):
        # Arrange
        sut = Sum(distributions)

        # Act
        cdf_values = default_interpolation_grid
        sut_p = sut.percentiles(cdf_values)
        exp_p = distributions[0].percentiles(cdf_values)

        # Assert
        np.testing.assert_array_almost_equal(sut_p, exp_p, decimal=3)

