"""tests for lognormal probability distribution"""

import numpy as np
import pytest
from PySDM.initialisation.spectra import Lognormal


class TestSpectraLognormal:
    """checks ctor args against computed values"""

    @staticmethod
    @pytest.mark.parametrize(
        "median, s_geom",
        (
            (0.01, 1.5),
            (0.1, 1.1),
            (1, 1.01),
        ),
    )
    def test_median(median, s_geom):
        # arrange
        sut = Lognormal(median=median, norm_factor=1, s_geom=s_geom)

        # act
        median = sut.percentiles(0.5)

        # assert
        assert median == median

    @staticmethod
    @pytest.mark.parametrize(
        "median, s_geom",
        (
            (0.01, 3.5),
            (0.1, 2.1),
            (1, 1.5),
        ),
    )
    def test_mean(median, s_geom):
        # arrange
        sut = Lognormal(median=median, norm_factor=1, s_geom=s_geom)
        x = np.linspace(median / 1000, median * 1000, num=10000)

        # act
        mean = np.sum(sut.pdf(x) * x) / np.sum(sut.pdf(x))

        # assert
        np.testing.assert_approx_equal(
            actual=np.log(median) + 0.5 * np.log(s_geom) ** 2,
            desired=np.log(mean),
            significant=3,
        )

    @staticmethod
    def test_mode():
        # TODO
        spectrum = Lognormal(median=...)
        assert False

    @staticmethod
    def test_instantiation_passing_mode():
        # arrange & act
        spectrum = Lognormal(mode=1, s_geom=1, norm_factor=1)

        # act
        np.testing.assert_approx_equal(actual=spectrum.median, desired=1)
