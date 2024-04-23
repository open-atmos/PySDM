""" tests for lognormal probability distribution """

import numpy as np
import pytest
from PySDM.initialisation.spectra import Lognormal


class TestSpectraLognormal:
    """checks ctor args against computed values"""

    @staticmethod
    @pytest.mark.parametrize(
        "m_mode, s_geom",
        (
            (0.01, 1.5),
            (0.1, 1.1),
            (1, 1.01),
        ),
    )
    def test_median(m_mode, s_geom):
        # arrange
        sut = Lognormal(m_mode=m_mode, norm_factor=1, s_geom=s_geom)

        # act
        median = sut.percentiles(0.5)

        # assert
        assert median == m_mode

    @staticmethod
    @pytest.mark.parametrize(
        "m_mode, s_geom",
        (
            (0.01, 3.5),
            (0.1, 2.1),
            (1, 1.5),
        ),
    )
    def test_mean(m_mode, s_geom):
        # arrange
        sut = Lognormal(m_mode=m_mode, norm_factor=1, s_geom=s_geom)
        x = np.linspace(m_mode / 1000, m_mode * 1000, num=10000)

        # act
        mean = np.sum(sut.pdf(x) * x) / np.sum(sut.pdf(x))

        # assert
        np.testing.assert_approx_equal(
            actual=np.log(m_mode) + 0.5 * np.log(s_geom) ** 2,
            desired=np.log(mean),
            significant=3,
        )
