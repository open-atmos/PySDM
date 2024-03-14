# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from scipy.special import erfinv  # pylint: disable=no-name-in-module

from PySDM import Formulae
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics import constants_defaults


class TestTrivia:
    @staticmethod
    @pytest.mark.parametrize("x", (-0.9, -0.1, -0.01, 0, 0.01, 0.1, 0.9))
    def test_erfinv_approx_reltol(x):
        # arrange
        trivia = Formulae().trivia
        expected = erfinv(x)

        # act
        actual = trivia.erfinv_approx(x)

        # assert
        if expected == 0:
            assert actual == 0
        elif np.isnan(expected):
            assert np.isnan(actual)
        elif np.isinf(expected):
            assert np.isinf(actual)
            assert np.sign(actual) == np.sign(expected)
        else:
            assert np.abs(np.log(actual / expected)) < 1e-3

    @staticmethod
    def test_erfinv_approx_abstol():
        # arrange
        formulae = Formulae()

        # act
        params = formulae.trivia.erfinv_approx(0.25)

        # assert
        diff = np.abs(params - 0.2253)
        np.testing.assert_array_less(diff, 1e-3)

    @staticmethod
    def test_isotopic_enrichment_to_delta_SMOW():
        # arrange
        formulae = Formulae()
        ARBITRARY_VALUE = 44

        # act
        delta = formulae.trivia.isotopic_enrichment_to_delta_SMOW(ARBITRARY_VALUE, 0)

        # assert
        assert delta == ARBITRARY_VALUE

    @staticmethod
    def test_schmidt_number():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae()
            si = constants_defaults.si
            sut = formulae.trivia.air_schmidt_number
            eta_air = formulae.air_dynamic_viscosity.eta_air(temperature=300 * si.K)

            # Act
            sc = sut(
                dynamic_viscosity=eta_air,
                diffusivity=constants_defaults.D0,
                density=1 * si.kg / si.m**3,
            )

            # Assert
            assert sc.check("[]")
