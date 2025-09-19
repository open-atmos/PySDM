# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from scipy.special import erfinv  # pylint: disable=no-name-in-module

from PySDM import Formulae
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics import constants_defaults, si


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
            assert sc.check(si.dimensionless)

    @staticmethod
    def test_poissonian_avoidance_function():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae()
            si = constants_defaults.si
            sut = formulae.trivia.poissonian_avoidance_function

            # Act
            prob = sut(
                r=1 / si.s,
                dt=10 * si.min,
            )

            # Assert
            assert prob.check(si.dimensionless)

    @staticmethod
    def test_kelvin_to_celsius():
        # arrange
        formulae = Formulae()
        temperature_in_kelvin = 44

        # act
        temperature_in_celsius = formulae.trivia.K2C(temperature_in_kelvin)

        # assert
        assert temperature_in_celsius == temperature_in_kelvin - 273.15

    @staticmethod
    def test_celsius_to_kelvin():
        # arrange
        formulae = Formulae()
        temperature_in_celsius = 666

        # act
        temperature_in_kelvin = formulae.trivia.C2K(temperature_in_celsius)

        # assert
        assert temperature_in_kelvin == temperature_in_celsius + 273.15

    @staticmethod
    @pytest.mark.parametrize("delta", (0.86 - 1, 0.9 - 1, 0.98 - 1))
    def test_moles_heavy_atom(
        delta,
        m_t=1 * si.ng,
    ):
        # arrange
        formulae = Formulae()
        const = formulae.constants

        from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES

        attributes = {
            "multiplicity": np.asarray([0]),
            "signed water mass": m_t,
            "moles_2H": formulae.trivia.moles_heavy_atom(
                delta=delta,
                mass_total=m_t,
                molar_mass_heavy_molecule=const.M_2H_1H_16O,
                R_STD=const.VSMOW_R_2H,
                light_atoms_per_light_molecule=2,
            ),
        }
        for isotope in HEAVY_ISOTOPES:
            if isotope != "2H":
                attributes[f"moles_{isotope}"] = 0
        attributes["moles light water"] = (
            attributes["signed water mass"] / const.M_1H2_16O
        )
        ratio = attributes["moles_2H"] / attributes["moles light water"]
        sut = formulae.trivia.isotopic_ratio_2_delta(ratio, const.VSMOW_R_2H)

        # assert
        np.testing.assert_approx_equal(actual=sut, desired=delta, significant=10)
