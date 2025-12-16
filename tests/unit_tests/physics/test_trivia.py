# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from scipy.special import erfinv  # pylint: disable=no-name-in-module

from PySDM import Formulae
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics import constants_defaults, si
from PySDM.physics.trivia import Trivia


CONST = Formulae().constants


class TestTrivia:
    @staticmethod
    @pytest.mark.parametrize("x", (-0.9, -0.1, -0.01, 0, 0.01, 0.1, 0.9))
    def test_erfinv_approx_reltol(x):
        # arrange
        expected = erfinv(x)

        # act
        actual = Trivia.erfinv_approx(const=CONST, c=x)

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

        # act
        params = Trivia.erfinv_approx(const=CONST, c=0.25)

        # assert
        diff = np.abs(params - 0.2253)
        np.testing.assert_array_less(diff, 1e-3)

    @staticmethod
    def test_isotopic_enrichment_to_delta_SMOW():
        # arrange
        ARBITRARY_VALUE = 44

        # act
        delta = Trivia.isotopic_enrichment_to_delta_SMOW(
            E=ARBITRARY_VALUE, delta_0_SMOW=0
        )

        # assert
        assert delta == ARBITRARY_VALUE

    @staticmethod
    def test_schmidt_number():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae()
            si = constants_defaults.si  # pylint: disable=redefined-outer-name
            sut = Trivia.air_schmidt_number
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
            si = constants_defaults.si  # pylint: disable=redefined-outer-name
            sut = Trivia.poissonian_avoidance_function

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
        temperature_in_kelvin = 44

        # act
        temperature_in_celsius = Trivia.K2C(const=CONST, TK=temperature_in_kelvin)

        # assert
        assert temperature_in_celsius == temperature_in_kelvin - 273.15

    @staticmethod
    def test_celsius_to_kelvin():
        # arrange
        temperature_in_celsius = 666

        # act
        temperature_in_kelvin = Trivia.C2K(const=CONST, TC=temperature_in_celsius)

        # assert
        assert temperature_in_kelvin == temperature_in_celsius + 273.15

    @staticmethod
    @pytest.mark.parametrize(
        "molecular_isotope_ratio",
        (0.86 * CONST.VSMOW_R_2H, 0.9 * CONST.VSMOW_R_2H, 0.98 * CONST.VSMOW_R_2H),
    )
    @pytest.mark.parametrize("water_mass", np.linspace(10**-2, 10**3, 9) * si.ng)
    @pytest.mark.parametrize(
        "mass_other_heavy_isotopes", np.linspace(10**-7, 10**-2, 3) * si.ng
    )
    @pytest.mark.parametrize(
        "heavy_isotope_name, heavy_isotope_molecule",
        (("2H", "2H_1H_16O"), ("17O", "1H2_17O"), ("18O", "1H2_17O")),
    )
    def test_moles_heavy_atom(
        molecular_isotope_ratio,
        water_mass,
        heavy_isotope_name,
        heavy_isotope_molecule,
        mass_other_heavy_isotopes,
    ):
        # arrange
        assert mass_other_heavy_isotopes <= water_mass
        molar_mass_heavy_molecule = getattr(CONST, f"M_{heavy_isotope_molecule}")
        molar_mass_light_molecule = CONST.M_1H2_16O
        if heavy_isotope_name[-1] == "O":
            atoms_per_heavy_molecule = 1
        elif heavy_isotope_name[-1] == "H":
            atoms_per_heavy_molecule = 1
        else:
            assert False

        # act
        moles_heavy_atom = Trivia.moles_heavy_atom(
            mass_total=water_mass,
            molecular_isotope_ratio=molecular_isotope_ratio,
            mass_other_heavy_isotopes=mass_other_heavy_isotopes,
            molar_mass_light_molecule=molar_mass_light_molecule,
            molar_mass_heavy_molecule=molar_mass_heavy_molecule,
            atoms_per_heavy_molecule=atoms_per_heavy_molecule,
        )

        moles_heavy_molecule = atoms_per_heavy_molecule * moles_heavy_atom
        moles_light_molecule = moles_heavy_molecule / molecular_isotope_ratio
        sut = (
            moles_heavy_molecule * molar_mass_heavy_molecule
            + moles_light_molecule * molar_mass_light_molecule
            + mass_other_heavy_isotopes
        )

        # assert
        np.testing.assert_approx_equal(actual=sut, desired=water_mass, significant=5)

    @staticmethod
    @pytest.mark.parametrize(
        "bolin_number, dm_dt_over_m, expected_tau",
        ((1, 2, 0.5), (2, 1, 0.5), (2, 2, 0.25)),
    )
    def test_tau(bolin_number, dm_dt_over_m, expected_tau):
        # arrange
        sut = Trivia.tau

        # act
        value = sut(Bo=bolin_number, dm_dt_over_m=dm_dt_over_m)

        # assert
        np.testing.assert_almost_equal(actual=value, desired=expected_tau)
