"""tests for hydrostatic profile considering variation in gravitational acceleration"""

from PySDM import Formulae
from PySDM import physics
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestHydrostaticsVarG:
    @staticmethod
    def test_units():
        """checks that pressure returns something in pressure units"""
        with DimensionalAnalysis():
            # arrange
            sut = Formulae(
                hydrostatics="VariableGIsothermal",
                constants={
                    "celestial_body_radius": 6378 * physics.si.km,
                    "g_std": 9.8 * physics.si.m / physics.si.s**2,
                },
            ).hydrostatics.pressure
            p0 = 1000 * physics.si.hPa
            z = 10 * physics.si.km
            temperature = 250 * physics.si.K
            molar_mass = physics.constants_defaults.Md

            # act
            result = sut(z, p0, temperature, molar_mass)

            # assert
            assert result.check("[pressure]")

    @staticmethod
    def test_sane_value_at_10km():
        """evaluates the pressure at a given large height and checks if within sane range"""
        # arrange
        sut = Formulae(
            hydrostatics="VariableGIsothermal",
            constants={
                "celestial_body_radius": 6378 * physics.si.km,
                "g_std": 9.8 * physics.si.m / physics.si.s**2,
            },
        ).hydrostatics.pressure

        # act
        result = sut(
            z=10 * physics.si.km,
            p0=1000 * physics.si.hPa,
            temperature=250 * physics.si.K,
            molar_mass=physics.constants_defaults.Md,
        )

        # assert
        assert 25 * physics.si.kPa < result < 26 * physics.si.kPa
