# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.formulae import Formulae
from PySDM.physics import constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestParticleShapeAndDensity:
    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres", "MixedPhaseSpheres"))
    def test_mass_to_volume_units(variant):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(particle_shape_and_density=variant)
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.mass_to_volume
            mass = 1 * si.gram

            # Act
            volume = sut(mass)

            # Assert
            assert volume.check("[volume]")

    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres", "MixedPhaseSpheres"))
    def test_volume_to_mass_units(variant):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(particle_shape_and_density=variant)
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.volume_to_mass
            volume = 1 * si.micrometre**3

            # Act
            mass = sut(volume)

            # Assert
            assert mass.check("[mass]")

    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres",))
    def test_reynolds_number(variant):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(particle_shape_and_density=variant)
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.reynolds_number

            # Act
            re = sut(
                radius=10 * si.um,
                dynamic_viscosity=20 * si.uPa * si.s,
                velocity_wrt_air=1 * si.cm / si.s,
                density=1 * si.kg / si.m**3,
            )

            # Assert
            assert re.check("[]")
