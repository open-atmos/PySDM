# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM.formulae import Formulae
from PySDM.physics import constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestParticleShapeAndDensity:
    @staticmethod
    def test_mass_to_volume_units():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae()
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.mass_to_volume
            mass = 1 * si.gram

            # Act
            volume = sut(mass)

            # Assert
            assert volume.check("[volume]")

    @staticmethod
    def test_volume_to_mass_units():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae()
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.volume_to_mass
            volume = 1 * si.micrometre**3

            # Act
            mass = sut(volume)

            # Assert
            assert mass.check("[mass]")
