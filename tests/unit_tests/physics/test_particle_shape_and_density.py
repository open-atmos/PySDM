# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from mpmath.libmp.backend import sage_utils

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
            assert re.check(si.dimensionless)

    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres",))
    def test_dm_dt_over_m_units(variant):
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(particle_shape_and_density=variant)
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.dm_dt_over_m

            # act
            re = sut(r=1 * si.um, r_dr_dt=1 * si.um**2 / si.s)

            # assert
            assert re.check("1/[time]")

    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres",))
    def test_r_dr_dt_units(variant):
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(particle_shape_and_density=variant)
            si = constants_defaults.si
            sut = formulae.particle_shape_and_density.r_dr_dt

            # act
            re = sut(r=1 * si.um, dm_dt_over_m=1 / si.s)

            # assert
            assert re.check("[area]/[time]")
