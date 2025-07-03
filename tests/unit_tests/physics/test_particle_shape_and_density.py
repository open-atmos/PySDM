# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

import matplotlib.pyplot as plt
import numpy as np

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
    @pytest.mark.parametrize("variant", ("ColumnarIce",))
    def test_spheroid_shape_units(variant):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            formulae = Formulae(particle_shape_and_density=variant)
            mass = 1e-10 * si.kg
            columnar_shape = formulae.particle_shape_and_density

            # Act
            polar_radius = columnar_shape.polar_radius_empirical_parametrisation(mass)
            aspect_ratio = columnar_shape.aspect_ratio_empirical_parametrisation(mass)
            eccentricity = columnar_shape.eccentricity(aspect_ratio)
            equatorial_radius = columnar_shape.equatorial_radius(
                polar_radius, aspect_ratio
            )

            # Assert
            assert polar_radius.check("[length]")
            assert equatorial_radius.check("[length]")
            assert aspect_ratio.check(si.dimensionless)
            assert eccentricity.check(si.dimensionless)

    @staticmethod
    def test_columnar_ice_geometric_values_against_spichtinger_and_gierens_2009_fig_1_and_2(
        plot=False,
    ):
        """Fig. 1 & 2 in [Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)"""
        # arrange
        si = constants_defaults.si
        mass = np.logspace(base=10, start=-16, stop=-7, num=10) * si.kg
        formulae = Formulae(particle_shape_and_density="ColumnarIce")
        columnar_shape = formulae.particle_shape_and_density

        polar_diameter_reference = (
            np.array(
                [
                    0.57,
                    1.24,
                    2.67,
                    5.74,
                    14.92,
                    42.51,
                    121.08,
                    344.85,
                    982.14,
                    2797.16,
                ]
            )
            * si.micrometer
        )
        aspect_ratio_reference = np.array(
            [
                1,
                1,
                1,
                1,
                1.32,
                2.01,
                3.05,
                4.64,
                7.05,
                10.73,
            ]
        )

        # Act
        polar_radius = columnar_shape.polar_radius_empirical_parametrisation(mass)
        aspect_ratio = columnar_shape.aspect_ratio_empirical_parametrisation(mass)
        equatorial_radius = columnar_shape.equatorial_radius(polar_radius, aspect_ratio)

        polar_diameter = polar_radius * 2
        equatorial_diameter = equatorial_radius * 2

        # plot
        plt.xlabel("mass (kg)")
        plt.ylabel("length (m)")
        plt.xlim(mass[0], mass[-1])
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()

        plt.title("ColumnarIce length")
        plt.plot(mass, polar_diameter * 2, color="red")
        plt.plot(mass, equatorial_diameter * 2, color="blue")

        if plot:
            plt.show()
        else:
            plt.clf()

        # Assert
        np.testing.assert_almost_equal(
            polar_diameter, polar_diameter_reference, decimal=1
        )
        np.testing.assert_almost_equal(aspect_ratio, aspect_ratio_reference, decimal=1)

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
