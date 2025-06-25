"""
test for diffusion ice capacity parametrisations
"""

import pytest
from matplotlib import pyplot
import numpy as np
from PySDM.formulae import Formulae, _choices
from PySDM.physics import diffusion_ice_capacity
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM import physics


class TestDiffusionIceCapacity:
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(diffusion_ice_capacity))
    def test_basics(variant, plot=False):
        # arrange
        si = physics.si
        masses = np.logspace(base=10, start=-16, stop=-8.5, num=10) * si.kg
        formulae = Formulae(
            diffusion_ice_capacity=variant,
        )
        sut = formulae.diffusion_ice_capacity

        # act
        values = sut.capacity(masses)

        pyplot.xlabel("mass (kg)")
        pyplot.ylabel("capacity (m)")
        pyplot.xlim(masses[0], masses[-1])
        pyplot.xscale("log")
        pyplot.ylim(1e-7, 5e-4)
        pyplot.yscale("log")
        pyplot.grid()
        pyplot.plot(masses, values, color="black")
        pyplot.title(f"variant={variant}")
        # plot
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (values > 0).all()
        assert (np.diff(values) > 0).all()

    @staticmethod
    @pytest.mark.parametrize("variant", _choices(diffusion_ice_capacity))
    def test_units(variant):

        with DimensionalAnalysis():
            # arrange
            si = physics.si
            formulae = Formulae(
                diffusion_ice_capacity=variant,
            )
            sut = formulae.diffusion_ice_capacity
            mass = 1e-12 * si.kg

            # act
            value = sut.capacity(mass)

            # assert
            assert value.check("[length]")

    @staticmethod
    @pytest.mark.parametrize(
        "mass", (44 * physics.si.ng, 666 * physics.si.ug, 123 * physics.si.mg)
    )
    def test_capacity_equals_radius_for_spherical(mass):
        # arrange
        formulae = Formulae(
            diffusion_ice_capacity="Spherical",
            particle_shape_and_density="MixedPhaseSpheres",
        )
        sut = formulae.diffusion_ice_capacity

        # act
        capacity = sut.capacity(mass)

        # assert
        np.testing.assert_approx_equal(
            desired=mass,
            actual=-formulae.particle_shape_and_density.radius_to_mass(-capacity),
            significant=15,
        )

    @staticmethod
    @pytest.mark.parametrize("mass", (44 * physics.si.ng, 0.666 * physics.si.ug))
    def test_columnar_capacity_difference_from_spherical_capacity(mass):
        # arrange
        sut = {
            key: Formulae(diffusion_ice_capacity=key).diffusion_ice_capacity
            for key in ("Spherical", "Columnar")
        }

        # act
        values = {key: sut[key].capacity(mass) for key in sut.keys()}

        # assert
        assert values["Spherical"] != values["Columnar"]
        np.testing.assert_allclose(
            values["Spherical"],
            values["Columnar"],
            rtol=0.2,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "mass", (1e-13 * physics.si.kg, 1e-10 * physics.si.kg, 1e-8 * physics.si.kg)
    )
    def test_prolate_ellipsoid_formula(mass):
        """TODO #1643"""
        # arrange
        formulae = Formulae(
            diffusion_ice_capacity="Columnar",
            particle_shape_and_density="ColumnarIce",
        )
        sut = formulae.diffusion_ice_capacity
        polar_radius_relation = formulae.particle_shape_and_density.polar_radius
        aspect_ratio_relation = formulae.particle_shape_and_density.aspect_ratio


        # act
        polar_radius = polar_radius_relation(mass)
        aspect_ratio = aspect_ratio_relation(mass)

        capacity = sut.capacity(mass)
