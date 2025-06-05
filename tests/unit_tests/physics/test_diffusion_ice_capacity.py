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
