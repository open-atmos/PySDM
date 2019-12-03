"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.state.state_factory import StateFactory
from tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM.simulation.initialisation.spectra import Lognormal
from PySDM.simulation.initialisation.spectral_discretisation import linear
from PySDM.backends.default import Default

backend = Default


class TestMaths:
    @staticmethod
    def test_moment_0d():
        # Arrange
        n_part = 10000
        v_mean = 2e-6
        d = 1.2

        v_min = 0.01e-6
        v_max = 10e-6
        n_sd = 32

        spectrum = Lognormal(n_part, v_mean, d)
        v, n = linear(n_sd, spectrum, (v_min, v_max))
        particles = DummyParticles(backend, n_sd)
        state = StateFactory.state_0d(n=n, extensive={'volume': v}, intensive={}, particles=particles)

        true_mean, true_var = spectrum.stats(moments='mv')

        # TODO: add a moments_0 wrapper
        moment_0 = np.empty((1,), dtype=int)
        moments = np.empty((1, 1), dtype=float)

        # Act
        state.moments(moment_0, moments, specs={'volume': (0,)})
        discr_zero = moments[0, 0]

        state.moments(moment_0, moments, specs={'volume': (1,)})
        discr_mean = moments[0, 0]

        state.moments(moment_0, moments, specs={'volume': (2,)})
        discr_mrsq = moments[0, 0]

        # Assert
        assert abs(discr_zero - 1) / 1 < 1e-3

        assert abs(discr_mean - true_mean) / true_mean < .01e-1

        true_mrsq = true_var + true_mean**2
        assert abs(discr_mrsq - true_mrsq) / true_mrsq < .05e-1

