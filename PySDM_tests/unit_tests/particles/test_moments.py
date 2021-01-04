"""
Created at 05.08.2019
"""

import numpy as np

# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.spectral_sampling import Linear
from PySDM_tests.unit_tests.dummy_core import DummyCore


class TestMaths:

    @staticmethod
    def test_moment_0d(backend):
        # Arrange
        n_part = 10000
        v_mean = 2e-6
        d = 1.2

        v_min = 0.01e-6
        v_max = 10e-6
        n_sd = 32

        spectrum = Lognormal(n_part, v_mean, d)
        v, n = Linear(spectrum, (v_min, v_max)).sample(n_sd)
        T = np.full_like(v, 300.)
        n = discretise_n(n)
        particles = DummyCore(backend, n_sd)
        attribute = {'n': n, 'volume': v, 'temperature': T}
        particles.build(attribute)
        state = particles.particles

        true_mean, true_var = spectrum.stats(moments='mv')

        # TODO: add a moments_0 wrapper
        moment_0 = particles.backend.Storage.empty((1,), dtype=int)
        moments = particles.backend.Storage.empty((1, 1), dtype=float)

        # Act
        state.moments(moment_0, moments, specs={'volume': (0,)})
        discr_zero = moments[0, 0]

        state.moments(moment_0, moments, specs={'volume': (1,)})
        discr_mean = moments[0, 0]

        state.moments(moment_0, moments, specs={'volume': (2,)})
        discr_mean_radius_squared = moments[0, 0]

        state.moments(moment_0, moments, specs={'temperature': (0,)})
        discr_zero_T = moments[0, 0]

        state.moments(moment_0, moments, specs={'temperature': (1,)})
        discr_mean_T = moments[0, 0]

        state.moments(moment_0, moments, specs={'temperature': (2,)})
        discr_mean_T_squared = moments[0, 0]

        # Assert
        assert abs(discr_zero - 1) / 1 < 1e-3

        assert abs(discr_mean - true_mean) / true_mean < .01e-1

        true_mrsq = true_var + true_mean**2
        assert abs(discr_mean_radius_squared - true_mrsq) / true_mrsq < .05e-1
        
        assert discr_zero_T == discr_zero
        assert discr_mean_T == 300.
        assert discr_mean_T_squared == 300. ** 2
