import numpy as np

from PySDM.initialisation.multiplicities import discretise_n
from PySDM.initialisation.spectral_sampling import Linear
from PySDM.physics.spectra import Lognormal
# noinspection PyUnresolvedReferences
from ...backends_fixture import backend
from ..dummy_particulator import DummyParticulator


class TestMaths:

    @staticmethod
    def test_moment_0d(backend):
        # Arrange
        n_part = 100000
        v_mean = 2e-6
        d = 1.2
        n_sd = 32

        spectrum = Lognormal(n_part, v_mean, d)
        v, n = Linear(spectrum).sample(n_sd)
        T = np.full_like(v, 300.)
        n = discretise_n(n)
        particles = DummyParticulator(backend, n_sd)
        attribute = {'n': n, 'volume': v, 'temperature': T, 'heat': T*v}
        particles.build(attribute)
        state = particles.attributes

        true_mean, true_var = spectrum.stats(moments='mv')

        # TODO #217 : add a moments_0 wrapper
        moment_0 = particles.backend.Storage.empty((1,), dtype=float)
        moments = particles.backend.Storage.empty((1, 1), dtype=float)

        # Act
        state.moments(moment_0, moments, specs={'volume': (0,)})
        discr_zero = moments[0, slice(0, 1)].to_ndarray()

        state.moments(moment_0, moments, specs={'volume': (1,)})
        discr_mean = moments[0, slice(0, 1)].to_ndarray()

        state.moments(moment_0, moments, specs={'volume': (2,)})
        discr_mean_radius_squared = moments[0, slice(0, 1)].to_ndarray()

        state.moments(moment_0, moments, specs={'temperature': (0,)})
        discr_zero_T = moments[0, slice(0, 1)].to_ndarray()

        state.moments(moment_0, moments, specs={'temperature': (1,)})
        discr_mean_T = moments[0, slice(0, 1)].to_ndarray()

        state.moments(moment_0, moments, specs={'temperature': (2,)})
        discr_mean_T_squared = moments[0, slice(0, 1)].to_ndarray()

        # Assert
        assert abs(discr_zero - 1) / 1 < 1e-3

        assert abs(discr_mean - true_mean) / true_mean < .01e-1

        true_mrsq = true_var + true_mean**2
        assert abs(discr_mean_radius_squared - true_mrsq) / true_mrsq < .05e-1
        
        assert discr_zero_T == discr_zero
        assert discr_mean_T == 300.
        np.testing.assert_approx_equal(discr_mean_T_squared, 300. ** 2, significant=6)


    @staticmethod
    def test_spectrum_moment_0d(backend):
        # Arrange
        n_part = 100000
        v_mean = 2e-6
        d = 1.2
        n_sd = 32

        spectrum = Lognormal(n_part, v_mean, d)
        v, n = Linear(spectrum).sample(n_sd)
        T = np.full_like(v, 300.)
        n = discretise_n(n)
        particles = DummyParticulator(backend, n_sd)
        attribute = {'n': n, 'volume': v, 'temperature': T, 'heat': T*v}
        particles.build(attribute)
        state = particles.attributes

        v_bins = np.linspace(0, 5e-6, num=5, endpoint=True)

        true_mean, true_var = spectrum.stats(moments='mv')

        # TODO #217 : add a moments_0 wrapper
        spectrum_moment_0 = particles.backend.Storage.empty((len(v_bins) - 1, 1), dtype=float)
        spectrum_moments = particles.backend.Storage.empty((len(v_bins) - 1, 1), dtype=float)
        moment_0 = particles.backend.Storage.empty((1,), dtype=float)
        moments = particles.backend.Storage.empty((1, 1), dtype=float)
        v_bins_edges= particles.backend.Storage.from_ndarray(v_bins)

        # Act
        state.spectrum_moments(spectrum_moment_0, spectrum_moments, attr='volume', rank=1, attr_bins=v_bins_edges)
        actual = spectrum_moments.to_ndarray()

        expected = np.empty((len(v_bins) - 1, 1), dtype=float)
        for i in range(len(v_bins) - 1):
            state.moments(moment_0, moments, specs={'volume': (1,)}, attr_range=(v_bins[i], v_bins[i+1]))
            expected[i, 0] = moments[0, 0]

        # Assert
        np.testing.assert_array_almost_equal(actual, expected)
