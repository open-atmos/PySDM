# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.initialisation.discretise_multiplicities import discretise_multiplicities
from PySDM.initialisation.sampling.spectral_sampling import Linear
from PySDM.initialisation.spectra.lognormal import Lognormal

from ..dummy_particulator import DummyParticulator


class TestMaths:
    @staticmethod
    # pylint: disable=too-many-locals
    def test_moment_0d(backend_class):
        # Arrange
        n_part = 100000
        v_mean = 2e-6
        d = 1.2
        n_sd = 32

        spectrum = Lognormal(n_part, v_mean, d)
        v, n = Linear(spectrum).sample(n_sd)
        T = 300.0
        n = discretise_multiplicities(n)
        particulator = DummyParticulator(backend_class, n_sd)
        attribute = {"multiplicity": n, "volume": v, "heat": T * v}
        particulator.request_attribute("temperature")
        particulator.build(attribute)

        true_mean, true_var = spectrum.stats(moments="mv")

        # TODO #217 : add a moments_0 wrapper
        moment_0 = particulator.backend.Storage.empty((1,), dtype=float)
        moments = particulator.backend.Storage.empty((1, 1), dtype=float)

        # Act
        particulator.moments(moment_0=moment_0, moments=moments, specs={"volume": (0,)})
        discr_zero = moments[0, slice(0, 1)].to_ndarray()

        particulator.moments(moment_0=moment_0, moments=moments, specs={"volume": (1,)})
        discr_mean = moments[0, slice(0, 1)].to_ndarray()

        particulator.moments(moment_0=moment_0, moments=moments, specs={"volume": (2,)})
        discr_mean_radius_squared = moments[0, slice(0, 1)].to_ndarray()

        particulator.moments(
            moment_0=moment_0, moments=moments, specs={"temperature": (0,)}
        )
        discr_zero_T = moments[0, slice(0, 1)].to_ndarray()

        particulator.moments(
            moment_0=moment_0, moments=moments, specs={"temperature": (1,)}
        )
        discr_mean_T = moments[0, slice(0, 1)].to_ndarray()

        particulator.moments(
            moment_0=moment_0, moments=moments, specs={"temperature": (2,)}
        )
        discr_mean_T_squared = moments[0, slice(0, 1)].to_ndarray()

        # Assert
        assert abs(discr_zero - 1) / 1 < 1e-3

        assert abs(discr_mean - true_mean) / true_mean < 0.01e-1

        true_mrsq = true_var + true_mean**2
        assert abs(discr_mean_radius_squared - true_mrsq) / true_mrsq < 0.05e-1

        assert discr_zero_T == discr_zero
        assert discr_mean_T == 300.0
        np.testing.assert_approx_equal(discr_mean_T_squared, 300.0**2, significant=6)

    @staticmethod
    # pylint: disable=too-many-locals
    def test_spectrum_moment_0d(backend_class):
        # Arrange
        n_part = 100000
        v_mean = 2e-6
        d = 1.2
        n_sd = 32

        spectrum = Lognormal(n_part, v_mean, d)
        v, n = Linear(spectrum).sample(n_sd)
        T = 300.0
        n = discretise_multiplicities(n)
        particulator = DummyParticulator(backend_class, n_sd)
        attribute = {"multiplicity": n, "volume": v, "heat": T * v}
        particulator.request_attribute("temperature")
        particulator.build(attribute)

        v_bins = np.linspace(0, 5e-6, num=5, endpoint=True)

        # TODO #217 : add a moments_0 wrapper
        spectrum_moment_0 = particulator.backend.Storage.empty(
            (len(v_bins) - 1, 1), dtype=float
        )
        spectrum_moments = particulator.backend.Storage.empty(
            (len(v_bins) - 1, 1), dtype=float
        )
        moment_0 = particulator.backend.Storage.empty((1,), dtype=float)
        moments = particulator.backend.Storage.empty((1, 1), dtype=float)
        v_bins_edges = particulator.backend.Storage.from_ndarray(v_bins)

        # Act
        particulator.spectrum_moments(
            moment_0=spectrum_moment_0,
            moments=spectrum_moments,
            attr="volume",
            rank=1,
            attr_bins=v_bins_edges,
        )
        actual = spectrum_moments.to_ndarray()

        expected = np.empty((len(v_bins) - 1, 1), dtype=float)
        for i in range(len(v_bins) - 1):
            particulator.moments(
                moment_0=moment_0,
                moments=moments,
                specs={"volume": (1,)},
                attr_range=(v_bins[i], v_bins[i + 1]),
            )
            expected[i, 0] = moments[0, 0]

        # Assert
        np.testing.assert_array_almost_equal(actual, expected)
