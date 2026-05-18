# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation import spectra
from PySDM.physics import si

m_mode = 0.5e-5
n_part = 256 * 16
s_geom = 1.5
spectrum = spectra.Lognormal(n_part, m_mode, s_geom)
m_range = (0.1 * si.um, 100 * si.um)
formulae = Formulae()


class TestSpectralDiscretisation:
    @staticmethod
    @pytest.mark.parametrize(
        "discretisation",
        (
            pytest.param(spectral_sampling.Linear(spectrum, size_range=m_range)),
            pytest.param(spectral_sampling.Logarithmic(spectrum, size_range=m_range)),
            pytest.param(
                spectral_sampling.ConstantMultiplicity(spectrum, size_range=m_range)
            ),
            pytest.param(spectral_sampling.Linear(spectrum)),
            pytest.param(spectral_sampling.Logarithmic(spectrum)),
            pytest.param(spectral_sampling.ConstantMultiplicity(spectrum)),
        ),
    )
    @pytest.mark.parametrize("method", ("deterministic", "quasirandom", "pseudorandom"))
    def test_all_methods_with_lognormal(
        discretisation, method, backend_instance, plot=False
    ):
        # Arrange
        n_sd = 10000

        # Act
        m, n = getattr(discretisation, f"sample_{method}")(
            n_sd, backend=backend_instance
        )

        # Plot
        pyplot.title(f"{discretisation.__class__.__name__} {method=}")
        pyplot.scatter(m, n)
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # Assert
        assert m.shape == n.shape
        assert n.shape == (n_sd,)
        assert np.min(m) >= m_range[0]
        assert np.max(m) <= m_range[1]
        actual = np.sum(n)
        desired = spectrum.cumulative(m_range[1]) - spectrum.cumulative(m_range[0])
        np.testing.assert_approx_equal(
            actual=actual / desired,
            desired=1.0,
            significant=2 if method == "pseudorandom" else 4,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "sampling_class, error_threshold",
        (
            (spectral_sampling.ConstantMultiplicity, None),
            (spectral_sampling.Logarithmic, None),
            (spectral_sampling.Linear, None),
            pytest.param(
                spectral_sampling.ConstantMultiplicity,
                1e-5,
                marks=pytest.mark.xfail(strict=True, raises=ValueError),
            ),
            pytest.param(
                spectral_sampling.Logarithmic,
                1e-5,
                marks=pytest.mark.xfail(strict=True, raises=ValueError),
            ),
            pytest.param(
                spectral_sampling.Linear,
                1e-5,
                marks=pytest.mark.xfail(strict=True, raises=ValueError),
            ),
        ),
    )
    def test_error_threshold_with_deterministic_sampling(
        sampling_class, error_threshold
    ):
        sampling_class(
            spectra.Lognormal(n_part, m_mode, s_geom), error_threshold=error_threshold
        ).sample_deterministic(n_sd=10)
