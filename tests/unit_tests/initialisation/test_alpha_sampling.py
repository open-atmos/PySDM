# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation.spectra.exponential import Exponential
from PySDM.physics import si


@pytest.mark.parametrize(
    "alpha_class, alias_class, alphavalue",
    (
        pytest.param(
            (spectral_sampling.AlphaSampling),
            (spectral_sampling.Linear),
            (1),
        ),
        pytest.param(
            (spectral_sampling.AlphaSampling),
            (spectral_sampling.ConstantMultiplicity),
            (0),
        ),
    ),
)
@pytest.mark.parametrize("method", ("deterministic", "quasirandom", "pseudorandom"))
def test_spectral_discretisation(
    alpha_class, alias_class, alphavalue, backend_instance, method
):
    # Arrange
    n_sd = 100
    backend = backend_instance

    spectrum = Exponential(
        norm_factor=2**23 / si.metre**3,
        scale=0.03 * si.micrometre,
    )

    error_threshold = None if method != "pseudorandom" else 0.1
    alpha = alpha_class(
        spectrum,
        alpha=alphavalue,
        dist_1_inv=lambda y, size_range: (size_range[1] - size_range[0]) * y
        + size_range[0],
        interp_points=10000,
        error_threshold=error_threshold,
    )
    alias = alias_class(spectrum, error_threshold=error_threshold)

    # Act
    m_alpha, n_alpha = getattr(alpha, f"sample_{method}")(n_sd, backend=backend)
    m_alias, n_alias = getattr(alias, f"sample_{method}")(n_sd, backend=backend)

    # Assert
    np.testing.assert_allclose(
        m_alpha, m_alias, rtol=1e-3, atol=1e-6, err_msg="Size bins do not match"
    )
    np.testing.assert_allclose(
        n_alpha, n_alias, rtol=1e-3, atol=1e-6, err_msg="Multiplicity does not match"
    )
