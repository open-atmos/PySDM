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
            id="full range",
        ),
        pytest.param(
            (spectral_sampling.AlphaSampling),
            (spectral_sampling.ConstantMultiplicity),
            (0),
            id="partial range",
        ),
    ),
)
def test_spectral_discretisation(
    alpha_class, alias_class, alphavalue, backend_instance
):
    # Arrange
    n_sd = 100
    backend = backend_instance

    spectrum = Exponential(
        norm_factor=2**23 / si.metre**3,
        scale=0.03 * si.micrometre,
    )

    alpha = alpha_class(spectrum, alpha=alphavalue)
    alias = alias_class(spectrum)

    # Act
    m_alpha, n_alpha = alpha.sample(n_sd, backend=backend)
    m_alias, n_alias = alias.sample(n_sd, backend=backend)

    # Assert
    np.testing.assert_allclose(
        m_alpha, m_alias, rtol=1e-3, atol=1e-6, err_msg="Size bins do not match"
    )
    np.testing.assert_allclose(
        n_alpha, n_alias, rtol=1e-3, atol=1e-6, err_msg="Multiplicity does not match"
    )
