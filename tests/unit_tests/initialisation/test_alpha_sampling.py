# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation.spectra.exponential import Exponential
from PySDM.physics import si

formulae = Formulae()
n_sd = 2**13
spectrum = Exponential(
                norm_factor=2**23 / si.metre**3,
            scale=0.03 * si.micrometre,
        )
formulae = Formulae()

class AliasLinear(spectral_sampling.DeterministicSpectralSampling):  # pylint: disable=too-few-public-methods
    def sample(self, n_sd, *, backend=None):  # pylint: disable=unused-argument
        grid = np.linspace(*self.size_range, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)

class AliasConstantMultiplicity(
    spectral_sampling.DeterministicSpectralSampling
):  # pylint: disable=too-few-public-methods
    def __init__(self, spectrum, size_range=None):
        super().__init__(spectrum, size_range)

        self.cdf_range = (
            spectrum.cumulative(self.size_range[0]),
            spectrum.cumulative(self.size_range[1]),
        )
        assert 0 < self.cdf_range[0] < self.cdf_range[1]

    def sample(self, n_sd, *, backend=None):  # pylint: disable=unused-argument
        cdf_arg = np.linspace(self.cdf_range[0], self.cdf_range[1], num=2 * n_sd + 1)
        cdf_arg /= self.spectrum.norm_factor
        percentiles = self.spectrum.percentiles(cdf_arg)

        assert np.isfinite(percentiles).all()

        return self._sample(percentiles, self.spectrum)

@pytest.mark.parametrize(
    "alpha_class, alias_class",
    (
        pytest.param((spectral_sampling.Linear), (AliasLinear), id="full range"),
        pytest.param((spectral_sampling.ConstantMultiplicity), (AliasConstantMultiplicity), id="partial range"),
    ),
)
def test_spectral_discretisation(alpha_class, alias_class, backend_instance):
    # Arrange
    n_sd = 100
    backend = backend_instance

    alpha = alpha_class(spectrum)
    alias = alias_class(spectrum)

    # Act
    m_alpha, n_alpha = alpha.sample(n_sd, backend=backend)
    m_alias, n_alias = alias.sample(n_sd, backend=backend)

    # Assert
    np.testing.assert_allclose(
        m_alpha, m_alias, rtol=1e-1, atol=1.5, err_msg="Size bins do not match"
    )
    np.testing.assert_allclose(
        n_alpha, n_alias, rtol=1e-1, atol=1.5, err_msg="Multiplicity does not match"
    )
