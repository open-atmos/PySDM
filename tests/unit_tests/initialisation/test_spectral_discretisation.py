# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation.spectra.lognormal import Lognormal
from PySDM.physics import si

m_mode = 0.5e-5
n_part = 256 * 16
s_geom = 1.5
spectrum = Lognormal(n_part, m_mode, s_geom)
m_range = (0.1 * si.um, 100 * si.um)
formulae = Formulae()


@pytest.mark.parametrize(
    "discretisation",
    (
        pytest.param(spectral_sampling.Linear(spectrum, m_range)),
        pytest.param(spectral_sampling.Logarithmic(spectrum, m_range)),
        pytest.param(spectral_sampling.ConstantMultiplicity(spectrum, m_range)),
        pytest.param(spectral_sampling.UniformRandom(spectrum, m_range)),
    ),
)
def test_spectral_discretisation(discretisation, backend_instance):
    # Arrange
    n_sd = 100000
    backend = backend_instance

    # Act
    m, n = discretisation.sample(n_sd, backend=backend)

    # Assert
    assert m.shape == n.shape
    assert n.shape == (n_sd,)
    assert np.min(m) >= m_range[0]
    assert np.max(m) <= m_range[1]
    actual = np.sum(n)
    desired = spectrum.cumulative(m_range[1]) - spectrum.cumulative(m_range[0])
    quotient = actual / desired
    np.testing.assert_approx_equal(actual=quotient, desired=1.0, significant=2)
