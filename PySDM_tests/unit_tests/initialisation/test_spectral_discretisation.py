from PySDM.initialisation.spectral_sampling import Linear, Logarithmic, ConstantMultiplicity
from PySDM.physics.spectra import Lognormal
import numpy as np
import pytest


@pytest.mark.parametrize("discretisation", [
	pytest.param(Linear),
	pytest.param(Logarithmic),
	pytest.param(ConstantMultiplicity)
])
def test_spectral_discretisation(discretisation):
	# Arrange
	n_sd = 100
	m_mode = .5e-5
	n_part = 256*16
	s_geom = 1.5
	spectrum = Lognormal(n_part, m_mode, s_geom)
	m_range = (.1e-6, 100e-6)

	# Act
	m, n = discretisation(spectrum, m_range).sample(n_sd)

	# Assert
	assert m.shape == n.shape
	assert n.shape == (n_sd,)
	assert np.min(m) >= m_range[0]
	assert np.max(m) <= m_range[1]
	actual = np.sum(n)
	desired = spectrum.cumulative(m_range[1]) - spectrum.cumulative(m_range[0])
	quotient = actual / desired
	np.testing.assert_almost_equal(actual=quotient, desired=1.0, decimal=2)
