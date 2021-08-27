from PySDM.initialisation import spectral_sampling, spectro_glacial
from PySDM.physics.spectra import Lognormal
from PySDM.physics import Formulae, constants as const
import numpy as np
import pytest

m_mode = .5e-5
n_part = 256 * 16
s_geom = 1.5
spectrum = Lognormal(n_part, m_mode, s_geom)
m_range = (.1e-6, 100e-6)
formulae = Formulae()

@pytest.mark.parametrize("discretisation", [
	pytest.param(spectral_sampling.Linear(spectrum, m_range)),
	pytest.param(spectral_sampling.Logarithmic(spectrum, m_range)),
	pytest.param(spectral_sampling.ConstantMultiplicity(spectrum, m_range)),
	pytest.param(spectral_sampling.UniformRandom(spectrum, m_range)),
	pytest.param(spectro_glacial.Independent(
		size_spectrum=spectrum,
		freezing_temperature_spectrum=formulae.freezing_temperature_spectrum
	))
])
def test_spectral_discretisation(discretisation):
	# Arrange
	n_sd = 10000

	# Act
	if isinstance(discretisation, spectro_glacial.SpectroGlacialSampling):
		m, _, n = discretisation.sample(n_sd)
	else:
		m, n = discretisation.sample(n_sd)

	# Assert
	assert m.shape == n.shape
	assert n.shape == (n_sd,)
	assert np.min(m) >= m_range[0]
	assert np.max(m) <= m_range[1]
	actual = np.sum(n)
	desired = spectrum.cumulative(m_range[1]) - spectrum.cumulative(m_range[0])
	quotient = actual / desired
	np.testing.assert_approx_equal(actual=quotient, desired=1.0, significant=2)
