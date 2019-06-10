from SDM.discretisations import *
from SDM.spectra import Lognormal
import numpy as np
import pytest


@pytest.mark.parametrize("discretisation", [
	pytest.param(linear),
	pytest.param(logarithmic)
])
def test_discretisation(discretisation):
	# Arrange
	n_sd = 64
	m_mode = 2e-6
	n_part = 32 * n_sd
	s_geom = 1.2
	spectrum = Lognormal(m_mode, n_part, s_geom)
	m_range = (0, 1000)

	# Act
	m, n = discretisation(n_sd, spectrum, m_range)

	# Assert
	assert m.shape == n.shape == (n_sd,)
	print(m)
	assert np.min(m) >= m_range[0]
	assert np.max(m) <= m_range[1]


def test_linear():
	# Arrange
	pass

	# Act

	# Assert


def test_logarithmic():
	# Arrange
	pass

	# Act

	# Assert
