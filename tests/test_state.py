from SDM.state import State
from SDM.discretisations import Linear
from SDM.spectra import Lognormal
from SDM import debug

from numpy.testing import assert_almost_equal

class TestState:
	def test_moment(self):
		# Arrange (parameters from Clark 1976)
		n_part = 1000# 190 # cm-3 # TODO!!!!
		r_mean = 0#6.0 # um    # TODO: geom mean?
		d = 0.2 # dimensionless -> geom. standard dev

		r_min = 0.1
		r_max = 100
		n_sd = 16

		spectrum = Lognormal(n_part, r_mean, d)
		discretise = Linear(r_min, r_max)
		sut = State(n_sd, discretise, spectrum)

		debug.plot(sut)

		# Act
		mom_discr = sut.moment(1)

		# Assert
		mom_true = spectrum.stats(moments='m')
		assert abs(mom_discr - mom_true) / mom_true < 1e-1
		#assert_almost_equal(mom_discr, mom_true, decimal=100)
