from SDM.state import State
from SDM.discretisations import linear
from SDM.spectra import Lognormal
from SDM import debug

from numpy.testing import assert_almost_equal


class TestState:

	# TODO: test copy() calls in ctor

	def test_moment(self):
		# Arrange (parameters from Clark 1976)
		n_part = 1000  # 190 # cm-3 # TODO!!!!
		mmean = 2e-6  # 6.0 # um    # TODO: geom mean?
		d = 1.2  # dimensionless -> geom. standard dev

		mmin = 0.01e-6
		mmax = 10e-6
		n_sd = 32

		spectrum = Lognormal(n_part, mmean, d)
		sut = State(*linear(n_sd, spectrum, (mmin, mmax)))

		#debug.plot(sut)
		true_mean, true_var = spectrum.stats(moments='mv')

		# Act
		discr_zero = sut.moment(0)
		discr_mean = sut.moment(1)
		discr_mrsq = sut.moment(2)

		# Assert
		assert discr_zero == 1

		assert abs(discr_mean - true_mean) / true_mean < .01e-1

		true_mrsq = true_var + true_mean**2
		assert abs(discr_mrsq - true_mrsq) / true_mrsq < .05e-1