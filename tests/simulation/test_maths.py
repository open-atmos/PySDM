"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.simulation.maths import Maths
from SDM.simulation.state import State
from SDM.simulation.spectra import Lognormal
from SDM.simulation.discretisations.spectral import linear
from SDM.backends.default import Default

backend = Default()


class TestMaths:
    @staticmethod
    def test_moment():
        # Arrange (parameters from Clark 1976)
        n_part = 10000  # 190 # cm-3 # TODO!!!!
        x_mean = 2e-6  # 6.0 # um    # TODO: geom mean?
        d = 1.2  # dimensionless -> geom. standard dev

        x_min = 0.01e-6
        x_max = 10e-6
        n_sd = 32

        spectrum = Lognormal(n_part, x_mean, d)
        x, n = linear(n_sd, spectrum, (x_min, x_max))
        state = State.state_0d(n=n, extensive={'x': x}, intensive={}, backend=backend)

        true_mean, true_var = spectrum.stats(moments='mv')

        # Act
        discr_zero = Maths.moment_0d(state, 0) / n_part
        discr_mean = Maths.moment_0d(state, 1) / n_part
        discr_mrsq = Maths.moment_0d(state, 2) / n_part

        # Assert
        assert abs(discr_zero - 1) / 1 < 1e-3

        assert abs(discr_mean - true_mean) / true_mean < .01e-1

        true_mrsq = true_var + true_mean**2
        assert abs(discr_mrsq - true_mrsq) / true_mrsq < .05e-1
