from PySDM.physics import Formulae, constants as const
from matplotlib import pylab
import numpy as np


def test_freezing_temperature_spectra(plot=False):
    # Arrange
    formulae = Formulae()
    T = np.linspace(const.T0 - 40, const.T0, num=100)

    # Act
    p = formulae.freezing_temperature_spectrum.pdf(T)

    # Plot
    if plot:
        pylab.plot(T, p, linestyle='-', marker='o')
        pylab.xlabel('T [K]')
        pylab.ylabel('pdf [K$^{-1}$]')
        pylab.show()

    # Assert
    dT = T[1] - T[0]
    np.testing.assert_approx_equal(np.sum(p * dT), 1)
