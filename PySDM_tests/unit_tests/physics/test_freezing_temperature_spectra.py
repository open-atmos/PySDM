from PySDM.physics import Formulae, constants as const
from matplotlib import pylab
import numpy as np


def test_freezing_temperature_spectra(plot=False):
    # Arrange
    formulae = Formulae()
    T = np.linspace(const.T0, const.T0 - 40, num=100)

    # Act
    pdf = formulae.freezing_temperature_spectrum.pdf(T)
    cdf = formulae.freezing_temperature_spectrum.cdf(T)

    # Plot
    if plot:
        pylab.plot(T, pdf, linestyle='-', marker='o', label='pdf')
        pdfax = pylab.gca()
        cdfax = pdfax.twinx()
        cdfax.plot(T, cdf, linestyle='--', marker='x', label='cdf')
        pylab.xlabel('T [K]')
        pylab.xlim(np.amax(T), np.amin(T))
        pdfax.set_ylabel('pdf [K$^{-1}$]')
        cdfax.set_ylabel('cdf [1]')
        pylab.grid()
        pylab.show()

    # Assert
    dT = abs(T[1] - T[0])
    np.testing.assert_approx_equal(np.sum(pdf * dT), 1)
    np.testing.assert_approx_equal(cdf[0]+1, 1)
    np.testing.assert_approx_equal(cdf[-1], 1)
