import numpy as np
import pytest
from matplotlib import pylab
from PySDM.physics import Formulae, si, constants as const
from PySDM.physics.freezing_temperature_spectrum import niemand_et_al_2012, bigg_1953


A = 1 * si.um**2


@pytest.mark.parametrize("model", (
        'Niemand_et_al_2012',
        'Bigg_1953'
))
def test_freezing_temperature_spectra(model, plot=False):
    # Arrange
    bigg_1953.DT_median = 33
    niemand_et_al_2012.a = -0.517
    niemand_et_al_2012.b = 8.934
    niemand_et_al_2012.A_insol = 1 * si.um ** 2

    formulae = Formulae(freezing_temperature_spectrum=model)
    T = np.linspace(const.T0, const.T0 - 40, num=100)

    # Act
    pdf = formulae.freezing_temperature_spectrum.pdf(T, A)
    cdf = formulae.freezing_temperature_spectrum.cdf(T, A)

    # Plot
    pylab.plot(T, pdf, linestyle='-', marker='o', label='pdf')
    pdfax = pylab.gca()
    cdfax = pdfax.twinx()
    cdfax.plot(T, cdf, linestyle='--', marker='x', label='cdf')
    pylab.xlabel('T [K]')
    pylab.xlim(np.amax(T), np.amin(T))
    pdfax.set_ylabel('pdf [K$^{-1}$]')
    cdfax.set_ylabel('cdf [1]')
    pylab.grid()
    pylab.title(model)
    if plot:
        pylab.show()

    # Assert
    dT = abs(T[1] - T[0])
    np.testing.assert_approx_equal(np.sum(pdf * dT), 1, significant=3)
    np.testing.assert_approx_equal(cdf[0]+1, 1, significant=3)
    np.testing.assert_approx_equal(cdf[-1], 1, significant=3)
