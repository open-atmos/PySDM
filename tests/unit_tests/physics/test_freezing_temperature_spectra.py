# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import constants as const
from PySDM.physics import si

A = 1 * si.um**2


@pytest.mark.parametrize("model", ("Niemand_et_al_2012", "Bigg_1953"))
def test_freezing_temperature_spectra(model, plot=False):
    # Arrange
    formulae = Formulae(
        freezing_temperature_spectrum=model,
        constants={"NIEMAND_A": -0.517, "NIEMAND_B": 8.934, "BIGG_DT_MEDIAN": 33},
    )
    temperature = np.linspace(const.T0, const.T0 - 40, num=100)

    # Act
    pdf = formulae.freezing_temperature_spectrum.pdf(temperature, A)
    cdf = formulae.freezing_temperature_spectrum.cdf(temperature, A)

    # Plot
    pyplot.plot(temperature, pdf, linestyle="-", marker="o", label="pdf")
    pdfax = pyplot.gca()
    cdfax = pdfax.twinx()
    cdfax.plot(temperature, cdf, linestyle="--", marker="x", label="cdf")
    pyplot.xlabel("T [K]")
    pyplot.xlim(np.amax(temperature), np.amin(temperature))
    pdfax.set_ylabel("pdf [K$^{-1}$]")
    cdfax.set_ylabel("cdf [1]")
    pyplot.grid()
    pyplot.title(model)
    if plot:
        pyplot.show()

    # Assert
    dT = abs(temperature[1] - temperature[0])
    np.testing.assert_approx_equal(np.sum(pdf * dT), 1, significant=3)
    np.testing.assert_approx_equal(cdf[0] + 1, 1, significant=3)
    np.testing.assert_approx_equal(cdf[-1], 1, significant=3)

    if hasattr(formulae.freezing_temperature_spectrum, "invcdf"):
        invcdf = formulae.freezing_temperature_spectrum.invcdf(cdf, A)
        np.testing.assert_allclose(invcdf, temperature)
