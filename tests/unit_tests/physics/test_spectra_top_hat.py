# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot

from PySDM.initialisation import spectra


class TestSpectraTopHat:
    @staticmethod
    def test_cumulative(plot=False):
        # arrange
        endpoints = (44, 666)
        norm_factor = np.pi
        spectrum = spectra.TopHat(norm_factor=norm_factor, endpoints=endpoints)
        x = np.linspace(0, 1000)

        # act
        y = spectrum.cumulative(x)

        # plot
        if plot:
            pyplot.axhline(0)
            pyplot.axhline(norm_factor)
            pyplot.axvline(endpoints[0])
            pyplot.axvline(endpoints[1])
            pyplot.plot(x, y, color="red")
            pyplot.xlim(x[0], x[-1])
            pyplot.grid()
            pyplot.show()

        # assert
        for point in y:
            assert 0 <= point <= norm_factor
        assert y[-1] == norm_factor

        hy, hx = np.histogram(np.diff(y) / np.diff(x))
        assert hx[0] == 0
        np.testing.assert_approx_equal(
            hx[-1], norm_factor / (endpoints[1] - endpoints[0])
        )
        assert np.sum(hy[1:-1]) == 2

    @staticmethod
    def test_percentiles(plot=False):
        # arrange
        spectrum = spectra.TopHat(norm_factor=np.e, endpoints=(-np.pi, np.pi))
        x = np.linspace(0, 1)

        # act
        y = spectrum.percentiles(x)

        # plot
        if plot:
            pyplot.axvline(0)
            pyplot.axvline(1)
            pyplot.axhline(spectrum.endpoints[0])
            pyplot.axhline(spectrum.endpoints[1])
            pyplot.plot(x, y, color="red")
            pyplot.show()

        # assert
        assert y[0] == spectrum.endpoints[0]
        assert y[-1] == spectrum.endpoints[1]
        np.testing.assert_allclose(
            np.diff(y) / np.diff(x), spectrum.endpoints[1] - spectrum.endpoints[0]
        )
