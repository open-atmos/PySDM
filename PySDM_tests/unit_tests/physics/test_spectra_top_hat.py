from PySDM.physics import spectra
from matplotlib import pylab
import numpy as np


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
            pylab.axhline(0)
            pylab.axhline(norm_factor)
            pylab.axvline(endpoints[0])
            pylab.axvline(endpoints[1])
            pylab.plot(x, y, color='red')
            pylab.xlim(x[0], x[-1])
            pylab.grid()
            pylab.show()

        # assert
        for point in y:
            assert 0 <= point <= norm_factor
        assert y[-1] == norm_factor

        hy, hx = np.histogram(np.diff(y) / np.diff(x))
        assert hx[0] == 0
        np.testing.assert_approx_equal(hx[-1], norm_factor / (endpoints[1] - endpoints[0]))
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
            pylab.axvline(0)
            pylab.axvline(1)
            pylab.axhline(spectrum.endpoints[0])
            pylab.axhline(spectrum.endpoints[1])
            pylab.plot(x, y, color='red')
            pylab.show()

        # assert
        assert y[0] == spectrum.endpoints[0]
        assert y[-1] == spectrum.endpoints[1]
        np.testing.assert_allclose(
            np.diff(y) / np.diff(x),
            spectrum.endpoints[1] - spectrum.endpoints[0]
        )
