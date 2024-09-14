""" unit tests for bulk phase partitioning formulae """

import numpy as np
from matplotlib import pyplot
from PySDM import Formulae
from PySDM.physics import si


def test_bulk_phase_partitioning(plot=False):
    """hello-world plot for bulk phase partitioning as formulated in Kaul et al. 2015"""
    # arrange
    sut = Formulae(
        bulk_phase_partitioning="KaulEtAl2015",
        constants={"bulk_phase_partitioning_exponent": 0.1},
    ).bulk_phase_partitioning.liquid_fraction
    T = np.linspace(start=200, stop=300, num=100) * si.K

    # act
    fl = sut(T)

    # plot
    pyplot.plot(T, fl)
    pyplot.xlabel("temperature [K]")
    pyplot.ylabel("liquid fraction [1]")
    pyplot.grid()
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    assert np.isfinite(fl).all()
    assert np.amin(fl) == 0
    assert np.amax(fl) == 1
    assert (np.diff(fl) >= 0).all()
