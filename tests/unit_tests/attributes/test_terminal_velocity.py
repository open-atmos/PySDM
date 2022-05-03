# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np

from PySDM.dynamics.terminal_velocity import Interpolation, RogersYau
from PySDM.physics import constants as const
from tests.backends_fixture import backend_class
from tests.unit_tests.dummy_particulator import DummyParticulator

assert hasattr(backend_class, "_pytestfixturefunction")


# pylint: disable=redefined-outer-name
def test_approximation(backend_class, plot=False):
    r = (
        np.array(
            [0.078, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
        )
        * const.si.mm
        / 2
    )
    particulator = DummyParticulator(backend_class, n_sd=len(r))
    r = particulator.backend.Storage.from_ndarray(r)
    u = (
        np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565])
        / 100
    )

    u_term_ry = particulator.backend.Storage.empty((len(u),), float)
    RogersYau(particulator=particulator)(u_term_ry, r)

    u_term_inter = particulator.backend.Storage.from_ndarray(u_term_ry.to_ndarray())
    Interpolation(particulator)(u_term_inter, r)

    assert np.mean((u - u_term_ry) ** 2) < 2e-2
    assert np.mean((u - u_term_inter) ** 2) < 1e-6

    if plot:
        plt.plot(r, u_term_ry)
        plt.plot(r, u_term_inter)
        plt.plot(r, u)
        plt.grid()
        plt.show()
