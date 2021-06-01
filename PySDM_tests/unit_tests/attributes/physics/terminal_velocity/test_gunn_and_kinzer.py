import matplotlib.pyplot as plt
import numpy as np

from PySDM.physics import constants as const
from PySDM.physics.terminal_velocity.gunn_and_kinzer import RogersYau, Interpolation

# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM_tests.unit_tests.dummy_core import DummyCore


def test_approximation(backend, plot=False):
    r = np.array([.078, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.4, 1.6]) * const.si.mm / 2
    r = backend.Storage.from_ndarray(r)
    u = np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565]) / 100
    n_sd = len(r)
    core = DummyCore(backend, n_sd=n_sd)
    # radius = np.linspace(4e-6, 200e-6, 1000, endpoint=True)

    u_term_ry = core.backend.Storage.empty((len(u),), float)
    RogersYau(core)(u_term_ry, r)

    u_term_inter = backend.Storage.from_ndarray(u_term_ry.to_ndarray())
    Interpolation(core)(u_term_inter, r)

    assert np.mean((u - u_term_ry)**2) < 2e-2
    assert np.mean((u - u_term_inter) ** 2) < 1e-6

    if plot:
        # r, u = r[:5], u[:5]
        plt.plot(r, u_term_ry)
        plt.plot(r, u_term_inter)
        plt.plot(r, u)
        plt.grid()
        plt.show()
