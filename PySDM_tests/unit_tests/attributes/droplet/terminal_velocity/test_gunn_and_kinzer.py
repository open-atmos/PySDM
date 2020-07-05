"""
Created at 10.06.2020
"""

from PySDM.physics import constants as const
from PySDM.attributes.droplet.terminal_velocity.gunn_and_kinzer import RogersYau, Interpolation, TpDependent
import matplotlib.pyplot as plt
import numpy as np

from PySDM.backends.default import Default
from PySDM_tests.unit_tests.state.dummy_core import DummyCore


def test_approximation(plot=False):
    r = np.array([.078, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.4, 1.6]) * const.si.mm / 2
    u = np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565]) / 100
    n_sd = len(r)
    particles = DummyCore(Default, n_sd=n_sd)
    # radius = np.linspace(4e-6, 200e-6, 1000, endpoint=True)

    u_term_ry = particles.backend.array((len(u),), float)
    RogersYau(particles)(u_term_ry, r)

    u_term_inter = np.copy(u_term_ry)
    Interpolation(particles)(u_term_inter, r)

    assert np.mean((u - u_term_ry)**2) < 2e-2
    assert np.mean((u - u_term_inter) ** 2) < 1e-6

    if plot:
        # r, u = r[:5], u[:5]
        plt.plot(r, u_term_ry)
        plt.plot(r, u_term_inter)
        plt.plot(r, u)
        plt.grid()
        plt.show()
