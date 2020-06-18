"""
Created at 10.06.2020
"""

from PySDM.physics import constants as const
from PySDM.attributes.droplet.terminal_velocity.gunn_and_kinzer import RogersYau, Interpolation, TpDependent
import matplotlib.pyplot as plt
import numpy as np

from PySDM.backends.default import Default
from PySDM_tests.unit_tests.state.dummy_particles import DummyParticles


def test_approximation(plot=False):
    # TODO: rethink
    pass
    # r = np.array([.078, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.4, 1.6]) * const.si.mm / 2
    # u = np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565]) / 100
    # n_sd = len(r)
    # particles = DummyParticles(Default, n_sd=n_sd)
    # u_term = particles.backend.array((n_sd,), float)
    # radius = np.linspace(4e-6, 200e-6, 1000, endpoint=True)
    # RogersYau(particles)(u_term, radius)
    # u_term_2 = np.copy(u_term)
    # Interpolation(None)(u_term_2, radius)
    #
    # r, u = r[:5], u[:5]
    # if plot:
    #     plt.plot(radius, u_term)
    #     plt.plot(radius, u_term_2)
    #     plt.plot(r, u)
    #     plt.grid()
    #     plt.show()