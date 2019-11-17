"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.initialisation.spectra import Exponential
from PySDM.simulation.dynamics.coalescence.kernels.golovin import Golovin
from PySDM.backends.default import Default


def x2r(x):
    return (x * 3 / 4 / np.pi) ** (1 / 3)


def r2x(r):
    return 4 / 3 * np.pi * r ** 3


kg2g = 1e3
m2um = 1e6


class SetupA:
    x_min = r2x(10e-6)  # not given in the paper
    x_max = r2x(100e-6)  # not given in the paper

    n_sd = 2 ** 13
    n_part = 2 ** 23  # [m-3]
    X0 = 4 / 3 * np.pi * 30.531e-6 ** 3
    dv = 1e6  # [m3]
    norm_factor = n_part * dv
    rho = 1000  # [kg m-3]

    dt = 1  # [s]

    steps = [0, 1200, 2400, 3600]

    kernel = Golovin(b=1.5e3)  # [s-1]
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    backend = Default

    # TODO: rename?
    # TODO: as backend method?
    def check(self, state, step):
        check_LWC = 1e-3  # kg m-3
        check_ksi = self.n_part * self.dv / self.n_sd

        # multiplicities
        if step == 0:
            np.testing.assert_approx_equal(np.amin(state['n']), np.amax(state['n']), 1)
            np.testing.assert_approx_equal(state['n'][0], check_ksi, 1)

        # liquid water content
        LWC = self.rho * np.dot(state['n'], state['x']) / self.dv
        np.testing.assert_approx_equal(LWC, check_LWC, 3)