"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from SDM.simulation.spectra import Exponential
from SDM.simulation.kernels import Golovin
from SDM.backends.default import Default


def x2r(x):
    return (x * 3 / 4 / np.pi) ** (1 / 3)


def r2x(r):
    return 4 / 3 * np.pi * r ** 3


kg2g = 1e3
m2um = 1e6


class Setup:  # TODO pint
    #grid = (75, 75)  # dx=dz=20m
    grid = (25,25)
    size = (1500, 1500)  # [m]

    field_values = {'th': 300,
                    'qv': 10e-3}

    def stream_function(self, x, z):
        w_max = .6
        X = self.size[0]
        Z = self.size[1]
        return - w_max * X / np.pi * np.sin(np.pi * z / Z) * np.cos (2 * np.pi * x / X)

    x_min = r2x(10e-6)  # not given in the paper
    x_max = r2x(100e-6)  # not given in the paper

    n_sd = grid[0] * grid[1] * 2
    n_part = 2 ** 23  # [m-3]
    X0 = 4 / 3 * np.pi * 30.531e-6 ** 3
    dv = size[0] / grid[0] * size[1] / grid[1]  # [m3]
    norm_factor = n_part * dv
    rho = 1000  # [kg m-3]

    dt = 1  # [s]

    steps = np.arange(0, 50000, 100)

    kernel = Golovin(b=1.5e3)  # [s-1]
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    backend = Default()

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