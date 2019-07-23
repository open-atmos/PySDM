"""
Created at 09.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.spectra import Exponential
from SDM.kernels import Golovin
from examples.Shima_et_al_2009_Fig2.utils import *
import numpy as np


class SetupA:
    x_min = r2x(10e-6)   # not given in the paper
    x_max = r2x(100e-6)  # not given in the paper

    n_sd = 2 ** 13
    n_part = 2 ** 23  # [m-3]
    X0 = 4/3 * np.pi * 30.531e-6**3
    dv = 1e6  # [m3]
    norm_factor = n_part * dv
    rho = 1000  # [kg m-3]

    dt = 1  # [s]
    steps = [0, 1200, 2400, 3600]

    kernel = Golovin(b=1.5e3)  # [s-1]
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    # TODO: rename?
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
