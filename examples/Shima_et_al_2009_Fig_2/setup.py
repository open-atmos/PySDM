"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.initialisation.spectra import Exponential
from PySDM.simulation.dynamics.coalescence.kernels.golovin import Golovin
from PySDM.backends.default import Default
from PySDM.simulation.physics.constants import si
from PySDM.simulation.physics import formulae as phys


class SetupA:
    x_min = phys.volume(radius=10 * si.micrometres)  # not given in the paper
    x_max = phys.volume(radius=100 * si.micrometres)  # not given in the paper

    n_sd = 2 ** 13
    n_part = 2 ** 23 / si.metre**3
    X0 = 4 / 3 * np.pi * 30.531e-6 ** 3
    dv = 1e6 * si.metres**3
    norm_factor = n_part * dv
    rho = 1000 * si.kilogram / si.metre**3
    dt = 1 * si.seconds
    seed = 44
    steps = [0, 1200, 2400, 3600]

    kernel = Golovin(b=1.5e3 / si.second)
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    backend = Default

    # TODO: rename?
    # TODO: as backend method?
    def check(self, state, step):
        check_LWC = 1e-3  * si.kilogram / si.metre**3
        check_ksi = self.n_part * self.dv / self.n_sd

        # multiplicities
        if step == 0:
            np.testing.assert_approx_equal(np.amin(state['n']), np.amax(state['n']), 1)
            np.testing.assert_approx_equal(state['n'][0], check_ksi, 1)

        # liquid water content
        LWC = self.rho * np.dot(state['n'], state['volume']) / self.dv
        np.testing.assert_approx_equal(LWC, check_LWC, 3)