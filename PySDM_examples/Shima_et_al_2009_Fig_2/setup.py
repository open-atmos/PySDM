"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.initialisation.spectra import Exponential
from PySDM.dynamics import Golovin
from PySDM.backends import Default
from PySDM.physics.constants import si
from PySDM.physics import formulae as phys


class SetupA:
    init_x_min = phys.volume(radius=10 * si.micrometres)  # not given in the paper
    init_x_max = phys.volume(radius=100 * si.micrometres)  # not given in the paper

    n_sd = 2 ** 13
    n_part = 2 ** 23 / si.metre**3
    X0 = 4 / 3 * np.pi * (30.531 * si.micrometres) ** 3
    dv = 1e6 * si.metres**3
    norm_factor = n_part * dv
    rho = 1000 * si.kilogram / si.metre**3
    dt = 1 * si.seconds
    seed = 44
    steps = [0, 1200, 2400, 3600]

    kernel = Golovin(b=1.5e3 / si.second)
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    radius_bins_edges = np.logspace(np.log10(10 * si.um), np.log10(5e3 * si.um), num=64, endpoint=True)

    backend = Default
