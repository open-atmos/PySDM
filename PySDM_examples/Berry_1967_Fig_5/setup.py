"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.initialisation.spectra import Exponential
from PySDM.dynamics.coalescence.kernels import Golovin
from PySDM.backends import Default
from PySDM.physics.constants import si
from PySDM.physics import formulae as phys


class Setup:
    init_x_min = phys.volume(radius=3.94 * si.micrometre)
    init_x_max = phys.volume(radius=25 * si.micrometres)

    n_sd = 2 ** 13
    n_part = 239 / si.cm**3
    X0 = 4 / 3 * np.pi * (10 * si.micrometres) ** 3
    dv = 1e6 * si.metres**3
    norm_factor = n_part * dv
    rho = 1000 * si.kilogram / si.metre**3
    dt = 0.5 * si.seconds
    seed = 44
    steps = [200 * i for i in range(10)]

    kernel = Golovin(b=1.5e3 / si.second)
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    radius_bins_edges = np.logspace(np.log10(3.94 * si.um), np.log10(200 * si.um), num=128, endpoint=True)

    backend = Default
