"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.initialisation.spectra import Lognormal
from PySDM.simulation.dynamics.coalescence.kernels.golovin import Golovin
from PySDM.backends.default import Default

from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
from PySDM.simulation.physics.constants import si


class Setup:
    backend = Default
    grid = (25, 25)  # dx=dz=20m
    size = (1500 * si.metres, 1500 * si.metres)
    n_sd_per_gridbox = 20
    dt = 0.25 * si.seconds  # TODO: was 1s in the ICMW case?
    w_max = .6 * si.metres / si.seconds

    # TODO: second mode
    spectrum_per_mass_of_dry_air = Lognormal(
      norm_factor=40 / si.centimetre**3 / const.rho_STP,
      m_mode=0.15 * si.micrometre,
      s_geom=1.6
    )

    processes = {
        "advection": True,
        "coalescence": True,
        "condensation": True
    }

    th0 = 289 * si.kelvins
    qv0 = 7.5 * si.grams / si.kilogram
    p0 = 1015 * si.hectopascals
    rho = 1000 * si.kilogram / si.metre**3
    kappa = 1

    field_values = {
        'th': phys.th_dry(th0, qv0),
        'qv': qv0
    }

    # TODO: potentially to be removed as already present in particles
    @property
    def dx(self):
        return self.size[0] / self.grid[0]

    @property
    def dz(self):
        return self.size[1] / self.grid[1]

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * self.n_sd_per_gridbox

    @property
    def eddy_period(self):
        raise NotImplementedError()

    def stream_function(self, xX, zZ):
        X = self.size[0]
        return - self.w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)

    def rhod(self, zZ):
        Z = self.size[1]
        z = zZ * Z  # :)

        # hydrostatic profile
        kappa = const.Rd / const.c_pd
        arg = np.power(self.p0/const.p1000, kappa) - z*kappa*const.g/self.th0/phys.R(self.qv0)
        p = const.p1000 * np.power(arg, 1/kappa)

        #np.testing.assert_array_less(p, Setup.p0) TODO: less or equal

        # density using "dry" potential temp.
        pd = p * (1 - self.qv0 / (self.qv0 + const.eps))
        rhod = pd / (np.power(p / const.p1000, kappa) * const.Rd * self.th0)

        return rhod

    # initial dry radius discretisation range
    r_min = .01 * si.micrometre
    r_max = 5 * si.micrometre


    # output steps
    steps = np.arange(0, 3600, 30)

    kernel = Golovin(b=1e-3)  # [s-1]

    specs = {'x': (1, 1/3)}
    output_vars = ["m0", "th", "qv", "RH", "x_m1"]  # TODO: add in a loop over specs
