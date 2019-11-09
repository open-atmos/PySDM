"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.spectra import Lognormal
from PySDM.simulation.kernels.Golovin import Golovin
from PySDM.simulation.ambient_air.moist_air import MoistAir
from PySDM.backends.default import Default
from PySDM.simulation.phys import si, mgn, th_dry

class Setup:
    grid = (75, 75)  # (75, 75)  # dx=dz=20m
    size = (1500, 1500)  # [m]
    n_sd_per_gridbox = 20

    @property
    def dv(self):
        return self.size[0] / self.grid[0] * self.size[1] / self.grid[1]  # [m3]

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * self.n_sd_per_gridbox

    # TODO: second mode
    # TODO: number -> mass distribution
    spectrum = Lognormal(
      norm_factor=mgn(40 / si.centimetre**3 * size[0]*si.metre * size[1]*si.metre * 1*si.metre),
      m_mode=mgn(0.15 * si.micrometre),
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

    field_values = {'th': mgn(th_dry(th0, qv0)),
                    'qv': mgn(qv0)}

    def stream_function(self, xX, zZ):
        w_max = .6 * si.metres / si.seconds
        X = self.size[0] * si.metres
        return mgn(- w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX))

    def rhod(self, zZ):
        from PySDM.simulation.phys import R, Rd, c_pd, p1000, g, eps

        Z = self.size[1] * si.metres
        z = zZ * Z  # :)

        # hydrostatic profile
        kappa = Rd / c_pd
        arg = np.power(Setup.p0/p1000, kappa) - z*kappa*g/Setup.th0/R(Setup.qv0)
        p = p1000 * np.power(arg, 1/kappa)

        #np.testing.assert_array_less(p, Setup.p0) TODO: less or equal

        # density using "dry" potential temp.
        pd = p * (1 - Setup.qv0 / (Setup.qv0 + eps))
        rhod = pd / (np.power(p / p1000, kappa) * Rd * Setup.th0)

        return mgn(rhod)

    x_min = .01e-6  # TODO: mass!
    x_max = 5e-6  # TODO: mass!

    dt = 0.25  # [s] #TODO: was 1s in the ICMW case?

    # output steps
    steps = np.arange(0, 3600, 30)

    kernel = Golovin(b=1e-3)  # [s-1]

    backend = Default

    ambient_air = MoistAir
