"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.initialisation.spectra import Lognormal
from PySDM.dynamics.coalescence.kernels import Gravitational
from PySDM.backends.default import Default
from PySDM.dynamics.condensation import condensation
from PySDM.physics import formulae as phys
from PySDM.physics import constants as const
from PySDM.physics.constants import si


class Setup:
    backend = Default
    condensation_coord = 'volume'

    condensation_rtol_x = condensation.default_rtol_x
    condensation_rtol_thd = condensation.default_rtol_thd
    adaptive = True

    grid = (25, 25)
    size = (1500 * si.metres, 1500 * si.metres)
    n_sd_per_gridbox = 20
    rho_w_max = .6 * si.metres / si.seconds * (si.kilogram / si.metre ** 3)

    # output steps
    n_steps = 3600
    outfreq = 60
    dt = 1 * si.seconds

    v_bins = phys.volume(np.logspace(np.log10(0.01 * si.micrometre), np.log10(100 * si.micrometre), 101, endpoint=True))

    @property
    def steps(self):
        return np.arange(0, self.n_steps+1, self.outfreq)

    # TODO: second mode
    spectrum_per_mass_of_dry_air = Lognormal(
      norm_factor=40 / si.centimetre**3 / const.rho_STP,
      m_mode=0.15 * si.micrometre,
      s_geom=1.6
    )

    processes = {
        "particle advection": True,
        "fluid advection": True,
        "coalescence": False,
        "condensation": True,
        "sedimentation": False,
        # "relaxation": False  # TODO
    }

    enable_particle_temperatures = False

    mpdata_iters = 2
    mpdata_iga = True
    mpdata_fct = True
    mpdata_tot = True

    th_std0 = 289 * si.kelvins
    qv0 = 7.5 * si.grams / si.kilogram
    p0 = 1015 * si.hectopascals
    kappa = 1

    @property
    def field_values(self):
        return {
            'th': phys.th_dry(self.th_std0, self.qv0),
            'qv': self.qv0
        }

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * self.n_sd_per_gridbox

    def stream_function(self, xX, zZ):
        X = self.size[0]
        return - self.rho_w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)

    def rhod(self, zZ):
        Z = self.size[1]
        z = zZ * Z  # :)

        # hydrostatic profile
        kappa = const.Rd / const.c_pd
        arg = np.power(self.p0/const.p1000, kappa) - z * kappa * const.g / self.th_std0 / phys.R(self.qv0)
        p = const.p1000 * np.power(arg, 1/kappa)

        # np.testing.assert_array_less(p, Setup.p0)  # TODO: less or equal

        # density using "dry" potential temp.
        pd = p * (1 - self.qv0 / (self.qv0 + const.eps))
        rhod = pd / (np.power(p / const.p1000, kappa) * const.Rd * self.th_std0)

        return rhod

    # initial dry radius discretisation range
    r_min = .01 * si.micrometre
    r_max = 5 * si.micrometre

    kernel = Gravitational(collection_efficiency=1)  # [s-1]  # TODO!
    aerosol_radius_threshold = 1 * si.micrometre

    n_spin_up = 1 * si.hour / dt
