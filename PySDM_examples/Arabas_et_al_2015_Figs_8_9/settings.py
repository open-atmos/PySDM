"""
Created at 25.09.2019
"""

from typing import Iterable

import numba
import numpy
import numpy as np
import scipy
from pystrict import strict

import PySDM
from PySDM.dynamics import condensation
from PySDM.dynamics.coalescence import coalescence
from PySDM.dynamics.coalescence.kernels import Geometric
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.spectra import Sum
from PySDM.physics import constants as const
from PySDM.physics import formulae as phys
from PySDM.physics.constants import si


# from PyMPDATA import __version__ as TODO #339

@strict
class Settings:
    def __dir__(self) -> Iterable[str]:
        return 'dt', 'grid', 'size', 'n_spin_up', 'versions', 'steps_per_output_interval'

    def __init__(self):
        key_packages = (PySDM, numba, numpy, scipy)
        self.versions = str({pkg.__name__: pkg.__version__ for pkg in key_packages})

        self.condensation_coord = 'volume logarithm'

        self.condensation_rtol_x = condensation.default_rtol_x
        self.condensation_rtol_thd = condensation.default_rtol_thd
        self.condensation_adaptive = True
        self.condensation_substeps = -1
        self.condensation_dt_cond_range = condensation.default_cond_range
        self.condensation_schedule = condensation.default_schedule

        self.coalescence_adaptive = True
        self.coalescence_dt_coal_range = coalescence.default_dt_coal_range
        self.coalescence_optimized_random = True
        self.coalescence_substeps = 1

        self.grid = (25, 25)
        self.size = (1500 * si.metres, 1500 * si.metres)
        self.n_sd_per_gridbox = 20
        self.rho_w_max = .6 * si.metres / si.seconds * (si.kilogram / si.metre ** 3)

        # output steps
        self.simulation_time = 90 * si.minute
        self.output_interval = 1 * si.minute
        self.dt = 5 * si.second
        self.spin_up_time = 1 * si.hour

        self.v_bins = phys.volume(np.logspace(np.log10(0.001 * si.micrometre), np.log10(100 * si.micrometre), 101, endpoint=True))

        self.mode_1 = Lognormal(
            norm_factor=60 / si.centimetre ** 3 / const.rho_STP,
            m_mode=0.04 * si.micrometre,
            s_geom=1.4
        )
        self.mode_2 = Lognormal(
          norm_factor=40 / si.centimetre**3 / const.rho_STP,
          m_mode=0.15 * si.micrometre,
          s_geom=1.6
        )
        self.spectrum_per_mass_of_dry_air = Sum((self.mode_1, self.mode_2))

        self.processes = {
            "particle advection": True,
            "fluid advection": True,
            "coalescence": True,
            "condensation": True,
            "sedimentation": True,
            # "relaxation": False  # TODO #338
        }

        self.enable_particle_temperatures = False

        self.mpdata_iters = 2
        self.mpdata_iga = True
        self.mpdata_fct = True
        self.mpdata_tot = True

        self.th_std0 = 289 * si.kelvins
        self.qv0 = 7.5 * si.grams / si.kilogram
        self.p0 = 1015 * si.hectopascals
        self.kappa = 1  # TODO #441!
        self.g = const.g_std
        self.kernel = Geometric(collection_efficiency=1)
        self.aerosol_radius_threshold = .5 * si.micrometre
        self.drizzle_radius_threshold = 25 * si.micrometre

    @property
    def n_steps(self) -> int:
        return int(self.simulation_time / self.dt)  # TODO #413

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def n_spin_up(self) -> int:
        return int(self.spin_up_time / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.n_steps + 1, self.steps_per_output_interval)

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
        p = phys.Hydrostatic.p_of_z_assuming_const_th_and_qv(self.g, self.p0, self.th_std0, self.qv0, z=zZ * self.size[-1])
        rhod = phys.ThStd.rho_d(p, self.qv0, self.th_std0)
        return rhod
