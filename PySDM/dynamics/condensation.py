"""
Created at 24.10.2019
"""

import numpy as np
from ..physics import si

default_rtol_x = 1e-6
default_rtol_thd = 1e-6
default_cond_range = (1e-4 * si.second, 10 * si.second)
default_schedule = 'prev_substeps'


class Condensation:

    def __init__(self,
                 kappa,
                 rtol_x=default_rtol_x,
                 rtol_thd=default_rtol_thd,
                 coord='volume logarithm',
                 substeps: int = 1,
                 adaptive: bool = True,
                 dt_cond_range: tuple = default_cond_range,
                 schedule: str = default_schedule
                 ):

        self.core = None
        self.enable = True

        self.kappa = kappa
        self.rtol_x = rtol_x
        self.rtol_thd = rtol_thd
        self.r_cr = None

        self.ripening_flags = None
        self.RH_max = None
        self.coord = coord

        self.__substeps = substeps
        self.adaptive = adaptive
        self.counters = {}
        self.dt_cond_range = dt_cond_range
        self.schedule = schedule

    def register(self, builder):
        self.core = builder.core

        builder._set_condensation_parameters(self.coord, self.dt_cond_range, self.adaptive)
        self.r_cr = builder.get_attribute('critical radius')

        for counter in ('n_substeps', 'n_activating', 'n_deactivating', 'n_ripening'):
            self.counters[counter] = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
            if counter == 'n_substeps':
                self.counters[counter][:] = self.__substeps if not self.adaptive else -1
            else:
                self.counters[counter][:] = -1

        self.RH_max = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)
        self.RH_max[:] = np.nan

    def __call__(self):
        if self.enable:
            self.core.condensation(
                kappa=self.kappa,
                rtol_x=self.rtol_x,
                rtol_thd=self.rtol_thd,
                counters=self.counters,
                RH_max=self.RH_max,
                schedule=self.schedule
            )
