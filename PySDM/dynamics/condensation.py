"""
Created at 24.10.2019
"""

import numpy as np
from ..physics import si

default_rtol_x = 1e-6
default_rtol_thd = 1e-6


class Condensation:

    def __init__(self,
                 kappa,
                 rtol_x=default_rtol_x,
                 rtol_thd=default_rtol_thd,
                 coord='volume logarithm',
                 substeps: int = 1,
                 adaptive: bool = True
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
        self.dt_cond_range = (0 * si.second, 1 * si.second)  # TODO!

    def register(self, builder):
        self.core = builder.core

        builder._set_condensation_parameters(self.coord, self.adaptive)
        self.r_cr = builder.get_attribute('critical radius')

        for counter in ('n_substeps', 'n_activating', 'n_deactivating', 'n_ripening'):
            self.counters[counter] = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
            if counter == 'n_substeps':
                self.counters[counter][:] = self.__substeps
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
                RH_max=self.RH_max
            )
            if self.adaptive:
                self.counters['n_substeps'][:] = np.maximum(self.counters['n_substeps'][:], int(self.core.dt / self.dt_cond_range[1]))
                if self.dt_cond_range[0] != 0:
                    self.counters['n_substeps'][:] = np.minimum(self.counters['n_substeps'][:], int(self.core.dt / self.dt_cond_range[0]))
