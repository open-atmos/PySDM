"""
Created at 24.10.2019
"""

import numpy as np
from PySDM.physics import si

default_rtol_x = 1e-6
default_rtol_thd = 1e-6
default_dt_cond_range = (0 * si.second, 1 * si.second)


class Condensation:

    def __init__(self,
                 kappa,
                 rtol_x=default_rtol_x,
                 rtol_thd=default_rtol_thd,
                 coord='volume logarithm',
                 substeps: int = 1,
                 adaptive: bool = True,
                 dt_cond_range=default_dt_cond_range
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
        self.n_substep = None
        self.dt_cond_range = dt_cond_range

    def register(self, builder):
        self.core = builder.core
        builder._set_condensation_parameters(self.coord, self.adaptive)
        self.r_cr = builder.get_attribute('critical radius')

        self.n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.n_substep[:] = self.__substeps

        self.ripening_flags = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.ripening_flags[:] = 0

        self.RH_max = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)

    def __call__(self):
        if self.enable:
            self.core.condensation(
                kappa=self.kappa,
                rtol_x=self.rtol_x,
                rtol_thd=self.rtol_thd,
                substeps=self.n_substep,
                ripening_flags=self.ripening_flags,
                RH_max=self.RH_max
            )
            if self.adaptive:
                self.n_substep[:] = np.maximum(self.n_substep[:], int(self.core.dt / self.dt_cond_range[1]))
                if self.dt_cond_range[0] != 0:
                    self.n_substep[:] = np.minimum(self.n_substep[:], int(self.core.dt / self.dt_cond_range[0]))
