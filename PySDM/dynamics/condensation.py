"""
Created at 24.10.2019
"""

import numpy as np


default_rtol_x = 1e-8
default_rtol_thd = 1e-8


class Condensation:

    def __init__(self, kappa,
                 rtol_x=default_rtol_x,
                 rtol_thd=default_rtol_thd,
                 coord='volume logarithm', adaptive=True,
                 ):
        self.core = None
        self.kappa = kappa
        self.rtol_x = rtol_x
        self.rtol_thd = rtol_thd
        self.r_cr = None

        self.max_substeps = None
        self.substeps = None
        self.ripening_flags = None

        self.coord = coord
        self.adaptive = adaptive

    def register(self, builder):
        self.core = builder.core
        builder._set_condensation_parameters(self.coord, self.adaptive)
        self.r_cr = builder.get_attribute('critical radius')
        self.max_substeps = int(self.core.dt)
        self.max_substeps = int(self.core.dt)
        self.substeps = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.substeps[:] = int(np.maximum(1, int(self.core.dt)))  # TODO: min substep length
        self.ripening_flags = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.ripening_flags[:] = 0

    def __call__(self):
        self.core.condensation(
            kappa=self.kappa,
            rtol_x=self.rtol_x,
            rtol_thd=self.rtol_thd,
            substeps=self.substeps,
            ripening_flags=self.ripening_flags
        )
        self.substeps[:] = np.maximum(self.substeps[:], int(self.core.dt))
