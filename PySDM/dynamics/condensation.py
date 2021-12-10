"""
Bespoke condensational growth solver with implicit-in-particle-size integration and adaptive timestepping
"""
import numpy as np
from ..physics import si

default_rtol_x = 1e-6
default_rtol_thd = 1e-6
default_cond_range = (1e-4 * si.second, 1 * si.second)
default_schedule = 'dynamic'


class Condensation:

    def __init__(self,
                 rtol_x=default_rtol_x,
                 rtol_thd=default_rtol_thd,
                 substeps: int = 1,
                 adaptive: bool = True,
                 dt_cond_range: tuple = default_cond_range,
                 schedule: str = default_schedule,
                 max_iters: int = 16
                 ):

        self.particulator = None
        self.enable = True

        self.rtol_x = rtol_x
        self.rtol_thd = rtol_thd

        self.RH_max = None
        self.success = None

        self.__substeps = substeps
        self.adaptive = adaptive
        self.counters = {}
        self.dt_cond_range = dt_cond_range
        self.schedule = schedule
        self.max_iters = max_iters

        self.cell_order = None

    def register(self, builder):
        self.particulator = builder.particulator

        builder._set_condensation_parameters(dt_range=self.dt_cond_range, adaptive=self.adaptive,
                                             fuse=32, multiplier=2, RH_rtol=1e-7, max_iters=self.max_iters)
        builder.request_attribute('critical volume')
        builder.request_attribute('kappa')
        builder.request_attribute('dry volume organic fraction')

        for counter in ('n_substeps', 'n_activating', 'n_deactivating', 'n_ripening'):
            self.counters[counter] = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=int)
            if counter == 'n_substeps':
                self.counters[counter][:] = self.__substeps if not self.adaptive else -1
            else:
                self.counters[counter][:] = -1

        self.RH_max = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)
        self.RH_max[:] = np.nan
        self.success = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=bool)
        self.success[:] = False
        self.cell_order = np.arange(self.particulator.mesh.n_cell)

    def __call__(self):
        if self.enable:
            if self.schedule == 'dynamic':
                self.condensation_cell_order = np.argsort(self.counters['n_substeps'])
            elif self.schedule == 'static':
                pass
            else:
                raise NotImplementedError()

            self.particulator.condensation(
                rtol_x=self.rtol_x,
                rtol_thd=self.rtol_thd,
                counters=self.counters,
                RH_max=self.RH_max,
                success=self.success,
                cell_order=self.cell_order
            )
            if not self.success.all():
                raise RuntimeError("Condensation failed")
            # note: this makes order of dynamics matter (e.g., condensation after chemistry or before)
            self.particulator.update_TpRH()

            if self.adaptive:
                self.counters['n_substeps'][:] = np.maximum(self.counters['n_substeps'][:], int(self.particulator.dt / self.dt_cond_range[1]))
                if self.dt_cond_range[0] != 0:
                    self.counters['n_substeps'][:] = np.minimum(self.counters['n_substeps'][:], int(self.particulator.dt / self.dt_cond_range[0]))
            self.particulator.attributes.attributes['volume'].mark_updated()
