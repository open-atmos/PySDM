import numba
import numpy as np
from ...numba import conf
from ....physics import constants as const


class FreezingMethods:
    def __init__(self):
        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath, 'parallel': False})
        def _unfrozen(volume, i):
            return volume[i] > 0

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath, 'parallel': False})
        def _freeze(volume, i):
            volume[i] = -1 * volume[i] * const.rho_w / const.rho_i
            # TODO #599: change thd (latent heat)!
            # TODO #599: handle the negative volume in tests, attributes, products, dynamics, ...

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def freeze_singular_body(T_fz, v_wet, T, RH, cell):
            for i in numba.prange(len(T_fz)):
                if _unfrozen(v_wet, i) and RH[cell[i]] > 1 and T[cell[i]] <= T_fz[i]:
                    _freeze(v_wet, i)
        self.freeze_singular_body = freeze_singular_body

        J_het = self.formulae.heterogeneous_ice_nucleation_rate.J_het

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def freeze_time_dependent_body(rand, immersed_surface_area, volume, dt, cell, a_w_ice):
            n_sd = len(volume)
            for i in numba.prange(n_sd):
                if _unfrozen(volume, i):
                    p = 1 - np.exp(-J_het(a_w_ice[cell[i]]) * immersed_surface_area[i] * dt)  # TODO #599: assert if > 1?
                    if rand[i] < p:
                        _freeze(volume, i)
        self.freeze_time_dependent_body = freeze_time_dependent_body

    def freeze_singular(self, T_fz, v_wet, T, RH, cell):
        self.freeze_singular_body(T_fz.data, v_wet.data, T.data, RH.data, cell.data)

    def freeze_time_dependent(self, rand, immersed_surface_area, volume, dt, cell, a_w_ice):
        self.freeze_time_dependent_body(rand.data, immersed_surface_area.data, volume.data, dt, cell.data, a_w_ice.data)
