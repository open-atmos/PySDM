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

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def freeze_time_dependent_body(rand, nucleation_sites, volume, r, dt):
            p = 1 - np.exp(-r * dt)
            # TODO #599: assert if > 1?
            for i in numba.prange(len(nucleation_sites)):
                if _unfrozen(volume, i):
                    if rand[i] < p:
                        _freeze(volume, i)
        self.freeze_time_dependent_body = freeze_time_dependent_body

    def freeze_singular(self, T_fz, v_wet, T, RH, cell):
        self.freeze_singular_body(T_fz.data, v_wet.data, T.data, RH.data, cell.data)

    def freeze_time_dependent(self, rand, nucleation_sites, volume, r, dt):
        self.freeze_time_dependent_body(rand.data, nucleation_sites.data, volume.data, r, dt)
