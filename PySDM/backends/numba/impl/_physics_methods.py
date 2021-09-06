import numba
from numba import prange
from PySDM.backends.numba import conf
from PySDM.physics import constants as const
import numpy as np


class PhysicsMethods:
    def __init__(self):
        pvs_C = self.formulae.saturation_vapour_pressure.pvs_Celsius
        phys_T = self.formulae.state_variable_triplet.T
        phys_p = self.formulae.state_variable_triplet.p
        phys_pv = self.formulae.state_variable_triplet.pv
        explicit_euler = self.formulae.trivia.explicit_euler
        phys_sigma = self.formulae.surface_tension.sigma
        phys_volume = self.formulae.trivia.volume
        phys_r_cr = self.formulae.hygroscopicity.r_cr

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def explicit_euler_body(y, dt, dy_dt):
            y[:] = explicit_euler(y, dt, dy_dt)
        self.explicit_euler_body = explicit_euler_body

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def critical_volume(v_cr, kappa, f_org, v_dry, v_wet, T, cell):
            for i in prange(len(v_cr)):
                sigma = phys_sigma(T[cell[i]], v_wet[i], v_dry[i], f_org[i])
                v_cr[i] = phys_volume(phys_r_cr(
                    kp=kappa[i],
                    rd3=v_dry[i] / const.pi_4_3,
                    T=T[cell[i]],
                    sgm=sigma
                ))
        self.critical_volume_body = critical_volume

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def temperature_pressure_RH_body(rhod, thd, qv, T, p, RH):
            for i in prange(T.shape[0]):
                T[i] = phys_T(rhod[i], thd[i])
                p[i] = phys_p(rhod[i], T[i], qv[i])
                RH[i] = phys_pv(p[i], qv[i]) / pvs_C(T[i] - const.T0)
        self.temperature_pressure_RH_body = temperature_pressure_RH_body

        @numba.njit(**{**conf.JIT_FLAGS, 'fastmath': self.formulae.fastmath})
        def terminal_velocity_body(values, radius, k1, k2, k3, r1, r2):
            for i in prange(len(values)):
                if radius[i] < r1:
                    values[i] = k1 * radius[i] ** 2
                elif radius[i] < r2:
                    values[i] = k2 * radius[i]
                else:
                    values[i] = k3 * radius[i] ** (1 / 2)
        self.terminal_velocity_body = terminal_velocity_body

    def temperature_pressure_RH(self, rhod, thd, qv, T, p, RH):
        self.temperature_pressure_RH_body(rhod.data, thd.data, qv.data, T.data, p.data, RH.data)

    def terminal_velocity(self, values, radius, k1, k2, k3, r1, r2):
        self.terminal_velocity_body(values, radius, k1, k2, k3, r1, r2)

    def explicit_euler(self, y, dt, dy_dt):
        self.explicit_euler_body(y.data, dt, dy_dt)

    def critical_volume(self, v_cr, kappa, f_org, v_dry, v_wet, T, cell):
        self.critical_volume_body(v_cr.data, kappa.data, f_org.data, v_dry.data, v_wet.data, T.data, cell.data)
