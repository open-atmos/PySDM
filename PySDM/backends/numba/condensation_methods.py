"""
Created at 11.2019
"""

from ...physics import constants as const
from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import \
    radius, temperature_pressure_RH, dr_dt_MM, dr_dt_FF, dT_i_dt_FF, dthd_dt, within_tolerance, bisec
from .coordinates import mapper as coordinates
import numba
import numpy as np


class CondensationMethods:
    @staticmethod
    def make_condensation_solver(coord='volume logarithm', adaptive=True):

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def solve(v, particle_T, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean,
                  rtol_x, rtol_thd, dt, n_substeps):
            args = (v, particle_T, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean, rtol_x)
            if adaptive:
                n_substeps = adapt_substeps(args, n_substeps, dt, thd, rtol_thd)
            qv, thd = step(args, dt, n_substeps)

            return qv, thd, n_substeps

        fuse = 100
        multiplier = 2

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def adapt_substeps(args, n_substeps, dt, thd, rtol_thd):
            n_substeps = np.maximum(1, n_substeps // multiplier)
            thd_new_long = step_fake(args, dt, n_substeps)
            for burnout in range(fuse + 1):
                if burnout == fuse:
                    raise RuntimeError("Cannot find solution!")
                thd_new_short = step_fake(args, dt, n_substeps * multiplier)
                dthd_long = thd_new_long - thd
                dthd_short = thd_new_short - thd
                error_estimate = np.abs(dthd_long - multiplier * dthd_short)
                thd_new_long = thd_new_short
                if within_tolerance(error_estimate, thd, rtol_thd):
                    break
                n_substeps *= multiplier
            return n_substeps

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_fake(args, dt, n_substeps):
            dt /= n_substeps
            _, thd_new = step_impl(*args, dt, 1, True)
            return thd_new

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step(args, dt, n_substeps):
            return step_impl(*args, dt, n_substeps, False)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_impl(v, particle_T, n, vdry, cell_idx, kappa, thd, qv, dthd_dt_pred, dqv_dt_pred,
                      m_d, rhod_mean, rtol_x, dt, n_substeps, fake):
            dt /= n_substeps
            ml_old = calculate_ml_old(v, n, cell_idx)
            for t in range(n_substeps):
                thd += dt * dthd_dt_pred / 2  # TODO: test showing that it makes sense
                qv += dt * dqv_dt_pred / 2
                T, p, RH = temperature_pressure_RH(rhod_mean, thd, qv)
                ml_new = calculate_ml_new(dt, fake, T, p, RH, v, particle_T, n, vdry, cell_idx, kappa, qv, rtol_x)
                dml_dt = (ml_new - ml_old) / dt
                dqv_dt_corr = - dml_dt / m_d
                dthd_dt_corr = dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt_corr)
                thd += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
                qv += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
                ml_old = ml_new

            return qv, thd

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def calculate_ml_old(v, n, cell_idx):
            result = 0
            for drop in cell_idx:
                result += n[drop] * v[drop] * const.rho_w
            return result

        dx_dt, volume, x = coordinates.get(coord)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def calculate_ml_new(dt, fake, T, p, RH, v, particle_T, n, vdry, cell_idx, kappa, qv, rtol_x):
            result = 0
            using_drop_temperatures = len(particle_T) > 0  # TODO: move outside numba
            for drop in cell_idx:
                x_old = x(v[drop])
                if using_drop_temperatures:
                    particle_T_old = particle_T[drop]
                r_old = radius(v[drop])
                rd = radius(volume=vdry[drop])
                dr_dt_old = (
                    dr_dt_MM(r_old, T, p, RH, kappa, rd) if not using_drop_temperatures else
                    dr_dt_FF(r_old, T, p, qv, kappa, rd, particle_T_old)
                )
                dx_old = dt * dx_dt(x_old, dr_dt_old)
                if dx_old < 0:
                    dx_old = np.maximum(dx_old, x(vdry[drop]) - x_old)
                a = x_old
                interval = dx_old
                if not using_drop_temperatures:
                    args_MM = (x_old, dt, T, p, RH, kappa, rd)
                    x_new = bisec(_minfun_MM, a, interval, args_MM, rtol_x)
                else:
                    args_FF = (x_old, dt, T, p, qv, kappa, rd, particle_T_old)
                    x_new = bisec(_minfun_FF, a, interval, args_FF, rtol_x)
                v_new = volume(x_new)
                if not fake:
                    if using_drop_temperatures:
                        T_i_new = particle_T_old + dt * dT_i_dt_FF(r_old, T, p, particle_T_old, dr_dt_old)
                        particle_T[drop] = T_i_new
                    v[drop] = v_new
                result += n[drop] * v_new * const.rho_w
            return result

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def _minfun_FF(x_new, x_old, dt, T, p, qv, kappa, rd, T_i):
            r_new = radius(volume(x_new))
            dr_dt = dr_dt_FF(r_new, T, p, qv, kappa, rd, T_i)
            return x_old - x_new + dt * dx_dt(x_new, dr_dt)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def _minfun_MM(x_new, x_old, dt, T, p, RH, kappa, rd):
            r_new = radius(volume(x_new))
            dr_dt = dr_dt_MM(r_new, T, p, RH, kappa, rd)
            return x_old - x_new + dt * dx_dt(x_new, dr_dt)

        return solve
