"""
Created at 11.2019
"""

from PySDM.physics import constants as const
from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import \
    radius, temperature_pressure_RH, dr_dt_MM, dr_dt_FF, dT_i_dt_FF, dthd_dt, within_tolerance, bisec
from PySDM.backends.numba.coordinates import mapper as coordinates
import numba
import numpy as np


class CondensationMethods:
    @staticmethod
    def make_adapt_substeps(step_fake, fuse=100, multiplier=2):

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

        return adapt_substeps

    @staticmethod
    def make_step_fake(step_impl):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_fake(args, dt, n_substeps):
            dt /= n_substeps
            _, thd_new, _ = step_impl(*args, dt, 1, True)
            return thd_new

        return step_fake

    @staticmethod
    def make_step(step_impl):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step(args, dt, n_substeps):
            return step_impl(*args, dt, n_substeps, False)

        return step

    @staticmethod
    def make_step_impl(calculate_ml_old, calculate_ml_new):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_impl(v, particle_T, n, vdry, cell_idx, kappa, thd, qv, dthd_dt_pred, dqv_dt_pred,
                      m_d, rhod_mean, rtol_x, dt, n_substeps, fake):
            dt /= n_substeps
            ml_old = calculate_ml_old(v, n, cell_idx)
            ripenings = 0
            for t in range(n_substeps):
                thd += dt * dthd_dt_pred / 2  # TODO: test showing that it makes sense
                qv += dt * dqv_dt_pred / 2
                T, p, RH = temperature_pressure_RH(rhod_mean, thd, qv)
                ml_new, ripening = calculate_ml_new(dt, fake, T, p, RH, v, particle_T, n, vdry, cell_idx, kappa, qv, rtol_x)
                dml_dt = (ml_new - ml_old) / dt
                dqv_dt_corr = - dml_dt / m_d
                dthd_dt_corr = dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt_corr)
                thd += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
                qv += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
                ml_old = ml_new
                ripenings += ripening
            return qv, thd, ripenings

        return step_impl

    @staticmethod
    def make_calculate_ml_old():
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def calculate_ml_old(v, n, cell_idx):
            result = 0
            for drop in cell_idx:
                result += n[drop] * v[drop] * const.rho_w
            return result

        return calculate_ml_old

    @staticmethod
    def make_calculate_ml_new(dx_dt, volume, x, enable_drop_temperatures):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def _minfun_FF(x_new, x_old, dt, T, p, qv, kappa, rd, T_i):
            r_new = radius(volume(x_new))
            dr_dt = dr_dt_FF(r_new, T, p, qv, kappa, rd, T_i)
            return x_old - x_new + dt * dx_dt(x_new, dr_dt)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def _minfun_MM(x_new, x_old, dt, T, p, RH, kappa, rd, _):
            r_new = radius(volume(x_new))
            dr_dt = dr_dt_MM(r_new, T, p, RH, kappa, rd)
            return x_old - x_new + dt * dx_dt(x_new, dr_dt)

        minfun = _minfun_FF if enable_drop_temperatures else _minfun_MM

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def calculate_ml_new(dt, fake, T, p, RH, v, particle_T, n, vdry, cell_idx, kappa, qv, rtol_x):
            result = 0
            growing = 0
            decreasing = 0
            for drop in cell_idx:
                x_old = x(v[drop])
                r_old = radius(v[drop])
                rd = radius(volume=vdry[drop])
                if enable_drop_temperatures:
                    particle_T_old = particle_T[drop]
                    dr_dt_old = dr_dt_FF(r_old, T, p, qv, kappa, rd, particle_T_old)
                    args = (x_old, dt, T, p, qv, kappa, rd, particle_T_old)
                else:
                    dr_dt_old = dr_dt_MM(r_old, T, p, RH, kappa, rd)
                    args = (x_old, dt, T, p, RH, kappa, rd, 0)
                dx_old = dt * dx_dt(x_old, dr_dt_old)
                if dx_old < 0:
                    dx_old = np.maximum(dx_old, x(vdry[drop]) - x_old)
                a = x_old
                interval = dx_old
                x_new = bisec(minfun, a, interval, args, rtol_x)
                v_new = volume(x_new)
                if not fake:
                    if enable_drop_temperatures:
                        T_i_new = particle_T_old + dt * dT_i_dt_FF(r_old, T, p, particle_T_old, dr_dt_old)
                        particle_T[drop] = T_i_new
                    # if v_new > 4/3 * np.pi * (r_cr[drop])**3: # TODO: difference if r<r_cr, filter out noise
                    if abs((v_new-v[drop])/v_new) > .5:
                        if v_new - v[drop] > 0:
                            growing += 1
                        else:
                            decreasing += 1
                    v[drop] = v_new
                result += n[drop] * v_new * const.rho_w
            return result, (growing > 0 and decreasing > 0 )

        return calculate_ml_new

    @staticmethod
    def make_condensation_solver(coord='volume logarithm', adaptive=True, enable_drop_temperatures=False):
        dx_dt, volume, x = coordinates.get(coord)
        calculate_ml_old = CondensationMethods.make_calculate_ml_old()
        calculate_ml_new = CondensationMethods.make_calculate_ml_new(dx_dt, volume, x, enable_drop_temperatures)
        step_impl = CondensationMethods.make_step_impl(calculate_ml_old, calculate_ml_new)
        step_fake = CondensationMethods.make_step_fake(step_impl)
        adapt_substeps = CondensationMethods.make_adapt_substeps(step_fake)
        step = CondensationMethods.make_step(step_impl)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def solve(v, particle_T, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean,
                  rtol_x, rtol_thd, dt, n_substeps):
            args = (v, particle_T, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean, rtol_x)
            if adaptive:
                n_substeps = adapt_substeps(args, n_substeps, dt, thd, rtol_thd)
            qv, thd, ripenings = step(args, dt, n_substeps)

            return qv, thd, n_substeps, ripenings

        return solve
