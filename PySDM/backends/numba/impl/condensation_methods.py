"""
Created at 11.2019
"""

from PySDM.physics import constants as const
from PySDM.backends.numba import conf
from PySDM.physics.formulae import temperature_pressure_pv, radius, dthd_dt, \
    within_tolerance, D as phys_D, K as phys_K
import PySDM.physics.formulae as phys
from PySDM.backends.numba.toms748 import toms748_solve
import numba
import numpy as np
import math
from functools import lru_cache


class CondensationMethods:
    @staticmethod
    def make_adapt_substeps(dt, step_fake, dt_range, fuse=100, multiplier=2):
        if not isinstance(multiplier, int):
            raise ValueError()
        if dt_range[1] > dt:
            dt_range = (dt_range[0], dt)
        if dt_range[0] == 0:
            raise NotImplementedError()  # TODO #490
            # TODO #437: n_substeps_max = ... (fuse)
        else:
            n_substeps_max = math.floor(dt / dt_range[0])
        n_substeps_min = math.ceil(dt / dt_range[1])

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def adapt_substeps(args, n_substeps, thd, rtol_thd):
            n_substeps = np.maximum(n_substeps_min, n_substeps // multiplier)
            thd_new_long, success = step_fake(args, dt, n_substeps)
            if not success:
                return 0, success
            for burnout in range(fuse + 1):
                if burnout == fuse:
                    print("burnout")
                    return 0, False
                thd_new_short, success = step_fake(args, dt, n_substeps * multiplier)
                dthd_long = thd_new_long - thd
                dthd_short = thd_new_short - thd
                error_estimate = np.abs(dthd_long - multiplier * dthd_short)
                thd_new_long = thd_new_short
                if within_tolerance(error_estimate, thd, rtol_thd) and success:
                    break
                n_substeps *= multiplier
                if n_substeps > n_substeps_max:
                    break
            return np.minimum(n_substeps_max, n_substeps), success

        return adapt_substeps

    @staticmethod
    def make_step_fake(step_impl):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_fake(args, dt, n_substeps):
            dt /= n_substeps
            _, thd_new, _, _, _, _, success = step_impl(*args, dt, 1, True)
            return thd_new, success

        return step_fake

    @staticmethod
    def make_step(step_impl):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step(args, dt, n_substeps):
            return step_impl(*args, dt, n_substeps, False)

        return step

    @staticmethod
    def make_step_impl(phys_pvs_C, phys_lv, calculate_ml_old, calculate_ml_new):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_impl(v, v_cr, n, vdry, cell_idx, kappa, thd, qv, dthd_dt_pred, dqv_dt_pred,
                      m_d, rhod_mean, rtol_x, dt, n_substeps, fake):
            dt /= n_substeps
            ml_old = calculate_ml_old(v, n, cell_idx)
            count_activating, count_deactivating, count_ripening = 0, 0, 0
            RH_max = 0
            success = True
            for t in range(n_substeps):
                thd += dt * dthd_dt_pred / 2  # TODO #48 example showing that it makes sense
                qv += dt * dqv_dt_pred / 2

                T, p, pv = temperature_pressure_pv(rhod_mean, thd, qv)
                lv = phys_lv(T)
                pvs = phys_pvs_C(T - const.T0)
                RH = pv / pvs
                ml_new, success_within_substep, n_activating, n_deactivating, n_ripening = \
                    calculate_ml_new(dt, fake, T, p, RH, v, v_cr, n, vdry, cell_idx, kappa, lv, pvs, rtol_x)
                dml_dt = (ml_new - ml_old) / dt
                dqv_dt_corr = - dml_dt / m_d
                dthd_dt_corr = dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt_corr, lv=lv)

                thd += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
                qv += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
                ml_old = ml_new
                count_activating += n_activating
                count_deactivating += n_deactivating
                count_ripening += n_ripening
                RH_max = max(RH_max, RH)
                success = success and success_within_substep
            return qv, thd, count_activating, count_deactivating, count_ripening, RH_max, success

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
    def make_calculate_ml_new(dx_dt, volume_of_x, x, phys_r_dr_dt):
        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def minfun(x_new, x_old, dt, p, kappa, rd, T, RH, lv, pvs, D, K):
            r_new = radius(volume_of_x(x_new))
            RH_eq = phys.RH_eq(r_new, T, kappa, rd)
            r_dr_dt = phys_r_dr_dt(RH_eq, T, RH, lv, pvs, D, K)
            return x_old - x_new + dt * dx_dt(x_new, r_dr_dt)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def calculate_ml_new(dt, fake, T, p, RH, v, v_cr, n, vdry, cell_idx, kappa, lv, pvs, rtol_x):
            result = 0
            n_activating = 0
            n_deactivating = 0
            n_activated_and_growing = 0
            success = True
            for drop in cell_idx:
                x_old = x(v[drop])
                r_old = radius(v[drop])
                rd = radius(vdry[drop])
                D = phys_D(r_old, T)
                K = phys_K(r_old, T, p)
                RH_eq = phys.RH_eq(r_old, T, kappa, rd)
                args = (x_old, dt, p, kappa, rd, T, RH, lv, pvs, D, K)
                r_dr_dt_old = phys_r_dr_dt(RH_eq, T, RH, lv, pvs, D, K)
                dx_old = dt * dx_dt(x_old, r_dr_dt_old)
                if dx_old == 0:
                    x_new = x_old
                else:
                    x_dry = x(vdry[drop])
                    a = x_old
                    b = max(x_dry, a + dx_old)
                    fa = minfun(a, *args)
                    fb = minfun(b, *args)

                    counter = 0
                    while not fa * fb < 0:
                        counter += 1
                        if counter > 16:
                            print("failed to find interval")
                            success = False
                            break
                        b = max(x_dry, a + math.ldexp(dx_old, counter))
                        fb = minfun(b, *args)

                    if not success:
                        x_new = np.nan
                    elif a != b:
                        if a > b:
                            a, b = b, a
                            fa, fb = fb, fa

                        max_iters = 16
                        x_new, iters_taken = toms748_solve(minfun, args, a, b, fa, fb, rtol_x, max_iters)
                        if iters_taken in (-1, max_iters):
                            print("TOMS failed")
                            success = False
                    else:
                        x_new = x_old

                v_new = volume_of_x(x_new)
                result += n[drop] * v_new * const.rho_w
                if not fake:
                    if v_new > v_cr[drop] and v_new > v[drop]:
                        n_activated_and_growing += n[drop]
                    if v_new > v_cr[drop] > v[drop]:
                        n_activating += n[drop]
                    if v_new < v_cr[drop] < v[drop]:
                        n_deactivating += n[drop]
                    v[drop] = v_new
            n_ripening = n_activated_and_growing if n_deactivating > 0 else 0
            return result, success, n_activating, n_deactivating, n_ripening

        return calculate_ml_new

    def make_condensation_solver(self, dt, dt_range, adaptive=True):
        phys_pvs_C = self.formulae.saturation_vapour_pressure.pvs_Celsius
        phys_lv = self.formulae.latent_heat.lv
        phys_r_dr_dt = self.formulae.drop_growth.r_dr_dt
        return CondensationMethods.make_condensation_solver_impl(
            phys_pvs_C = phys_pvs_C,
            phys_lv=phys_lv,
            phys_r_dr_dt=phys_r_dr_dt,
            dx_dt=self.formulae.condensation_coordinate.dx_dt,
            volume=self.formulae.condensation_coordinate.volume,
            x=self.formulae.condensation_coordinate.x,
            dt=dt,
            dt_range=dt_range,
            adaptive=adaptive
        )

    @staticmethod
    @lru_cache()
    def make_condensation_solver_impl(phys_pvs_C, phys_lv, phys_r_dr_dt, dx_dt, volume, x, dt, dt_range, adaptive):
        calculate_ml_old = CondensationMethods.make_calculate_ml_old()
        calculate_ml_new = CondensationMethods.make_calculate_ml_new(dx_dt, volume, x, phys_r_dr_dt)
        step_impl = CondensationMethods.make_step_impl(phys_pvs_C, phys_lv, calculate_ml_old, calculate_ml_new)
        step_fake = CondensationMethods.make_step_fake(step_impl)
        adapt_substeps = CondensationMethods.make_adapt_substeps(dt, step_fake, dt_range)
        step = CondensationMethods.make_step(step_impl)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def solve(v, v_cr, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean,
                  rtol_x, rtol_thd, dt, n_substeps):
            args = (v, v_cr, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean, rtol_x)
            success = True
            if adaptive:
                n_substeps, success = adapt_substeps(args, n_substeps, thd, rtol_thd)
            if success:
                qv, thd, n_activating, n_deactivating, n_ripening, RH_max, success = step(args, dt, n_substeps)
            else:
                n_activating, n_deactivating, n_ripening, RH_max = -1, -1, -1, -1
            return success, qv, thd, n_substeps, n_activating, n_deactivating, n_ripening, RH_max

        return solve
