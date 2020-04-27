"""
Created at 11.2019
"""

from PySDM.simulation.physics import constants as const
from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import \
    radius, temperature_pressure_RH, dr_dt_MM, dr_dt_FF, dT_i_dt_FF, dthd_dt, within_tolerance, bisec
from .coordinates import volume as coord_volume
from .coordinates import volume_logarithm as coord_volume_logarithm
import numba
import numpy as np


class CondensationMethods:
    @staticmethod
    def make_condensation_solver(coord='volume logarithm', adaptive=True):
        if coord == 'volume':
            coord = coord_volume
        elif coord == 'volume logarithm':
            coord = coord_volume_logarithm
        else:
            raise ValueError()

        x = coord.x
        volume = coord.volume
        dx_dt = coord.dx_dt

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def solve(v, particle_temperatures, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d_mean, rhod_mean,
                  rtol_x, rtol_thd, dt, substeps_hint):

            args = (v, particle_temperatures, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d_mean, rhod_mean, rtol_x)

            n_substeps = substeps_hint

            multiplier = 2
            thd_new_long = step_fake(args, dt / n_substeps)

            counter = 0
            while True:
                counter += 1
                if counter > 100:
                    raise RuntimeError("Cannot find solution!")

                thd_new_short = step_fake(args, dt / (multiplier * n_substeps))
                dthd_long = (thd_new_long - thd)
                dthd_short = (thd_new_short - thd)
                error_estimate = np.abs(dthd_long - multiplier * dthd_short)
                if not adaptive or within_tolerance(error_estimate, thd, rtol_thd):
                    break
                n_substeps *= multiplier
                thd_new_long = thd_new_short

            qv, thd = step_true(args, dt, n_substeps)

            return qv, thd, np.maximum(1, n_substeps // multiplier)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_fake(args, dt):
            _, thd_new = step_impl(*args, dt, 1, True)
            return thd_new

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_true(args, dt, n_substeps):
            return step_impl(*args, dt, n_substeps, False)

        @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
        def step_impl(
            v, particle_temperatures, n, vdry, cell_idx, kappa, thd, qv, dthd_dt_pred, dqv_dt_pred, m_d_mean,
            rhod_mean, rtol_x, dt, n_substeps, fake
        ):
            using_drop_temperatures = len(particle_temperatures) > 0

            dt /= n_substeps
            n_sd_in_cell = len(cell_idx)

            ml_old = 0
            for i in range(n_sd_in_cell):
                ml_old += n[cell_idx[i]] * v[cell_idx[i]] * const.rho_w

            for t in range(n_substeps):
                thd += dt * dthd_dt_pred / 2  # TODO: test showing that it makes sense
                qv += dt * dqv_dt_pred / 2
                T, p, RH = temperature_pressure_RH(rhod_mean, thd, qv)

                ml_new = 0
                for i in range(n_sd_in_cell):
                    x_old = x(v[cell_idx[i]])
                    if using_drop_temperatures:
                        T_i_old = particle_temperatures[cell_idx[i]]

                    r_old = radius(v[cell_idx[i]])
                    rd = radius(volume=vdry[cell_idx[i]])
                    dr_dt_old = (
                        dr_dt_MM(r_old, T, p, RH, kappa, rd) if not using_drop_temperatures else
                        dr_dt_FF(r_old, T, p, qv, kappa, rd, T_i_old)
                    )
                    dx_old = dt * dx_dt(x_old, dr_dt_old)

                    if dx_old < 0:
                        dx_old = np.maximum(dx_old, x(vdry[cell_idx[i]]) - x_old)

                    a = x_old
                    interval = dx_old
                    if not using_drop_temperatures:
                        args_MM = (x_old, dt, T, p, RH, kappa, rd)
                        x_new = bisec(_minfun_MM, a, interval, args_MM, rtol_x, n_substeps)
                    else:
                        args_FF = (x_old, dt, T, p, qv, kappa, rd, T_i_old)
                        x_new = bisec(_minfun_FF, a, interval, args_FF, rtol_x, n_substeps)

                    v_new = volume(x_new)

                    if not fake:
                        if using_drop_temperatures:
                            T_i_new = T_i_old + dt * dT_i_dt_FF(r_old, T, p, T_i_old, dr_dt_old)
                            particle_temperatures[cell_idx[i]] = T_i_new
                        v[cell_idx[i]] = v_new
                    ml_new += n[cell_idx[i]] * v_new * const.rho_w

                dml_dt = (ml_new - ml_old) / dt
                dqv_dt_corr = - dml_dt / m_d_mean
                dthd_dt_corr = dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt_pred + dqv_dt_corr)
                thd += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
                qv += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
                ml_old = ml_new

            return qv, thd

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


