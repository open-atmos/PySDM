from PySDM.simulation.physics import constants as const
from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import \
    radius, temperature_pressure_RH, dr_dt_MM, dr_dt_FF, dT_i_dt_FF, dlnv_dt, dthd_dt
import numba
import numpy as np


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def solve(v, particle_temperatures, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d_mean, rhod_mean,
          rtol_lnv, rtol_thd, dt, substeps_hint):

    args = (v, particle_temperatures, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d_mean, rhod_mean, rtol_lnv)

    n_substeps = substeps_hint

    multiplier = 2
    thd_new_long = step_fake(args, dt / n_substeps)
    while True:
        thd_new_short = step_fake(args, dt / (multiplier * n_substeps))
        dthd_long = (thd_new_long - thd)
        dthd_short = (thd_new_short - thd)
        error_estimate = np.abs(dthd_long - multiplier * dthd_short)
        if within_tolerance(error_estimate, thd, rtol_thd):
            break
        n_substeps *= multiplier
        thd_new_long = thd_new_short

    qv, thd = step_true(args, dt, n_substeps)

    return qv, thd, np.maximum(1, n_substeps // multiplier)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def step_fake(args, dt):
    _, thd_new = step_impl(*args, dt, 1, True)
    return thd_new


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def step_true(args, dt, n_substeps):
    return step_impl(*args, dt, n_substeps, False)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def step_impl(
    v, particle_temperatures, n, vdry, cell_idx, kappa, thd, qv, dthd_dt_pred, dqv_dt_pred, m_d_mean,
    rhod_mean, rtol_lnv, dt, n_substeps, fake
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
            lnv_old = np.log(v[cell_idx[i]])  # TODO: abstract out coord logic
            if using_drop_temperatures:
                T_i_old = particle_temperatures[cell_idx[i]]

            r_old = radius(v[cell_idx[i]])
            rd = radius(volume=vdry[cell_idx[i]])
            dr_dt_old = (
                dr_dt_MM(r_old, T, p, RH, kappa, rd) if not using_drop_temperatures else
                dr_dt_FF(r_old, T, p, qv, kappa, rd, T_i_old)
            )
            dlnv_old = dt * dlnv_dt(lnv_old, dr_dt_old)

            if dlnv_old < 0:
                dlnv_old = np.maximum(dlnv_old, np.log(vdry[cell_idx[i]]) - lnv_old)

            a = lnv_old
            interval = dlnv_old
            if not using_drop_temperatures:
                args_MM = (lnv_old, dt, T, p, RH, kappa, rd)
                lnv_new = bisec(_minfun_MM, a, interval, args_MM, rtol_lnv, n_substeps)
            else:
                args_FF = (lnv_old, dt, T, p, qv, kappa, rd, T_i_old)
                lnv_new = bisec(_minfun_FF, a, interval, args_FF, rtol_lnv, n_substeps)

            v_new = np.exp(lnv_new)

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


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def within_tolerance(error_estimate, value, rtol):
    return error_estimate < rtol * np.abs(value)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}} )
def bisec(minfun, a, interval, args, rtol, n_substeps):
    b = a + interval

    i = 0
    while minfun(a, *args) * minfun(b, *args) > 0:
        i += 1
        b = a + interval * 2**i

    if b < a:
        a, b = b, a

    fa = minfun(a, *args) # TODO: computed above

    iter = 0
    while not within_tolerance(error_estimate=(b-a), value=(a+b)/2, rtol=rtol):
        lnv_new = (a + b) / 2
        f = minfun(lnv_new, *args)
        if f * fa > 0:
            a = lnv_new
        else:
            b = lnv_new
        iter += 1
    lnv_new = (a + b) / 2

    return lnv_new


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def _minfun_FF(lnv_new, lnv_old, dt, T, p, qv, kappa, rd, T_i):
    r_new = radius(np.exp(lnv_new))
    dr_dt = dr_dt_FF(r_new, T, p, qv, kappa, rd, T_i)
    return lnv_old - lnv_new + dt * dlnv_dt(lnv_new, dr_dt)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def _minfun_MM(lnv_new, lnv_old, dt, T, p, RH, kappa, rd):
    r_new = radius(np.exp(lnv_new))
    dr_dt = dr_dt_MM(r_new, T, p, RH, kappa, rd)
    return lnv_old - lnv_new + dt * dlnv_dt(lnv_new, dr_dt)
