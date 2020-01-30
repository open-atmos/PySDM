from ....physics import formulae as phys
from ....physics import constants as const
from .....backends.numba import conf
import numba
import numpy as np


@numba.njit(**conf.JIT_FLAGS)
def impl(y, v, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d_mean, rhod_mean,
         rtol_lnv, rtol_thd, dt, substeps_hint):

    args = (y, v, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d_mean, rhod_mean, rtol_lnv)

    n_substeps = substeps_hint

    multiplier = 1.2
    thd_new_dt_all = step_fake(args, dt / n_substeps)
    while True:
        thd_new_dt_one = step_fake(args, dt / (multiplier * n_substeps))
        if thd_new_dt_one != -1 and thd_new_dt_all != -1:
            dthd_all = (thd_new_dt_all - thd)
            dthd_one = (thd_new_dt_one - thd)
            error_estimate = np.abs(dthd_all - multiplier * dthd_one)
            if within_tolerance(error_estimate, thd, rtol_thd):
                break
        n_substeps *= multiplier
        thd_new_dt_all = thd_new_dt_one

    qv, thd = step_true(args, dt, n_substeps)

    return qv, thd, np.maximum(1, n_substeps // multiplier)


@numba.njit(**conf.JIT_FLAGS)
def step_fake(args, dt):
    _, thd_new = step_impl(*args, dt, 1, True)
    return thd_new


@numba.njit(**conf.JIT_FLAGS)
def step_true(args, dt, n_substeps):
    return step_impl(*args, dt, n_substeps, False)


@numba.njit(**conf.JIT_FLAGS)
def step_impl(
    _, v, n, vdry, cell_idx, kappa, thd, qv, dthd_dt_pred, dqv_dt_pred, m_d_mean, rhod_mean, rtol_lnv,
    dt, n_substeps, fake
):
    dt /= n_substeps
    n_sd_in_cell = len(cell_idx)

    ml_old = 0
    for i in range(n_sd_in_cell):
        ml_old += n[cell_idx[i]] * v[cell_idx[i]] * const.rho_w

    for t in range(n_substeps):
        thd += dt * dthd_dt_pred / 2  # TODO: test showing that it makes sense
        qv += dt * dqv_dt_pred / 2
        T, p, RH = phys.temperature_pressure_RH(rhod_mean, thd, qv)

        ml_new = 0
        for i in range(n_sd_in_cell):
            lnv_old = np.log(v[cell_idx[i]])  # TODO: abstract out coord logic
            rd = phys.radius(volume=vdry[cell_idx[i]])
            dlnv_old = dt * phys.dlnv_dt(lnv_old, T, p, RH, kappa, rd)
            if fake and phys.radius(np.exp(lnv_old + dlnv_old)) < rd:
                return -1, -1
            a = lnv_old
            interval = dlnv_old
            args = (lnv_old, dt, T, p, RH, kappa, rd)
            lnv_new = bisec(_minfun, a, interval, args, rtol_lnv, n_substeps)
            v_new = np.exp(lnv_new)
            if not fake:
                v[cell_idx[i]] = v_new
            ml_new += n[cell_idx[i]] * v_new * const.rho_w

        dml_dt = (ml_new - ml_old) / dt
        dqv_dt_corr = - dml_dt / m_d_mean
        dthd_dt_corr = phys.dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt_pred + dqv_dt_corr)
        thd += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
        qv += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
        ml_old = ml_new

    return qv, thd


@numba.njit(**conf.JIT_FLAGS)
def within_tolerance(error_estimate, value, rtol):
    return error_estimate < rtol * np.abs(value)


@numba.njit(**conf.JIT_FLAGS)
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


@numba.njit(**conf.JIT_FLAGS)
def _minfun(lnv_new,
    lnv_old, dt, T, p, RH, kappa, rd):
    return lnv_old - lnv_new + dt * phys.dlnv_dt(lnv_new, T, p, RH, kappa, rd)
