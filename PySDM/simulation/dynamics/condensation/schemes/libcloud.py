from ....physics import formulae as phys
from ....physics import constants as const
from .....backends.numba import conf
import numba
import numpy as np


@numba.njit(fastmath=conf.NUMBA_FASTMATH, error_model=conf.NUMBA_ERROR_MODEL)
def impl(y, v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean, rtol, atol):

    # TODO: track substep count and decrease/increase
    # TODO: error estimate based on dt * (dthe_dt(N) - dthe_dt(N+1))
    n_substeps = np.maximum(1, int(dt))
    while not within_tolerance(error_estimate=error_estimate(y, v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean, rtol, atol, n_substeps=n_substeps), value=thd, rtol=rtol, atol=atol):
        n_substeps *= 2

    qv, thd, thd_err_est = impl_impl(y, v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean, rtol, atol, n_substeps=n_substeps,
                                     update_v=True)
    # TODO: if very good: lower number of substeps
    return qv, thd

@numba.njit(fastmath=conf.NUMBA_FASTMATH, error_model=conf.NUMBA_ERROR_MODEL)
def error_estimate(y, v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean, rtol, atol, n_substeps):

    _, _, thd_err_est = impl_impl(y, v, n, vdry,
             cell_idx,
             dt/n_substeps, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean, rtol, atol, n_substeps=1,
                                     update_v=False)
    return thd_err_est

@numba.njit(fastmath=conf.NUMBA_FASTMATH, error_model=conf.NUMBA_ERROR_MODEL)
def impl_impl(y, v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt_pred, dqv_dt_pred,
             m_d_mean, rhod_mean, rtol, atol, n_substeps, update_v):
    assert update_v or n_substeps == 1

    dt /= n_substeps
    n_sd_in_cell = len(cell_idx)

    ml_old = 0
    for i in range(n_sd_in_cell):
        ml_old += n[cell_idx[i]] * v[cell_idx[i]] * const.rho_w

    thd_err_est = 0
    for t in range(n_substeps):
        thd += dt * dthd_dt_pred / 2  # TODO: test showing that it makes sense
        qv += dt * dqv_dt_pred / 2
        T, p, RH = phys.temperature_pressure_RH(rhod_mean, thd, qv)

        ml_new = 0
        for i in range(n_sd_in_cell):
            lnv_old = np.log(v[cell_idx[i]])  # TODO: abstract out coord logic
            rd = phys.radius(volume=vdry[cell_idx[i]])
            dlnv_old = dt * phys.dlnv_dt(lnv_old, T, p, RH, kappa, rd)
            a = lnv_old
            interval = dlnv_old
            args = (lnv_old, dt, T, p, RH, kappa, rd)
            lnv_new = bisec(_minfun, a, interval, args, rtol, atol)
            v_new = np.exp(lnv_new)
            if update_v:
                v[cell_idx[i]] = v_new
            ml_new += n[cell_idx[i]] * v_new * const.rho_w

        dml_dt = (ml_new - ml_old) / dt
        dqv_dt_corr = - dml_dt / m_d_mean
        dthd_dt_corr = phys.dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt_pred + dqv_dt_corr)
        thd_err_est = np.maximum(thd_err_est, np.abs(dt * dthd_dt_corr))
        thd += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
        qv += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
        ml_old = ml_new

    return qv, thd, thd_err_est


@numba.njit(fastmath=conf.NUMBA_FASTMATH, error_model=conf.NUMBA_ERROR_MODEL)
def within_tolerance(error_estimate, value, rtol, atol):
    return error_estimate < atol + rtol * np.abs(value)


@numba.njit(fastmath=conf.NUMBA_FASTMATH, error_model=conf.NUMBA_ERROR_MODEL)
def bisec(minfun, a, interval, args, rtol, atol):
        b = a + interval

        i = 0
        while minfun(a, *args) * minfun(b, *args) > 0:
            i += 1
            b = a + interval * 2**i

        if b < a:
            a, b = b, a

        fa = minfun(a, *args) # TODO: computed above

        iter = 0
        while not within_tolerance(error_estimate=b-a, value=(a+b)/2, rtol=rtol, atol=atol):
            lnv_new = (a + b) / 2
            f = minfun(lnv_new, *args)
            if f * fa > 0:
                a = lnv_new
            else:
                b = lnv_new
            iter += 1
        lnv_new = (a + b) / 2
        return lnv_new


@numba.njit(fastmath=conf.NUMBA_FASTMATH, error_model=conf.NUMBA_ERROR_MODEL)
def _minfun(lnv_new,
    lnv_old, dt, T, p, RH, kappa, rd):
    return lnv_old - lnv_new + dt * phys.dlnv_dt(lnv_new, T, p, RH, kappa, rd)
