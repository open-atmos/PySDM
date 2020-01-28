from ....physics import formulae as phys
from ....physics import constants as const
import numba
import numpy as np


@numba.njit()
def impl(y, v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean, rtol, atol, dt_max):
    n_sd_in_cell = len(cell_idx)
    n_substeps = np.maximum(1, int(dt / dt_max))
    dt /= n_substeps

    ml_old = 0
    for i in range(n_sd_in_cell):
        ml_old += n[cell_idx[i]] * v[cell_idx[i]] * const.rho_w

    for t in range(n_substeps):
        thd += dt * dthd_dt / 2  # TODO: test showing that it makes sense
        qv += dt * dqv_dt / 2
        T, p, RH = phys.temperature_pressure_RH(rhod_mean, thd, qv)

        for i in range(n_sd_in_cell):
            lnv_old = np.log(v[cell_idx[i]])  # TODO: abstract out coord logic
            rd = phys.radius(volume=vdry[cell_idx[i]])
            dlnv_old = dt * phys.dlnv_dt(lnv_old, T, p, RH, kappa, rd)
            a = lnv_old
            interval = dlnv_old
            args = (lnv_old, dt, T, p, RH, kappa, rd)
            lnv_new = bisec(_minfun, a, interval, args, rtol, atol)
            v[cell_idx[i]] = np.exp(lnv_new)

        ml_new = 0
        for i in range(n_sd_in_cell):
            ml_new += n[cell_idx[i]] * v[cell_idx[i]] * const.rho_w

        dml_dt = (ml_new - ml_old) / dt
        thd += dt * (dthd_dt / 2 + phys.dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt - dml_dt / m_d_mean))
        qv += dt * (dqv_dt / 2 - dml_dt / m_d_mean)
        ml_old = ml_new

    return qv, thd

@numba.njit()
def bisec(minfun, a, interval, args, rtol, atol):
        b = a + interval
        if b < a:
            a, b = b, a
        fa = minfun(a, *args)
        i = 0
        while minfun(b, *args) * fa > 0:
            i += 1
            b = a + interval * 2**i
        iter = 0
        while (b - a) / (-a) > rtol or (b - a) > atol:  # TODO: rethink (SciPy definition:  solver keeps the local error estimates less than atol + rtol * abs(y).)
            lnv_new = (a + b) / 2
            f = minfun(lnv_new, *args)
            if f * fa > 0:
                a = lnv_new
            else:
                b = lnv_new
            iter += 1
        lnv_new = (a + b) / 2
        return lnv_new


@numba.njit()
def _minfun(lnv_new,
    lnv_old, dt, T, p, RH, kappa, rd):
    return lnv_old - lnv_new + dt * phys.dlnv_dt(lnv_new, T, p, RH, kappa, rd)
