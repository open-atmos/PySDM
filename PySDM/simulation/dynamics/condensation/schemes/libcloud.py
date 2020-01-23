from ....physics import formulae as phys
from ....physics import constants as const
import numba
import numpy as np


class ImplicitInSizeExplicitInTheta:
    def __init__(self, backend, rtol, atol, dt_max):
        self.backend = backend
        self.rtol = rtol
        self.atol = atol
        self.dt_max = dt_max  # TODO: more clever - as a function of supersaturation!

    def step(self,
             v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean
             ):
        return impl(v, n, vdry,
                    cell_idx,
                    dt, kappa,
                    thd, qv,
                    dthd_dt, dqv_dt,
                    m_d_mean, rhod_mean, self.rtol, self.atol, self.dt_max)


@numba.njit()
def _minfun(lnv_new,
    lnv_old, dt, T, p, RH, kappa, rd):
    return lnv_old - lnv_new + dt * phys.dlnv_dt(lnv_new, T, p, RH, kappa, rd)


@numba.njit()
def impl(v, n, vdry,
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
        thd += dt * dthd_dt / 2
        qv += dt * dqv_dt / 2
        T, p, RH = phys.temperature_pressure_RH(rhod_mean, thd, qv)

        for i in range(n_sd_in_cell):
            lnv_old = np.log(v[cell_idx[i]])
            rd = phys.radius(volume=vdry[cell_idx[i]])
            dlnv_old = dt * phys.dlnv_dt(lnv_old, T, p, RH, kappa, rd)
            a = lnv_old
            b = lnv_old + 2 * dlnv_old
            if a > b:
                a, b = b, a
            args = (lnv_old, dt, T, p, RH, kappa, rd)
            fa = _minfun(a, *args)
            if fa * _minfun(b, *args) >= 0:
                lnv_new = lnv_old + dlnv_old
            else:
                while (b - a) / (-a) > rtol or (b - a) > atol:  # TODO: rethink (SciPy definition:  solver keeps the local error estimates less than atol + rtol * abs(y).)
                    lnv_new = (a + b) / 2
                    f = _minfun(lnv_new, *args)
                    if f * fa > 0:
                        a = lnv_new
                    else:
                        b = lnv_new
                lnv_new = (a + b) / 2
            v[cell_idx[i]] = np.exp(lnv_new)

        ml_new = 0
        for i in range(n_sd_in_cell):
            ml_new += n[cell_idx[i]] * v[cell_idx[i]] * const.rho_w

        dml_dt = (ml_new - ml_old) / dt
        thd += dt * (dthd_dt / 2 + phys.dthd_dt(rhod=rhod_mean, thd=thd, T=T, dqv_dt=dqv_dt - dml_dt / m_d_mean))
        qv += dt * (dqv_dt / 2 - dml_dt / m_d_mean)
        ml_old = ml_new

    return qv, thd


