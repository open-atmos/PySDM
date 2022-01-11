"""
condensation/evaporation solver drop-in replacement implemented using
 SciPy adaptive-timestep BDF solver, for use in tests only
"""
import types
import warnings
from functools import lru_cache
import numpy as np
import numba
import scipy.integrate
from PySDM.physics.constants_defaults import rho_w, K0, T0, PI_4_3
from PySDM.backends import Numba
from PySDM.backends.impl_numba.conf import JIT_FLAGS

idx_thd = 0
idx_x = 1
rtol = 1e-4


def patch_particulator(particulator):
    particulator.condensation_solver = _make_solve(particulator.formulae)
    particulator.condensation = types.MethodType(_bdf_condensation, particulator)


def _bdf_condensation(particulator, rtol_x, rtol_thd, counters, RH_max, success, cell_order):
    func = Numba._condensation
    if not numba.config.DISABLE_JIT:  # pylint: disable=no-member
        func = func.py_func
    func(
        solver=particulator.condensation_solver,
        n_threads=1,
        n_cell=particulator.mesh.n_cell,
        cell_start_arg=particulator.attributes.cell_start.data,
        v=particulator.attributes["volume"].data,
        v_cr=None,
        n=particulator.attributes['n'].data,
        vdry=particulator.attributes["dry volume"].data,
        idx=particulator.attributes._ParticleAttributes__idx.data,
        rhod=particulator.environment["rhod"].data,
        thd=particulator.environment["thd"].data,
        qv=particulator.environment["qv"].data,
        dv_mean=particulator.environment.dv,
        prhod=particulator.environment.get_predicted("rhod").data,
        pthd=particulator.environment.get_predicted("thd").data,
        pqv=particulator.environment.get_predicted("qv").data,
        kappa=particulator.attributes["kappa"].data,
        f_org=particulator.attributes["dry volume organic fraction"].data,
        rtol_x=rtol_x,
        rtol_thd=rtol_thd,
        timestep=particulator.dt,
        counter_n_substeps=counters['n_substeps'],
        counter_n_activating=counters['n_activating'],
        counter_n_deactivating=counters['n_deactivating'],
        counter_n_ripening=counters['n_ripening'],
        cell_order=cell_order,
        RH_max=RH_max.data,
        success=success.data
    )


@lru_cache()
def _make_solve(formulae):
    x = formulae.condensation_coordinate.x
    volume = formulae.condensation_coordinate.volume
    dx_dt = formulae.condensation_coordinate.dx_dt
    pvs_C = formulae.saturation_vapour_pressure.pvs_Celsius
    lv = formulae.latent_heat.lv
    r_dr_dt = formulae.drop_growth.r_dr_dt
    RH_eq = formulae.hygroscopicity.RH_eq
    sigma = formulae.surface_tension.sigma
    phys_radius = formulae.trivia.radius
    phys_T = formulae.state_variable_triplet.T
    phys_p = formulae.state_variable_triplet.p
    phys_pv = formulae.state_variable_triplet.pv
    phys_dthd_dt = formulae.state_variable_triplet.dthd_dt
    phys_lambdaD = formulae.diffusion_kinetics.lambdaD
    phys_lambdaK = formulae.diffusion_kinetics.lambdaK
    phys_DK = formulae.diffusion_kinetics.DK
    phys_D = formulae.diffusion_thermics.D

    @numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
    def _ql(n, x, m_d_mean):
        return np.sum(n * volume(x)) * rho_w / m_d_mean

    @numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
    def _impl(dy_dt, x, T, p, n, RH, kappa, f_org, dry_volume, thd,
              dot_thd, dot_qv, m_d_mean, rhod_mean, pvs, lv):
        DTp = phys_D(T, p)
        lambdaD = phys_lambdaD(DTp, T)
        lambdaK = phys_lambdaK(T, p)
        for i, x_i in enumerate(x):
            v = volume(x_i)
            r = phys_radius(v)
            Dr = phys_DK(DTp, r, lambdaD)
            Kr = phys_DK(K0, r, lambdaK)
            sgm = sigma(T, v, dry_volume[i], f_org[i])
            dy_dt[idx_x + i] = dx_dt(
                x_i,
                r_dr_dt(
                    RH_eq(r, T, kappa[i], dry_volume[i] / PI_4_3, sgm),
                    T, RH, lv, pvs, Dr, Kr
                )
            )
        dqv_dt = dot_qv - np.sum(n * volume(x) * dy_dt[idx_x:]) * rho_w / m_d_mean
        dy_dt[idx_thd] = dot_thd + phys_dthd_dt(rhod_mean, thd, T, dqv_dt, lv)

    @numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
    def _odesys(t, y, kappa, f_org, dry_volume, n,
                dthd_dt, dqv_dt, m_d_mean, rhod_mean, qt):
        thd = y[idx_thd]
        x = y[idx_x:]

        qv = qt + dqv_dt * t - _ql(n, x, m_d_mean)
        T = phys_T(rhod_mean, thd)
        p = phys_p(rhod_mean, T, qv)
        pv = phys_pv(p, qv)
        pvs = pvs_C(T - T0)
        RH = pv / pvs

        dy_dt = np.empty_like(y)
        _impl(
            dy_dt, x, T, p, n, RH, kappa, f_org, dry_volume, thd,
            dthd_dt, dqv_dt, m_d_mean, rhod_mean, pvs, lv(T)
        )
        return dy_dt

    def solve(
            v, _, n, vdry,
            cell_idx, kappa, f_org, thd, qv,
            dthd_dt, dqv_dt, m_d_mean, rhod_mean,
            __, ___, dt, ____
    ):
        n_sd_in_cell = len(cell_idx)
        y0 = np.empty(n_sd_in_cell + idx_x)
        y0[idx_thd] = thd
        y0[idx_x:] = x(v[cell_idx])
        qt = qv + _ql(n[cell_idx], y0[idx_x:], m_d_mean)
        args = (
            kappa[cell_idx], f_org[cell_idx], vdry[cell_idx], n[cell_idx],
            dthd_dt, dqv_dt, m_d_mean, rhod_mean, qt
        )
        if dthd_dt == 0 and dqv_dt == 0 and (_odesys(0, y0, *args)[idx_x] == 0).all():
            y1 = y0
        else:
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("ignore")
                integ = scipy.integrate.solve_ivp(
                    fun=_odesys,
                    args=args,
                    t_span=(0, dt),
                    t_eval=(dt,),
                    y0=y0,
                    rtol=rtol,
                    atol=0,
                    method="BDF"
                )
            assert integ.success, integ.message
            y1 = integ.y[:, 0]

        m_new = 0
        for i in range(n_sd_in_cell):
            v_new = volume(y1[idx_x + i])
            m_new += n[cell_idx[i]] * v_new * rho_w
            v[cell_idx[i]] = v_new

        return integ.success, qt - m_new / m_d_mean, y1[idx_thd], 1, 1, 1, 1, np.nan

    return solve
