"""
condensation/evaporation solver drop-in replacement implemented using
 SciPy adaptive-timestep ODE solver, for use in tests only
"""

import types
from functools import lru_cache

import numba
import numpy as np
import scipy.integrate

from PySDM.backends import Numba
from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM.physics.constants_defaults import PI_4_3, T0

idx_thd = 0
idx_x = 1
rtol = 1e-6


def patch_particulator(particulator):
    particulator.condensation_solver = _make_solve(particulator.formulae)
    particulator.condensation = types.MethodType(_condensation, particulator)


def _condensation(
    particulator, *, rtol_x, rtol_thd, counters, RH_max, success, cell_order
):
    func = Numba._condensation
    if not numba.config.DISABLE_JIT:  # pylint: disable=no-member
        func = func.py_func
    func(
        solver=particulator.condensation_solver,
        n_threads=1,
        n_cell=particulator.mesh.n_cell,
        cell_start_arg=particulator.attributes.cell_start.data,
        water_mass=particulator.attributes["water mass"].data,
        v_cr=None,
        multiplicity=particulator.attributes["multiplicity"].data,
        vdry=particulator.attributes["dry volume"].data,
        idx=particulator.attributes._ParticleAttributes__idx.data,
        rhod=particulator.environment["rhod"].data,
        thd=particulator.environment["thd"].data,
        water_vapour_mixing_ratio=particulator.environment[
            "water_vapour_mixing_ratio"
        ].data,
        dv_mean=particulator.environment.dv,
        prhod=particulator.environment.get_predicted("rhod").data,
        pthd=particulator.environment.get_predicted("thd").data,
        predicted_water_vapour_mixing_ratio=particulator.environment.get_predicted(
            "water_vapour_mixing_ratio"
        ).data,
        kappa=particulator.attributes["kappa"].data,
        f_org=particulator.attributes["dry volume organic fraction"].data,
        rtol_x=rtol_x,
        rtol_thd=rtol_thd,
        timestep=particulator.dt,
        counter_n_substeps=counters["n_substeps"],
        counter_n_activating=counters["n_activating"],
        counter_n_deactivating=counters["n_deactivating"],
        counter_n_ripening=counters["n_ripening"],
        cell_order=cell_order,
        RH_max=RH_max.data,
        success=success.data,
    )
    particulator.attributes.mark_updated("water mass")


@lru_cache()
def _make_solve(formulae):  # pylint: disable=too-many-statements,too-many-locals
    x = formulae.condensation_coordinate.x
    volume_of_x = formulae.condensation_coordinate.volume
    volume_to_mass = formulae.particle_shape_and_density.volume_to_mass
    mass_to_volume = formulae.particle_shape_and_density.mass_to_volume
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
    phys_diff_kin_D = formulae.diffusion_kinetics.D
    phys_diff_kin_K = formulae.diffusion_kinetics.K
    phys_D = formulae.diffusion_thermics.D
    phys_K = formulae.diffusion_thermics.K

    @numba.njit(**{**JIT_FLAGS, **{"parallel": False}})
    def _liquid_water_mixing_ratio(n, x, m_d_mean):
        return np.sum(n * volume_to_mass(volume_of_x(x))) / m_d_mean

    @numba.njit(**{**JIT_FLAGS, **{"parallel": False}})
    def _impl(  # pylint: disable=too-many-arguments,too-many-locals
        dy_dt,
        x,
        T,
        p,
        n,
        RH,
        kappa,
        f_org,
        dry_volume,
        thd,
        dot_thd,
        dot_water_vapour_mixing_ratio,
        m_d_mean,
        rhod,
        pvs,
        lv,
    ):
        DTp = phys_D(T, p)
        KTp = phys_K(T, p)
        lambdaD = phys_lambdaD(DTp, T)
        lambdaK = phys_lambdaK(T, p)
        for i, x_i in enumerate(x):
            v = volume_of_x(x_i)
            r = phys_radius(v)
            Dr = phys_diff_kin_D(DTp, r, lambdaD)
            Kr = phys_diff_kin_K(KTp, r, lambdaK)
            sgm = sigma(T, v, dry_volume[i], f_org[i])
            dy_dt[idx_x + i] = dx_dt(
                x_i,
                r_dr_dt(
                    RH_eq(r, T, kappa[i], dry_volume[i] / PI_4_3, sgm),
                    T,
                    RH,
                    lv,
                    pvs,
                    Dr,
                    Kr,
                ),
            )
        d_water_vapour_mixing_ratio__dt = (
            dot_water_vapour_mixing_ratio
            - np.sum(n * volume_to_mass(volume_of_x(x)) * dy_dt[idx_x:]) / m_d_mean
        )
        dy_dt[idx_thd] = dot_thd + phys_dthd_dt(
            rhod, thd, T, d_water_vapour_mixing_ratio__dt, lv
        )

    @numba.njit(**{**JIT_FLAGS, **{"parallel": False}})
    def _odesys(  # pylint: disable=too-many-arguments,too-many-locals
        t,
        y,
        kappa,
        f_org,
        dry_volume,
        n,
        dthd_dt,
        d_water_vapour_mixing_ratio__dt,
        drhod_dt,
        m_d_mean,
        rhod,
        qt,
    ):
        thd = y[idx_thd]
        x = y[idx_x:]

        water_vapour_mixing_ratio = (
            qt
            + d_water_vapour_mixing_ratio__dt * t
            - _liquid_water_mixing_ratio(n, x, m_d_mean)
        )
        rhod += drhod_dt * t
        T = phys_T(rhod, thd)
        p = phys_p(rhod, T, water_vapour_mixing_ratio)
        pv = phys_pv(p, water_vapour_mixing_ratio)
        pvs = pvs_C(T - T0)
        RH = pv / pvs

        dy_dt = np.empty_like(y)
        _impl(
            dy_dt,
            x,
            T,
            p,
            n,
            RH,
            kappa,
            f_org,
            dry_volume,
            thd,
            dthd_dt,
            d_water_vapour_mixing_ratio__dt,
            m_d_mean,
            rhod,
            pvs,
            lv(T),
        )
        return dy_dt

    def solve(  # pylint: disable=too-many-arguments,too-many-locals
        water_mass,
        __,
        multiplicity,
        vdry,
        cell_idx,
        kappa,
        f_org,
        thd,
        water_vapour_mixing_ratio,
        rhod,
        dthd_dt,
        d_water_vapour_mixing_ratio__dt,
        drhod_dt,
        m_d_mean,
        ___,
        ____,
        dt,
        _____,
    ):
        n_sd_in_cell = len(cell_idx)
        y0 = np.empty(n_sd_in_cell + idx_x)
        y0[idx_thd] = thd
        y0[idx_x:] = x(mass_to_volume(water_mass[cell_idx]))
        total_water_mixing_ratio = (
            water_vapour_mixing_ratio
            + _liquid_water_mixing_ratio(multiplicity[cell_idx], y0[idx_x:], m_d_mean)
        )
        args = (
            kappa[cell_idx],
            f_org[cell_idx],
            vdry[cell_idx],
            multiplicity[cell_idx],
            dthd_dt,
            d_water_vapour_mixing_ratio__dt,
            drhod_dt,
            m_d_mean,
            rhod,
            total_water_mixing_ratio,
        )
        if (
            dthd_dt == 0
            and d_water_vapour_mixing_ratio__dt == 0
            and drhod_dt == 0
            and (_odesys(0, y0, *args)[idx_x:] == 0).all()
        ):
            y1 = y0
        else:
            integ = scipy.integrate.solve_ivp(
                fun=_odesys,
                args=args,
                t_span=(0, dt),
                t_eval=(dt,),
                y0=y0,
                rtol=rtol,
                atol=0,
                method="LSODA",
            )
            assert integ.success, integ.message
            y1 = integ.y[:, 0]

        m_new = 0
        for i in range(n_sd_in_cell):
            water_mass[cell_idx[i]] = volume_to_mass(volume_of_x(y1[idx_x + i]))
            m_new += multiplicity[cell_idx[i]] * water_mass[cell_idx[i]]

        return (
            integ.success,
            total_water_mixing_ratio - m_new / m_d_mean,
            y1[idx_thd],
            1,
            1,
            1,
            1,
            np.nan,
        )

    return solve
