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
from PySDM.backends.impl_numba.methods.condensation_methods import (
    _RelativeTolerances,
    _Attributes,
    _CellData,
    _Counters,
)
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
        attributes=_Attributes(
            water_mass=particulator.attributes["water mass"].data,
            v_cr=None,
            multiplicity=particulator.attributes["multiplicity"].data,
            vdry=particulator.attributes["dry volume"].data,
            kappa=particulator.attributes["kappa"].data,
            f_org=particulator.attributes["dry volume organic fraction"].data,
            reynolds_number=particulator.attributes["Reynolds number"].data,
        ),
        idx=particulator.attributes._ParticleAttributes__idx.data,
        cell_data=_CellData(
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
            air_density=particulator.environment["air density"].data,
            air_dynamic_viscosity=particulator.environment[
                "air dynamic viscosity"
            ].data,
        ),
        rtols=_RelativeTolerances(
            x=rtol_x,
            thd=rtol_thd,
        ),
        timestep=particulator.dt,
        counters=_Counters(
            n_substeps=counters["n_substeps"],
            n_activating=counters["n_activating"],
            n_deactivating=counters["n_deactivating"],
            n_ripening=counters["n_ripening"],
        ),
        cell_order=cell_order,
        RH_max=RH_max.data,
        success=success.data,
    )
    particulator.attributes.mark_updated("water mass")


@lru_cache()
def _make_solve(formulae):  # pylint: disable=too-many-statements,too-many-locals
    jit_formulae = formulae.flatten

    @numba.njit(**{**JIT_FLAGS, **{"parallel": False}})
    def _liquid_water_mixing_ratio(n, x, m_d_mean):
        return (
            np.sum(
                n
                * jit_formulae.particle_shape_and_density__volume_to_mass(
                    jit_formulae.condensation_coordinate__volume(x)
                )
            )
            / m_d_mean
        )

    @numba.njit(**{**JIT_FLAGS, **{"parallel": False}})
    def _impl(  # pylint: disable=too-many-arguments,too-many-locals
        dy_dt,
        x,
        T,
        p,
        n,
        RH,
        reynolds_number,
        kappa,
        f_org,
        dry_volume,
        thd,
        dot_thd,
        dot_water_vapour_mixing_ratio,
        m_d_mean,
        air_density,
        air_dynamic_viscosity,
        rhod,
        pvs,
        lv,
    ):
        DTp = jit_formulae.diffusion_thermics__D(T, p)
        KTp = jit_formulae.diffusion_thermics__K(T, p)
        lambdaD = jit_formulae.diffusion_kinetics__lambdaD(DTp, T)
        lambdaK = jit_formulae.diffusion_kinetics__lambdaK(T, p)
        schmidt_number = jit_formulae.trivia__air_schmidt_number(
            dynamic_viscosity=air_dynamic_viscosity,
            diffusivity=DTp,
            density=air_density,
        )
        for i, x_i in enumerate(x):
            v = jit_formulae.condensation_coordinate__volume(x_i)
            r = jit_formulae.trivia__radius(v)
            Dr = jit_formulae.diffusion_kinetics__D(DTp, r, lambdaD)
            Kr = jit_formulae.diffusion_kinetics__K(KTp, r, lambdaK)
            ventilation_factor = jit_formulae.ventilation__ventilation_coefficient(
                sqrt_re_times_cbrt_sc=jit_formulae.trivia__sqrt_re_times_cbrt_sc(
                    Re=reynolds_number[i],
                    Sc=schmidt_number,
                )
            )
            sgm = jit_formulae.surface_tension__sigma(T, v, dry_volume[i], f_org[i])
            dy_dt[idx_x + i] = jit_formulae.condensation_coordinate__dx_dt(
                x_i,
                jit_formulae.drop_growth__r_dr_dt(
                    jit_formulae.hygroscopicity__RH_eq(
                        r, T, kappa[i], dry_volume[i] / PI_4_3, sgm
                    ),
                    T,
                    RH,
                    lv,
                    pvs,
                    Dr,
                    Kr,
                    ventilation_factor,
                ),
            )
        d_water_vapour_mixing_ratio__dt = (
            dot_water_vapour_mixing_ratio
            - np.sum(
                n
                * jit_formulae.particle_shape_and_density__volume_to_mass(
                    jit_formulae.condensation_coordinate__volume(x)
                )
                * dy_dt[idx_x:]
            )
            / m_d_mean
        )
        dy_dt[idx_thd] = dot_thd + jit_formulae.state_variable_triplet__dthd_dt(
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
        reynolds_number,
        dthd_dt,
        d_water_vapour_mixing_ratio__dt,
        drhod_dt,
        m_d_mean,
        air_density,
        air_dynamic_viscosity,
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
        T = jit_formulae.state_variable_triplet__T(rhod, thd)
        p = jit_formulae.state_variable_triplet__p(rhod, T, water_vapour_mixing_ratio)
        pv = jit_formulae.state_variable_triplet__pv(p, water_vapour_mixing_ratio)
        pvs = jit_formulae.saturation_vapour_pressure__pvs_Celsius(T - T0)
        RH = pv / pvs

        dy_dt = np.empty_like(y)
        _impl(
            dy_dt,
            x,
            T,
            p,
            n,
            RH,
            reynolds_number,
            kappa,
            f_org,
            dry_volume,
            thd,
            dthd_dt,
            d_water_vapour_mixing_ratio__dt,
            m_d_mean,
            air_density,
            air_dynamic_viscosity,
            rhod,
            pvs,
            jit_formulae.latent_heat__lv(T),
        )
        return dy_dt

    def solve(  # pylint: disable=too-many-arguments,too-many-locals,unused-argument
        attributes,
        cell_idx,
        thd,
        water_vapour_mixing_ratio,
        rhod,
        dthd_dt,
        d_water_vapour_mixing_ratio__dt,
        drhod_dt,
        m_d,
        air_density,
        air_dynamic_viscosity,
        rtols,
        timestep,
        n_substeps,
    ):
        n_sd_in_cell = len(cell_idx)
        y0 = np.empty(n_sd_in_cell + idx_x)
        y0[idx_thd] = thd
        y0[idx_x:] = jit_formulae.condensation_coordinate__x(
            jit_formulae.particle_shape_and_density__mass_to_volume(
                attributes.water_mass[cell_idx]
            )
        )
        total_water_mixing_ratio = (
            water_vapour_mixing_ratio
            + _liquid_water_mixing_ratio(
                attributes.multiplicity[cell_idx], y0[idx_x:], m_d
            )
        )
        args = (
            attributes.kappa[cell_idx],
            attributes.f_org[cell_idx],
            attributes.vdry[cell_idx],
            attributes.multiplicity[cell_idx],
            attributes.reynolds_number[cell_idx],
            dthd_dt,
            d_water_vapour_mixing_ratio__dt,
            drhod_dt,
            m_d,
            air_density,
            air_dynamic_viscosity,
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
                t_span=(0, timestep),
                t_eval=(timestep,),
                y0=y0,
                rtol=rtol,
                atol=0,
                method="LSODA",
            )
            assert integ.success, integ.message
            y1 = integ.y[:, 0]

        m_new = 0
        for i in range(n_sd_in_cell):
            attributes.water_mass[cell_idx[i]] = (
                jit_formulae.particle_shape_and_density__volume_to_mass(
                    jit_formulae.condensation_coordinate__volume(y1[idx_x + i])
                )
            )
            m_new += (
                attributes.multiplicity[cell_idx[i]]
                * attributes.water_mass[cell_idx[i]]
            )

        return (
            integ.success,
            total_water_mixing_ratio - m_new / m_d,
            y1[idx_thd],
            1,
            1,
            1,
            1,
            np.nan,
        )

    return solve
