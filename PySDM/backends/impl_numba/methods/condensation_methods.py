"""
CPU implementation of backend methods for water condensation/evaporation
"""

from collections import namedtuple
import math
from functools import lru_cache

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_numba.toms748 import toms748_solve
from PySDM.backends.impl_numba.warnings import warn

_Counters = namedtuple(
    typename="_Counters",
    field_names=("n_substeps", "n_activating", "n_deactivating", "n_ripening"),
)
_Attributes = namedtuple(
    typename="_Attributes",
    field_names=(
        "water_mass",
        "v_cr",
        "multiplicity",
        "vdry",
        "kappa",
        "f_org",
        "reynolds_number",
    ),
)
_CellData = namedtuple(
    typename="_CellData",
    field_names=(
        "rhod",
        "thd",
        "water_vapour_mixing_ratio",
        "dv_mean",
        "prhod",
        "pthd",
        "predicted_water_vapour_mixing_ratio",
        "air_density",
        "air_dynamic_viscosity",
    ),
)
_RelativeTolerances = namedtuple(
    typename="_RelativeTolerances", field_names=("x", "thd")
)


class CondensationMethods(BackendMethods):
    # pylint: disable=unused-argument
    @staticmethod
    def condensation(**kwargs):
        n_threads = min(numba.get_num_threads(), kwargs["n_cell"])
        CondensationMethods._condensation(
            solver=kwargs["solver"],
            n_threads=n_threads,
            n_cell=kwargs["n_cell"],
            cell_start_arg=kwargs["cell_start_arg"].data,
            attributes=_Attributes(
                water_mass=kwargs["water_mass"].data,
                v_cr=kwargs["v_cr"].data,
                multiplicity=kwargs["multiplicity"].data,
                vdry=kwargs["vdry"].data,
                kappa=kwargs["kappa"].data,
                f_org=kwargs["f_org"].data,
                reynolds_number=kwargs["reynolds_number"].data,
            ),
            idx=kwargs["idx"].data,
            cell_data=_CellData(
                rhod=kwargs["rhod"].data,
                thd=kwargs["thd"].data,
                water_vapour_mixing_ratio=kwargs["water_vapour_mixing_ratio"].data,
                dv_mean=kwargs["dv"],
                prhod=kwargs["prhod"].data,
                pthd=kwargs["pthd"].data,
                predicted_water_vapour_mixing_ratio=(
                    kwargs["predicted_water_vapour_mixing_ratio"].data
                ),
                air_density=kwargs["air_density"].data,
                air_dynamic_viscosity=kwargs["air_dynamic_viscosity"].data,
            ),
            rtols=_RelativeTolerances(
                x=kwargs["rtol_x"],
                thd=kwargs["rtol_thd"],
            ),
            timestep=kwargs["timestep"],
            counters=_Counters(
                n_substeps=kwargs["counters"]["n_substeps"].data,
                n_activating=kwargs["counters"]["n_activating"].data,
                n_deactivating=kwargs["counters"]["n_deactivating"].data,
                n_ripening=kwargs["counters"]["n_ripening"].data,
            ),
            cell_order=kwargs["cell_order"],
            RH_max=kwargs["RH_max"].data,
            success=kwargs["success"].data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"cache": False}})
    def _condensation(  # pylint: disable=too-many-locals
        *,
        solver,
        n_threads,
        n_cell,
        cell_start_arg,
        attributes,
        cell_data,
        idx,
        rtols,
        timestep,
        counters,
        cell_order,
        RH_max,
        success,
    ):
        # arrays within namedtuples in prange loops do not work
        # https://github.com/numba/numba/issues/5872
        cdt_predicted_water_vapour_mixing_ratio = (
            cell_data.predicted_water_vapour_mixing_ratio
        )
        cdt_pthd = cell_data.pthd
        cnt_n_substeps = counters.n_substeps
        cnt_n_activating = counters.n_activating
        cnt_n_deactivating = counters.n_deactivating
        cnt_n_ripening = counters.n_ripening

        for thread_id in numba.prange(n_threads):  # pylint: disable=not-an-iterable
            for i in range(thread_id, n_cell, n_threads):
                cell_id = cell_order[i]
                cell_start = cell_start_arg[cell_id]
                cell_end = cell_start_arg[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                (
                    success[cell_id],
                    cdt_predicted_water_vapour_mixing_ratio[cell_id],
                    cdt_pthd[cell_id],
                    cnt_n_substeps[cell_id],
                    cnt_n_activating[cell_id],
                    cnt_n_deactivating[cell_id],
                    cnt_n_ripening[cell_id],
                    RH_max[cell_id],
                ) = solver(
                    attributes=attributes,
                    cell_idx=idx[cell_start:cell_end],
                    thd=cell_data.thd[cell_id],
                    water_vapour_mixing_ratio=cell_data.water_vapour_mixing_ratio[
                        cell_id
                    ],
                    rhod=cell_data.rhod[cell_id],
                    dthd_dt=(cell_data.pthd[cell_id] - cell_data.thd[cell_id])
                    / timestep,
                    d_water_vapour_mixing_ratio__dt=(
                        cell_data.predicted_water_vapour_mixing_ratio[cell_id]
                        - cell_data.water_vapour_mixing_ratio[cell_id]
                    )
                    / timestep,
                    drhod_dt=(cell_data.prhod[cell_id] - cell_data.rhod[cell_id])
                    / timestep,
                    m_d=(
                        (cell_data.prhod[cell_id] + cell_data.rhod[cell_id])
                        / 2
                        * cell_data.dv_mean
                    ),
                    air_density=cell_data.air_density[cell_id],
                    air_dynamic_viscosity=cell_data.air_dynamic_viscosity[cell_id],
                    rtols=rtols,
                    timestep=timestep,
                    n_substeps=counters.n_substeps[cell_id],
                )

    @staticmethod
    def make_adapt_substeps(
        *, jit_flags, formulae, timestep, step_fake, dt_range, fuse, multiplier
    ):
        if not isinstance(multiplier, int):
            raise ValueError()
        if dt_range[1] > timestep:
            dt_range = (dt_range[0], timestep)
        if dt_range[0] == 0:
            raise NotImplementedError()
        n_substeps_max = math.floor(timestep / dt_range[0])
        n_substeps_min = math.ceil(timestep / dt_range[1])

        @numba.njit(**jit_flags)
        def adapt_substeps(args, n_substeps, thd, rtol_thd):
            n_substeps = np.maximum(n_substeps_min, n_substeps // multiplier)
            success = False
            for burnout in range(fuse + 1):
                if burnout == fuse:
                    return warn(
                        "burnout (long)",
                        __file__,
                        context=(
                            "thd",
                            thd,
                        ),
                        return_value=(0, False),
                    )
                thd_new_long, success = step_fake(args, timestep, n_substeps)
                if success:
                    break
                n_substeps *= multiplier
            for burnout in range(fuse + 1):
                if burnout == fuse:
                    return warn("burnout (short)", __file__, return_value=(0, False))
                thd_new_short, success = step_fake(
                    args, timestep, n_substeps * multiplier
                )
                if not success:
                    return warn("short failed", __file__, return_value=(0, False))
                dthd_long = thd_new_long - thd
                dthd_short = thd_new_short - thd
                error_estimate = np.abs(dthd_long - multiplier * dthd_short)
                thd_new_long = thd_new_short
                if formulae.trivia__within_tolerance(error_estimate, thd, rtol_thd):
                    break
                n_substeps *= multiplier
                if n_substeps > n_substeps_max:
                    break
            return np.minimum(n_substeps_max, n_substeps), success

        return adapt_substeps

    @staticmethod
    def make_step_fake(jit_flags, step_impl):
        @numba.njit(**jit_flags)
        def step_fake(args, dt, n_substeps):
            dt /= n_substeps
            _, thd_new, _, _, _, _, success = step_impl(*args, dt, 1, True)
            return thd_new, success

        return step_fake

    @staticmethod
    def make_step(jit_flags, step_impl):
        @numba.njit(**jit_flags)
        def step(args, dt, n_substeps):
            return step_impl(*args, dt, n_substeps, False)

        return step

    @staticmethod
    def make_step_impl(
        *,
        jit_flags,
        formulae,
        calculate_ml_old,
        calculate_ml_new,
    ):
        @numba.njit(**jit_flags)
        def step_impl(  # pylint: disable=too-many-arguments,too-many-locals
            attributes,
            cell_idx,
            thd,
            water_vapour_mixing_ratio,
            rhod,
            dthd_dt_pred,
            d_water_vapour_mixing_ratio__dt_predicted,
            drhod_dt,
            m_d,
            air_density,
            air_dynamic_viscosity,
            rtol_x,
            timestep,
            n_substeps,
            fake,
        ):
            timestep /= n_substeps
            ml_old = calculate_ml_old(
                attributes.water_mass, attributes.multiplicity, cell_idx
            )
            count_activating, count_deactivating, count_ripening = 0, 0, 0
            RH_max = 0
            success = True
            for _ in range(n_substeps):
                # note: no example yet showing that the trapezoidal scheme brings any improvement
                thd += timestep * dthd_dt_pred / 2
                water_vapour_mixing_ratio += (
                    timestep * d_water_vapour_mixing_ratio__dt_predicted / 2
                )
                rhod += timestep * drhod_dt / 2
                T = formulae.state_variable_triplet__T(rhod, thd)
                p = formulae.state_variable_triplet__p(
                    rhod, T, water_vapour_mixing_ratio
                )
                pv = formulae.state_variable_triplet__pv(p, water_vapour_mixing_ratio)
                lv = formulae.latent_heat__lv(T)
                pvs = formulae.saturation_vapour_pressure__pvs_Celsius(
                    T - formulae.constants.T0
                )
                DTp = formulae.diffusion_thermics__D(T, p)
                RH = pv / pvs
                Sc = formulae.trivia__air_schmidt_number(
                    dynamic_viscosity=air_dynamic_viscosity,
                    diffusivity=DTp,
                    density=air_density,
                )
                (
                    ml_new,
                    success_within_substep,
                    n_activating,
                    n_deactivating,
                    n_ripening,
                ) = calculate_ml_new(
                    attributes,
                    timestep,
                    fake,
                    T,
                    p,
                    RH,
                    Sc,
                    cell_idx,
                    lv,
                    pvs,
                    DTp,
                    formulae.diffusion_thermics__K(T, p),
                    rtol_x,
                )
                dml_dt = (ml_new - ml_old) / timestep
                d_water_vapour_mixing_ratio__dt_corrected = -dml_dt / m_d
                dthd_dt_corr = formulae.state_variable_triplet__dthd_dt(
                    rhod=rhod,
                    thd=thd,
                    T=T,
                    d_water_vapour_mixing_ratio__dt=d_water_vapour_mixing_ratio__dt_corrected,
                    lv=lv,
                )

                thd += timestep * (dthd_dt_pred / 2 + dthd_dt_corr)
                water_vapour_mixing_ratio += timestep * (
                    d_water_vapour_mixing_ratio__dt_predicted / 2
                    + d_water_vapour_mixing_ratio__dt_corrected
                )
                rhod += timestep * drhod_dt / 2
                ml_old = ml_new
                count_activating += n_activating
                count_deactivating += n_deactivating
                count_ripening += n_ripening
                RH_max = max(RH_max, RH)
                success = success and success_within_substep
            return (
                water_vapour_mixing_ratio,
                thd,
                count_activating,
                count_deactivating,
                count_ripening,
                RH_max,
                success,
            )

        return step_impl

    @staticmethod
    def make_calculate_ml_old(jit_flags):
        @numba.njit(**jit_flags)
        def calculate_ml_old(water_mass, multiplicity, cell_idx):
            result = 0
            for drop in cell_idx:
                if water_mass[drop] > 0:
                    result += multiplicity[drop] * water_mass[drop]
            return result

        return calculate_ml_old

    @staticmethod
    def make_calculate_ml_new(  # pylint: disable=too-many-statements
        *,
        formulae,
        jit_flags,
        max_iters,
        RH_rtol,
    ):
        @numba.njit(**jit_flags)
        def minfun(  # pylint: disable=too-many-arguments,too-many-locals
            x_new,
            x_old,
            timestep,
            kappa,
            f_org,
            rd3,
            temperature,
            RH,
            lv,
            pvs,
            D,
            K,
            ventilation_factor,
        ):
            volume = formulae.condensation_coordinate__volume(x_new)
            RH_eq = formulae.hygroscopicity__RH_eq(
                formulae.trivia__radius(volume),
                temperature,
                kappa,
                rd3,
                formulae.surface_tension__sigma(
                    temperature, volume, formulae.constants.PI_4_3 * rd3, f_org
                ),
            )
            r_dr_dt = formulae.drop_growth__r_dr_dt(
                RH_eq,
                temperature,
                RH,
                lv,
                pvs,
                D,
                K,
                ventilation_factor,
            )
            return (
                x_old
                - x_new
                + timestep * formulae.condensation_coordinate__dx_dt(x_new, r_dr_dt)
            )

        @numba.njit(**jit_flags)
        def calculate_ml_new(  # pylint: disable=too-many-branches,too-many-arguments,too-many-locals
            attributes,
            timestep,
            fake,
            T,
            p,
            RH,
            Sc,
            cell_idx,
            lv,
            pvs,
            DTp,
            KTp,
            rtol_x,
        ):
            result = 0
            n_activating = 0
            n_deactivating = 0
            n_activated_and_growing = 0
            success = True
            lambdaK = formulae.diffusion_kinetics__lambdaK(T, p)
            lambdaD = formulae.diffusion_kinetics__lambdaD(DTp, T)
            for drop in cell_idx:
                v_drop = formulae.particle_shape_and_density__mass_to_volume(
                    attributes.water_mass[drop]
                )
                if v_drop < 0:
                    continue
                x_old = formulae.condensation_coordinate__x(v_drop)
                r_old = formulae.trivia__radius(v_drop)
                x_insane = formulae.condensation_coordinate__x(
                    attributes.vdry[drop] / 100
                )
                rd3 = attributes.vdry[drop] / formulae.constants.PI_4_3
                sgm = formulae.surface_tension__sigma(
                    T, v_drop, attributes.vdry[drop], attributes.f_org[drop]
                )
                RH_eq = formulae.hygroscopicity__RH_eq(
                    r_old, T, attributes.kappa[drop], rd3, sgm
                )
                if not formulae.trivia__within_tolerance(
                    np.abs(RH - RH_eq), RH, RH_rtol
                ):
                    Dr = formulae.diffusion_kinetics__D(DTp, r_old, lambdaD)
                    Kr = formulae.diffusion_kinetics__K(KTp, r_old, lambdaK)
                    ventilation_factor = formulae.ventilation__ventilation_coefficient(
                        sqrt_re_times_cbrt_sc=formulae.trivia__sqrt_re_times_cbrt_sc(
                            Re=attributes.reynolds_number[drop],
                            Sc=Sc,
                        )
                    )
                    args = (
                        x_old,
                        timestep,
                        attributes.kappa[drop],
                        attributes.f_org[drop],
                        rd3,
                        T,
                        RH,
                        lv,
                        pvs,
                        Dr,
                        Kr,
                        ventilation_factor,
                    )
                    r_dr_dt_old = formulae.drop_growth__r_dr_dt(
                        RH_eq,
                        T,
                        RH,
                        lv,
                        pvs,
                        Dr,
                        Kr,
                        ventilation_factor,
                    )
                    dx_old = timestep * formulae.condensation_coordinate__dx_dt(
                        x_old, r_dr_dt_old
                    )
                else:
                    dx_old = 0.0
                if dx_old == 0:
                    x_new = x_old
                else:
                    a = x_old
                    b = max(x_insane, a + dx_old)
                    fa = minfun(a, *args)
                    fb = minfun(b, *args)

                    counter = 0
                    while not fa * fb < 0:
                        counter += 1
                        if counter > max_iters:
                            if not fake:
                                warn(
                                    "failed to find interval",
                                    __file__,
                                    context=(
                                        "T",
                                        T,
                                        "p",
                                        p,
                                        "RH",
                                        RH,
                                        "a",
                                        a,
                                        "b",
                                        b,
                                        "fa",
                                        fa,
                                        "fb",
                                        fb,
                                    ),
                                )
                            success = False
                            break
                        b = max(x_insane, a + math.ldexp(dx_old, counter))
                        fb = minfun(b, *args)

                    if not success:
                        break
                    if a != b:
                        if a > b:
                            a, b = b, a
                            fa, fb = fb, fa

                        x_new, iters_taken = toms748_solve(
                            minfun,
                            args,
                            a,
                            b,
                            fa,
                            fb,
                            rtol_x,
                            max_iters,
                            formulae.trivia__within_tolerance,
                        )
                        if iters_taken in (-1, max_iters):
                            if not fake:
                                warn("TOMS failed", __file__)
                            success = False
                            break
                    else:
                        x_new = x_old

                mass_new = formulae.particle_shape_and_density__volume_to_mass(
                    formulae.condensation_coordinate__volume(x_new)
                )
                mass_cr = formulae.particle_shape_and_density__volume_to_mass(
                    attributes.v_cr[drop]
                )
                result += attributes.multiplicity[drop] * mass_new
                if not fake:
                    if mass_new > mass_cr and mass_new > attributes.water_mass[drop]:
                        n_activated_and_growing += attributes.multiplicity[drop]
                    if mass_new > mass_cr > attributes.water_mass[drop]:
                        n_activating += attributes.multiplicity[drop]
                    if mass_new < mass_cr < attributes.water_mass[drop]:
                        n_deactivating += attributes.multiplicity[drop]
                    attributes.water_mass[drop] = mass_new
            n_ripening = n_activated_and_growing if n_deactivating > 0 else 0
            return result, success, n_activating, n_deactivating, n_ripening

        return calculate_ml_new

    # pylint disable=unused-argument
    def make_condensation_solver(
        self,
        timestep,
        n_cell,
        *,
        dt_range,
        adaptive,
        fuse,
        multiplier,
        RH_rtol,
        max_iters,
    ):
        return CondensationMethods.make_condensation_solver_impl(
            formulae=self.formulae_flattened,
            timestep=timestep,
            dt_range=dt_range,
            adaptive=adaptive,
            fuse=fuse,
            multiplier=multiplier,
            RH_rtol=RH_rtol,
            max_iters=max_iters,
        )

    @staticmethod
    @lru_cache()
    def make_condensation_solver_impl(
        *,
        formulae,
        timestep,
        dt_range,
        adaptive,
        fuse,
        multiplier,
        RH_rtol,
        max_iters,
    ):
        jit_flags = {
            **conf.JIT_FLAGS,
            **{"parallel": False, "cache": False, "fastmath": formulae.fastmath},
        }

        step_impl = CondensationMethods.make_step_impl(
            jit_flags=jit_flags,
            formulae=formulae,
            calculate_ml_old=CondensationMethods.make_calculate_ml_old(jit_flags),
            calculate_ml_new=CondensationMethods.make_calculate_ml_new(
                jit_flags=jit_flags,
                formulae=formulae,
                max_iters=max_iters,
                RH_rtol=RH_rtol,
            ),
        )
        step_fake = CondensationMethods.make_step_fake(jit_flags, step_impl)
        adapt_substeps = CondensationMethods.make_adapt_substeps(
            jit_flags=jit_flags,
            formulae=formulae,
            timestep=timestep,
            step_fake=step_fake,
            dt_range=dt_range,
            fuse=fuse,
            multiplier=multiplier,
        )
        step = CondensationMethods.make_step(jit_flags, step_impl)

        @numba.njit(**jit_flags)
        def solve(  # pylint: disable=too-many-arguments,too-many-locals
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
            args = (
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
                rtols.x,
            )
            success = True
            if adaptive:
                n_substeps, success = adapt_substeps(args, n_substeps, thd, rtols.thd)
            if success:
                (
                    water_vapour_mixing_ratio,
                    thd,
                    n_activating,
                    n_deactivating,
                    n_ripening,
                    RH_max,
                    success,
                ) = step(args, timestep, n_substeps)
            else:
                n_activating, n_deactivating, n_ripening, RH_max = -1, -1, -1, -1
            return (
                success,
                water_vapour_mixing_ratio,
                thd,
                n_substeps,
                n_activating,
                n_deactivating,
                n_ripening,
                RH_max,
            )

        return solve
