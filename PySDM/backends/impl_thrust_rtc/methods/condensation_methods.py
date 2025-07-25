"""
GPU implementation of backend methods for water condensation/evaporation
"""

from functools import cached_property
from typing import Dict, Optional

from PySDM.backends.impl_common.storage_utils import StorageBase
from PySDM.backends.impl_thrust_rtc.bisection import BISECTION
from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods

MINFUN_ARGS_VARS = ("x_old", "dt", "kappa", "f_org", "rd3", "_T", "_RH", "Fk", "Fd")


def arg(name):
    return f"minfun_args[{MINFUN_ARGS_VARS.index(name)}]"


class CondensationMethods(
    ThrustRTCBackendMethods
):  # pylint: disable=too-many-instance-attributes
    keys = (
        "T",
        "p",
        "pv",
        "lv",
        "pvs",
        "RH",
        "DTp",
        "KTp",
        "lambdaK",
        "lambdaD",
        "schmidt_number",
    )

    @cached_property
    def __calculate_m_l(self):
        return trtc.For(
            param_names=("ml", "signed_water_mass", "multiplicity", "cell_id"),
            name_iter="i",
            body="""
            atomicAdd((real_type*) &ml[cell_id[i]], multiplicity[i] * signed_water_mass[i]);
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    def __init__(self):
        ThrustRTCBackendMethods.__init__(self)
        self.RH_rtol = None
        self.adaptive = None
        self.max_iters = None
        self.ml_old: Optional[StorageBase] = None
        self.ml_new: Optional[StorageBase] = None
        self.T: Optional[StorageBase] = None
        self.dthd_dt_pred: Optional[StorageBase] = None
        self.d_water_vapour_mixing_ratio__dt_predicted: Optional[StorageBase] = None
        self.drhod_dt_pred: Optional[StorageBase] = None
        self.rhod_copy: Optional[StorageBase] = None
        self.m_d: Optional[StorageBase] = None
        self.vars: Optional[Dict[str, StorageBase]] = None
        self.vars_data: Optional[Dict] = None

    @cached_property
    def __update_drop_masses(self):
        phys = self.formulae
        const = phys.constants
        return trtc.For(
            param_names=(
                "signed_water_mass",
                "vdry",
                *CondensationMethods.keys,
                "_kappa",
                "_f_org",
                "dt",
                "RH_rtol",
                "rtol_x",
                "max_iters",
                "cell_id",
                "reynolds_number",
            ),
            name_iter="i",
            body=f"""
            struct Minfun {{
                static __device__ real_type value(real_type x_new, void* minfun_args_p) {{
                    auto minfun_args = static_cast<real_type*>(minfun_args_p);
                    if (x_new > {phys.diffusion_coordinate.x_max.c_inline()}) {{
                        return {arg("x_old")} - x_new;
                    }}
                    auto m_new = {phys.diffusion_coordinate.mass.c_inline(x="x_new")};
                    auto v_new = {phys.particle_shape_and_density.mass_to_volume.c_inline(mass="m_new")};
                    auto r_new = {phys.trivia.radius.c_inline(volume="v_new")};
                    auto sgm = {phys.surface_tension.sigma.c_inline(
                        T=arg('_T'),
                        v_wet="v_new",
                        v_dry=f"const.PI_4_3 * {arg('rd3')}",
                        f_org=arg("f_org")
                    )};
                    auto RH_eq = {phys.hygroscopicity.RH_eq.c_inline(
                        r="r_new",
                        T=arg('_T'),
                        kp=arg("kappa"),
                        rd3=arg("rd3"),
                        sgm="sgm"
                    )};
                    auto r_dr_dt = {phys.drop_growth.r_dr_dt.c_inline(
                        RH_eq="RH_eq",
                        RH=arg("_RH"),
                        Fk=arg("Fk"),
                        Fd=arg("Fd"),
                    )};
                    auto dm_dt = {phys.particle_shape_and_density.dm_dt.c_inline(
                        r="r_new", r_dr_dt="r_dr_dt"
                    )};
                    return {arg("x_old")} - x_new + {arg("dt")} * {
                        phys.diffusion_coordinate.dx_dt.c_inline(m="m_new", dm_dt="dm_dt")
                    };
                }}
            }};
            {BISECTION}

            auto _T = T[cell_id[i]];
            auto _pv = pv[cell_id[i]];
            auto _lv = lv[cell_id[i]];
            auto _pvs = pvs[cell_id[i]];
            auto _RH = RH[cell_id[i]];
            auto _DTp = DTp[cell_id[i]];
            auto _KTp = KTp[cell_id[i]];
            auto _lambdaK = lambdaK[cell_id[i]];
            auto _lambdaD = lambdaD[cell_id[i]];
            auto _schmidt_number = schmidt_number[cell_id[i]];

            auto v_old = {phys.particle_shape_and_density.mass_to_volume.c_inline(
                mass="signed_water_mass[i]"
            )};
            auto x_old = {phys.diffusion_coordinate.x.c_inline(mass="signed_water_mass[i]")};
            auto r_old = {phys.trivia.radius.c_inline(volume="v_old")};
            auto m_insane = {phys.particle_shape_and_density.volume_to_mass.c_inline(volume="vdry[i] / 100")};
            auto x_insane = {phys.diffusion_coordinate.x.c_inline(mass="m_insane")};
            auto rd3 = vdry[i] / {const.PI_4_3};
            auto sgm = {phys.surface_tension.sigma.c_inline(
                T="_T", v_wet="v", v_dry="vdry[i]", f_org="_f_org[i]"
            )};
            auto RH_eq = {phys.hygroscopicity.RH_eq.c_inline(
                r="r_old", T="_T", kp="_kappa[i]", rd3="rd3", sgm="sgm"
            )};

            real_type Fk=0;
            real_type Fd=0;
            real_type r_dr_dt_old=0;
            real_type dm_dt_old=0;
            real_type dx_old=0;

            real_type x_new = 0;
            if ( ! {phys.trivia.within_tolerance.c_inline(
                return_type='bool', error_estimate="abs(_RH - RH_eq)", value="_RH", rtol="RH_rtol"
            )}) {{
                auto Dr = {phys.diffusion_kinetics.D.c_inline(D="_DTp", r="r_old", lmbd="_lambdaD")};
                auto Kr = {phys.diffusion_kinetics.K.c_inline(K="_KTp", r="r_old", lmbd="_lambdaK")};
                auto qrt_re_times_cbrt_sc={phys.trivia.sqrt_re_times_cbrt_sc.c_inline(
                    Re="reynolds_number[i]",
                    Sc="_schmidt_number",
                )};
                auto mass_ventilation_factor = {phys.ventilation.ventilation_coefficient.c_inline(
                    sqrt_re_times_cbrt_sc="qrt_re_times_cbrt_sc"
                )};
                auto heat_ventilation_factor = mass_ventilation_factor; // TODO #1588
                Fk = {phys.drop_growth.Fk.c_inline(
                    T="_T",
                    K="heat_ventilation_factor*Kr",
                    lv="_lv",
                )}; // TODO #1588
                Fd = {phys.drop_growth.Fd.c_inline(
                    T="_T",
                    D="mass_ventilation_factor*Dr",
                    pvs="_pvs",
                )};
                r_dr_dt_old = {phys.drop_growth.r_dr_dt.c_inline(
                    RH_eq="RH_eq",
                    RH="_RH",
                    Fk="Fk",
                    Fd="Fd",
                )};
                dm_dt_old = {phys.particle_shape_and_density.dm_dt.c_inline(r="r_old", r_dr_dt="r_dr_dt_old")};
                dx_old = dt * {phys.diffusion_coordinate.dx_dt.c_inline(
                    m="signed_water_mass[i]", dm_dt="dm_dt_old"
                )};
            }}
            else {{
                dx_old = 0;
            }}
            real_type kappa = _kappa[i];
            real_type f_org = _f_org[i];
            real_type minfun_args[] = {{{','.join(MINFUN_ARGS_VARS)}}}; // array

            if (dx_old == 0) {{
                x_new = x_old;
            }}
            else {{
                auto a = x_old;
                auto b = max(x_insane, a + dx_old);
                auto fa = Minfun::value(a, &minfun_args);
                auto fb = Minfun::value(b, &minfun_args);

                auto counter = 0;
                while ( ! fa * fb < 0) {{
                    counter += 1;
                    if (counter > max_iters) {{
                        printf("failed to find interval");
                        //success = False
                        return;
                    }}
                    b = max(x_insane, a + ldexp(dx_old, 1.*counter));
                    fb = Minfun::value(b, &minfun_args);
                }}

                //if not success:
                //    x_new = np.nan
                //    break
                if (a != b) {{
                    if (a > b) {{
                        auto tmp = a;
                        a = b;
                        b = tmp;
                        auto ftmp = fa;
                        fa = fb;
                        fb = ftmp;
                    }}
                    x_new = Bisect::bisect(Minfun::value, &minfun_args, a, b, fa, fb, rtol_x);
                    //if iters_taken in (-1, max_iters):
                    //    if not fake:
                    //        print("TOMS failed")
                    //    success = False
                    //    break
                }}
                else {{
                    x_new = x_old;
                }}
            }}
            signed_water_mass[i] = {phys.diffusion_coordinate.mass.c_inline(x="x_new")};
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __pre_for(self):
        return trtc.For(
            param_names=(
                "dthd_dt_pred",
                "d_water_vapour_mixing_ratio__dt_predicted",
                "drhod_dt_pred",
                "m_d",
                "pthd",
                "thd",
                "predicted_water_vapour_mixing_ratio",
                "water_vapour_mixing_ratio",
                "prhod",
                "rhod",
                "dt",
                "RH_max",
                "dv",
            ),
            name_iter="i",
            body="""
            dthd_dt_pred[i] = (pthd[i] - thd[i]) / dt;
            d_water_vapour_mixing_ratio__dt_predicted[i] = (
                predicted_water_vapour_mixing_ratio[i] - water_vapour_mixing_ratio[i]
            ) / dt;
            drhod_dt_pred[i] = (prhod[i] - rhod[i]) / dt;

            m_d[i] = (prhod[i] + rhod[i]) / 2 * dv;

            pthd[i] = thd[i];
            predicted_water_vapour_mixing_ratio[i] = water_vapour_mixing_ratio[i];

            RH_max[i] = 0;
        """,
        )

    @cached_property
    def __pre(self):
        phys = self.formulae
        return trtc.For(
            param_names=(
                *CondensationMethods.keys,
                "dthd_dt_pred",
                "d_water_vapour_mixing_ratio__dt_predicted",
                "drhod_dt_pred",
                "pthd",
                "predicted_water_vapour_mixing_ratio",
                "rhod_copy",
                "dt",
                "RH_max",
                "air_density",
                "air_dynamic_viscosity",
            ),
            name_iter="i",
            body=f"""
            pthd[i] += dt * dthd_dt_pred[i] / 2;
            predicted_water_vapour_mixing_ratio[i] += (
                dt * d_water_vapour_mixing_ratio__dt_predicted[i] / 2
            );
            rhod_copy[i] += dt * drhod_dt_pred[i] / 2;

            T[i] = {phys.state_variable_triplet.T.c_inline(
                rhod='rhod_copy[i]', thd='pthd[i]')};
            p[i] = {phys.state_variable_triplet.p.c_inline(
                rhod='rhod_copy[i]', T='T[i]',
                water_vapour_mixing_ratio='predicted_water_vapour_mixing_ratio[i]'
            )};
            pv[i] = {phys.state_variable_triplet.pv.c_inline(
                p='p[i]', water_vapour_mixing_ratio='predicted_water_vapour_mixing_ratio[i]')};
            lv[i] = {phys.latent_heat_vapourisation.lv.c_inline(
                T='T[i]')};
            pvs[i] = {phys.saturation_vapour_pressure.pvs_water.c_inline(
                T='T[i]')};
            RH[i] = pv[i] / pvs[i];
            RH_max[i] = max(RH_max[i], RH[i]);
            DTp[i] = {phys.diffusion_thermics.D.c_inline(
                T='T[i]', p='p[i]')};
            KTp[i] = {phys.diffusion_thermics.K.c_inline(
                T='T[i]', p='p[i]'
            )};
            lambdaK[i] = {phys.diffusion_kinetics.lambdaK.c_inline(
                T='T[i]', p='p[i]')};
            lambdaD[i] = {phys.diffusion_kinetics.lambdaD.c_inline(
                D='DTp[i]', T='T[i]')};
            schmidt_number[i] = {phys.trivia.air_schmidt_number.c_inline(
                dynamic_viscosity="air_dynamic_viscosity[i]",
                diffusivity="DTp[i]",
                density="air_density[i]",
            )};
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __post(self):
        phys = self.formulae
        return trtc.For(
            param_names=(
                "dthd_dt_pred",
                "d_water_vapour_mixing_ratio__dt_predicted",
                "drhod_dt_pred",
                "pthd",
                "predicted_water_vapour_mixing_ratio",
                "rhod_copy",
                "dt",
                "ml_new",
                "ml_old",
                "m_d",
                "T",
                "lv",
            ),
            name_iter="i",
            body=f"""
            auto dml_dt = (ml_new[i] - ml_old[i]) / dt;
            auto d_water_vapour_mixing_ratio__dt_corrected = - dml_dt / m_d[i];
            auto dthd_dt_corr = {phys.state_variable_triplet.dthd_dt.c_inline(
                rhod='rhod_copy[i]', thd='pthd[i]', T='T[i]',
                d_water_vapour_mixing_ratio__dt='d_water_vapour_mixing_ratio__dt_corrected',
                lv='lv[i]'
            )};
            pthd[i] += dt * (dthd_dt_pred[i] / 2 + dthd_dt_corr);
            predicted_water_vapour_mixing_ratio[i] += dt * (
                d_water_vapour_mixing_ratio__dt_predicted[i] / 2 +
                d_water_vapour_mixing_ratio__dt_corrected
            );
            rhod_copy[i] += dt * drhod_dt_pred[i] / 2;
            ml_old[i] = ml_new[i];
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    def calculate_m_l(self, ml, signed_water_mass, multiplicity, cell_id):
        ml[:] = 0
        self.__calculate_m_l.launch_n(
            n=len(multiplicity),
            args=(ml.data, signed_water_mass.data, multiplicity.data, cell_id.data),
        )

    # pylint: disable=unused-argument,too-many-locals
    @nice_thrust(**NICE_THRUST_FLAGS)
    def condensation(
        self,
        *,
        solver,
        n_cell,
        cell_start_arg,
        signed_water_mass,
        v_cr,
        multiplicity,
        vdry,
        idx,
        rhod,
        thd,
        water_vapour_mixing_ratio,
        dv,
        prhod,
        pthd,
        predicted_water_vapour_mixing_ratio,
        kappa,
        f_org,
        rtol_x,
        rtol_thd,
        timestep,
        counters,
        cell_order,
        RH_max,
        success,
        cell_id,
        reynolds_number,
        air_density,
        air_dynamic_viscosity,
    ):
        assert solver is None

        if self.adaptive:
            counters["n_substeps"][:] = 1  # TODO #527

        n_substeps = counters["n_substeps"][0]

        success[:] = True  # TODO #588
        dvfloat = self._get_floating_point
        self.rhod_copy.fill(rhod)

        self.__pre_for.launch_n(
            n=n_cell,
            args=(
                self.dthd_dt_pred.data,
                self.d_water_vapour_mixing_ratio__dt_predicted.data,
                self.drhod_dt_pred.data,
                self.m_d.data,
                pthd.data,
                thd.data,
                predicted_water_vapour_mixing_ratio.data,
                water_vapour_mixing_ratio.data,
                prhod.data,
                self.rhod_copy.data,
                dvfloat(timestep),
                RH_max.data,
                dvfloat(dv),
            ),
        )
        timestep /= n_substeps
        self.calculate_m_l(self.ml_old, signed_water_mass, multiplicity, cell_id)

        for _ in range(n_substeps):
            self.__pre.launch_n(
                n=n_cell,
                args=(
                    *self.vars_data.values(),
                    self.dthd_dt_pred.data,
                    self.d_water_vapour_mixing_ratio__dt_predicted.data,
                    self.drhod_dt_pred.data,
                    pthd.data,
                    predicted_water_vapour_mixing_ratio.data,
                    self.rhod_copy.data,
                    dvfloat(timestep),
                    RH_max.data,
                    air_density.data,
                    air_dynamic_viscosity.data,
                ),
            )
            self.__update_drop_masses.launch_n(
                n=len(multiplicity),
                args=(
                    signed_water_mass.data,
                    vdry.data,
                    *self.vars_data.values(),
                    kappa.data,
                    f_org.data,
                    dvfloat(timestep),
                    dvfloat(self.RH_rtol),
                    dvfloat(rtol_x),
                    dvfloat(self.max_iters),
                    cell_id.data,
                    reynolds_number.data,
                ),
            )
            self.calculate_m_l(self.ml_new, signed_water_mass, multiplicity, cell_id)
            self.__post.launch_n(
                n=n_cell,
                args=(
                    self.dthd_dt_pred.data,
                    self.d_water_vapour_mixing_ratio__dt_predicted.data,
                    self.drhod_dt_pred.data,
                    pthd.data,
                    predicted_water_vapour_mixing_ratio.data,
                    self.rhod_copy.data,
                    dvfloat(timestep),
                    self.ml_new.data,
                    self.ml_old.data,
                    self.m_d.data,
                    self.vars["T"].data,
                    self.vars["lv"].data,
                ),
            )

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
        self.adaptive = adaptive
        self.RH_rtol = RH_rtol
        self.max_iters = max_iters
        for attr in (
            "ml_old",
            "ml_new",
            "T",
            "dthd_dt_pred",
            "d_water_vapour_mixing_ratio__dt_predicted",
            "drhod_dt_pred",
            "m_d",
            "rhod_copy",
        ):
            setattr(
                self, attr, self.Storage.empty(shape=n_cell, dtype=self._get_np_dtype())
            )
        self.vars = {
            key: self.Storage.empty(shape=n_cell, dtype=self._get_np_dtype())
            for key in CondensationMethods.keys
        }
        self.vars_data = {key: val.data for key, val in self.vars.items()}
