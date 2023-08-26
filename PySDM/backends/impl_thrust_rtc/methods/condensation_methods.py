"""
GPU implementation of backend methods for water condensation/evaporation
"""
from typing import Dict, Optional

from PySDM.backends.impl_common.storage_utils import StorageBase
from PySDM.backends.impl_thrust_rtc.bisection import BISECTION
from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class CondensationMethods(
    ThrustRTCBackendMethods
):  # pylint: disable=too-many-instance-attributes
    keys = ("T", "p", "pv", "lv", "pvs", "RH", "DTp", "lambdaK", "lambdaD")

    def __init__(self):
        ThrustRTCBackendMethods.__init__(self)
        phys = self.formulae
        self.RH_rtol = None
        self.adaptive = None
        self.max_iters = None
        self.ml_old: Optional[StorageBase] = None
        self.ml_new: Optional[StorageBase] = None
        self.T: Optional[StorageBase] = None
        self.dthd_dt_pred: Optional[StorageBase] = None
        self.dqv_dt_pred: Optional[StorageBase] = None
        self.drhod_dt_pred: Optional[StorageBase] = None
        self.rhod_copy: Optional[StorageBase] = None
        self.m_d: Optional[StorageBase] = None
        self.vars: Optional[Dict[str, StorageBase]] = None
        self.vars_data: Optional[Dict] = None
        const = self.formulae.constants

        self.__calculate_m_l = trtc.For(
            ("ml", "v", "multiplicity", "cell_id"),
            "i",
            f"""
            atomicAdd((real_type*) &ml[cell_id[i]], multiplicity[i] * v[i] * {const.rho_w});
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

        args_vars = (
            "x_old",
            "dt",
            "kappa",
            "f_org",
            "rd3",
            "_T",
            "_RH",
            "_lv",
            "_pvs",
            "Dr",
            "Kr",
        )

        def args(arg):
            return f"args[{args_vars.index(arg)}]"

        self.__update_volume = trtc.For(
            (
                "v",
                "vdry",
                *CondensationMethods.keys,
                "_kappa",
                "_f_org",
                "dt",
                "RH_rtol",
                "rtol_x",
                "max_iters",
                "cell_id",
            ),
            "i",
            f"""
            struct Minfun {{
                static __device__ real_type value(real_type x_new, void* args_p) {{
                    auto args = static_cast<real_type*>(args_p);
                    auto vol = {phys.condensation_coordinate.volume.c_inline(x="x_new")};
                    auto r_new = {phys.trivia.radius.c_inline(volume="vol")};
                    auto sgm = {phys.surface_tension.sigma.c_inline(
                        T=args('_T'),
                        v_wet="vol",
                        v_dry=f"const.PI_4_3 * {args('rd3')}",
                        f_org=args("f_org")
                    )};
                    auto RH_eq = {phys.hygroscopicity.RH_eq.c_inline(
                        r="r_new",
                        T=args('_T'),
                        kp=args("kappa"),
                        rd3=args("rd3"),
                        sgm="sgm"
                    )};
                    auto r_dr_dt = {phys.drop_growth.r_dr_dt.c_inline(
                        RH_eq="RH_eq",
                        T=args("_T"),
                        RH=args("_RH"),
                        lv=args("_lv"),
                        pvs=args("_pvs"),
                        D=args("Dr"),
                        K=args("Kr")
                    )};
                    return {args("x_old")} - x_new + {args("dt")} * {
                        phys.condensation_coordinate.dx_dt.c_inline(x="x_new", r_dr_dt="r_dr_dt")
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
            auto _lambdaK = lambdaK[cell_id[i]];
            auto _lambdaD = lambdaD[cell_id[i]];

            auto x_old = {phys.condensation_coordinate.x.c_inline(volume="v[i]")};
            auto r_old = {phys.trivia.radius.c_inline(volume="v[i]")};
            auto x_insane = {phys.condensation_coordinate.x.c_inline(volume="vdry[i]/100")};
            auto rd3 = vdry[i] / {const.PI_4_3};
            auto sgm = {phys.surface_tension.sigma.c_inline(
                T="_T", v_wet="v[i]", v_dry="vdry[i]", f_org="_f_org[i]"
            )};
            auto RH_eq = {phys.hygroscopicity.RH_eq.c_inline(
                r="r_old", T="_T", kp="_kappa[i]", rd3="rd3", sgm="sgm"
            )};

            real_type Dr=0;
            real_type Kr=0;
            real_type r_dr_dt_old=0;
            real_type dx_old=0;

            real_type x_new = 0;
            if ( ! {phys.trivia.within_tolerance.c_inline(
                return_type='bool', error_estimate="abs(_RH - RH_eq)", value="_RH", rtol="RH_rtol"
            )}) {{
                Dr = {phys.diffusion_kinetics.D.c_inline(
                    D="_DTp", r="r_old", lmbd="_lambdaD"
                )};
                Kr = {phys.diffusion_kinetics.K.c_inline(
                    K="const.K0", r="r_old", lmbd="_lambdaK"
                )};
                r_dr_dt_old = {phys.drop_growth.r_dr_dt.c_inline(
                    RH_eq="RH_eq", T="_T", RH="_RH", lv="_lv", pvs="_pvs", D="Dr", K="Kr"
                )};
                dx_old = dt * {phys.condensation_coordinate.dx_dt.c_inline(
                    x="x_old", r_dr_dt="r_dr_dt_old"
                )};
            }}
            else {{
                dx_old = 0;
            }}
            real_type kappa = _kappa[i];
            real_type f_org = _f_org[i];
            real_type args[] = {{{','.join(args_vars)}}}; // array

            if (dx_old == 0) {{
                x_new = x_old;
            }}
            else {{
                auto a = x_old;
                auto b = max(x_insane, a + dx_old);
                auto fa = Minfun::value(a, &args);
                auto fb = Minfun::value(b, &args);

                auto counter = 0;
                while ( ! fa * fb < 0) {{
                    counter += 1;
                    if (counter > max_iters) {{
                        printf("failed to find interval");
                        //success = False
                        return;
                    }}
                    b = max(x_insane, a + ldexp(dx_old, 1.*counter));
                    fb = Minfun::value(b, &args);
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
                    x_new = Bisect::bisect(Minfun::value, &args, a, b, fa, fb, rtol_x);
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
            v[i] = {phys.condensation_coordinate.volume.c_inline(x="x_new")};
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

        self.__pre_for = trtc.For(
            (
                "dthd_dt_pred",
                "dqv_dt_pred",
                "drhod_dt_pred",
                "m_d",
                "pthd",
                "thd",
                "pqv",
                "qv",
                "prhod",
                "rhod",
                "dt",
                "RH_max",
                "dv",
            ),
            "i",
            """
            dthd_dt_pred[i] = (pthd[i] - thd[i]) / dt;
            dqv_dt_pred[i] = (pqv[i] - qv[i]) / dt;
            drhod_dt_pred[i] = (prhod[i] - rhod[i]) / dt;

            m_d[i] = (prhod[i] + rhod[i]) / 2 * dv;

            pthd[i] = thd[i];
            pqv[i] = qv[i];

            RH_max[i] = 0;
        """,
        )

        self.__pre = trtc.For(
            (
                *CondensationMethods.keys,
                "dthd_dt_pred",
                "dqv_dt_pred",
                "drhod_dt_pred",
                "pthd",
                "pqv",
                "rhod_copy",
                "dt",
                "RH_max",
            ),
            "i",
            f"""
            pthd[i] += dt * dthd_dt_pred[i] / 2;
            pqv[i] += dt * dqv_dt_pred[i] / 2;
            rhod_copy[i] += dt * drhod_dt_pred[i] / 2;

            T[i] = {phys.state_variable_triplet.T.c_inline(
                rhod='rhod_copy[i]', thd='pthd[i]')};
            p[i] = {phys.state_variable_triplet.p.c_inline(
                rhod='rhod_copy[i]', T='T[i]', qv='pqv[i]')};
            pv[i] = {phys.state_variable_triplet.pv.c_inline(
                p='p[i]', qv='pqv[i]')};
            lv[i] = {phys.latent_heat.lv.c_inline(
                T='T[i]')};
            pvs[i] = {phys.saturation_vapour_pressure.pvs_Celsius.c_inline(
                T='T[i] - const.T0')};
            RH[i] = pv[i] / pvs[i];
            RH_max[i] = max(RH_max[i], RH[i]);
            DTp[i] = {phys.diffusion_thermics.D.c_inline(
                T='T[i]', p='p[i]')};
            lambdaK[i] = {phys.diffusion_kinetics.lambdaK.c_inline(
                T='T[i]', p='p[i]')};
            lambdaD[i] = {phys.diffusion_kinetics.lambdaD.c_inline(
                D='DTp[i]', T='T[i]')};
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

        self.__post = trtc.For(
            (
                "dthd_dt_pred",
                "dqv_dt_pred",
                "drhod_dt_pred",
                "pthd",
                "pqv",
                "rhod_copy",
                "dt",
                "ml_new",
                "ml_old",
                "m_d",
                "T",
                "lv",
            ),
            "i",
            f"""
            auto dml_dt = (ml_new[i] - ml_old[i]) / dt;
            auto dqv_dt_corr = - dml_dt / m_d[i];
            auto dthd_dt_corr = {phys.state_variable_triplet.dthd_dt.c_inline(
                rhod='rhod_copy[i]', thd='pthd[i]', T='T[i]', dqv_dt='dqv_dt_corr', lv='lv[i]')};
            pthd[i] += dt * (dthd_dt_pred[i] / 2 + dthd_dt_corr);
            pqv[i] += dt * (dqv_dt_pred[i] / 2 + dqv_dt_corr);
            rhod_copy[i] += dt * drhod_dt_pred[i] / 2;
            ml_old[i] = ml_new[i];
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    def calculate_m_l(self, ml, v, multiplicity, cell_id):
        ml[:] = 0
        self.__calculate_m_l.launch_n(
            len(multiplicity), (ml.data, v.data, multiplicity.data, cell_id.data)
        )

    # pylint: disable=unused-argument,too-many-locals
    @nice_thrust(**NICE_THRUST_FLAGS)
    def condensation(
        self,
        *,
        solver,
        n_cell,
        cell_start_arg,
        v,
        v_cr,
        multiplicity,
        vdry,
        idx,
        rhod,
        thd,
        qv,
        dv,
        prhod,
        pthd,
        pqv,
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
    ):
        assert solver is None

        if self.adaptive:
            counters["n_substeps"][:] = 1  # TODO #527

        n_substeps = counters["n_substeps"][0]

        success[:] = True  # TODO #588
        dvfloat = self._get_floating_point
        self.rhod_copy.fill(rhod)

        self.__pre_for.launch_n(
            n_cell,
            (
                self.dthd_dt_pred.data,
                self.dqv_dt_pred.data,
                self.drhod_dt_pred.data,
                self.m_d.data,
                pthd.data,
                thd.data,
                pqv.data,
                qv.data,
                prhod.data,
                self.rhod_copy.data,
                dvfloat(timestep),
                RH_max.data,
                dvfloat(dv),
            ),
        )
        timestep /= n_substeps
        self.calculate_m_l(self.ml_old, v, multiplicity, cell_id)

        for _ in range(n_substeps):
            self.__pre.launch_n(
                n_cell,
                (
                    *self.vars_data.values(),
                    self.dthd_dt_pred.data,
                    self.dqv_dt_pred.data,
                    self.drhod_dt_pred.data,
                    pthd.data,
                    pqv.data,
                    self.rhod_copy.data,
                    dvfloat(timestep),
                    RH_max.data,
                ),
            )
            self.__update_volume.launch_n(
                len(multiplicity),
                (
                    v.data,
                    vdry.data,
                    *self.vars_data.values(),
                    kappa.data,
                    f_org.data,
                    dvfloat(timestep),
                    dvfloat(self.RH_rtol),
                    dvfloat(rtol_x),
                    dvfloat(self.max_iters),
                    cell_id.data,
                ),
            )
            self.calculate_m_l(self.ml_new, v, multiplicity, cell_id)
            self.__post.launch_n(
                n_cell,
                (
                    self.dthd_dt_pred.data,
                    self.dqv_dt_pred.data,
                    self.drhod_dt_pred.data,
                    pthd.data,
                    pqv.data,
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
            "dqv_dt_pred",
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
