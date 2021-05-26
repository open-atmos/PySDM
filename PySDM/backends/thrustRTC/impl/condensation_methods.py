from typing import Dict, Optional

from PySDM.physics import constants as const
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl import nice_thrust, c_inline
from .precision_resolver import PrecisionResolver
from PySDM.backends.thrustRTC.bisection import BISECTION
from ..conf import trtc
from PySDM.backends.thrustRTC.storage import Storage


class CondensationMethods():
    keys = ['T', 'p', 'pv', 'lv', 'pvs', 'RH', 'DTp', 'lambdaK', 'lambdaD']

    def __init__(self):
        phys = self.formulae
        self.RH_rtol = None
        self.adaptive = None
        self.max_iters = None
        self.ml_old: Optional[Storage] = None
        self.ml_new: Optional[Storage] = None
        self.T: Optional[Storage] = None
        self.dthd_dt_pred: Optional[Storage] = None
        self.dqv_dt_pred: Optional[Storage] = None
        self.rhod_mean: Optional[Storage] = None
        self.vars: Optional[Dict[str, Storage]] = None

        # TODO #526: precision for consts
        self.__calculate_m_l = trtc.For(("ml", "v", "n", "cell_id"), "i", f'''
            atomicAdd((real_type*) &ml[cell_id[i]], n[i] * v[i] * {const.rho_w}); 
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        args_vars = ('x_old', 'dt', 'kappa', 'rd3', '_T', '_RH', '_lv', '_pvs', 'Dr', 'Kr')

        def args(arg):
            return f"args[{args_vars.index(arg)}]"

        self.__update_volume = trtc.For(("v", "vdry", *CondensationMethods.keys, "kappa", "dt", "RH_rtol", "rtol_x", "max_iters", "cell_id"), "i",
            f'''            
            struct Minfun {{
                static __device__ real_type value(real_type x_new, void* args_p) {{
                    auto args = static_cast<real_type*>(args_p);
                    auto vol = {c_inline(phys.condensation_coordinate.volume, x="x_new")};
                    auto r_new = {c_inline(phys.trivia.radius, volume="vol")};
                    auto sgm = {c_inline(phys.surface_tension.sigma, T=args('_T'), v_wet="vol", v_dry=f"const.pi_4_3 * {args('rd3')}")};
                    auto RH_eq = {c_inline(phys.hygroscopicity.RH_eq, r="r_new", T=args('_T'), kp=args("kappa"), rd3=args("rd3"), sgm="sgm")};
                    auto r_dr_dt = {c_inline(phys.drop_growth.r_dr_dt, RH_eq="RH_eq", T=args("_T"), RH=args("_RH"), lv=args("_lv"), pvs=args("_pvs"), D=args("Dr"), K=args("Kr"))};
                    return {args("x_old")} - x_new + {args("dt")} * {c_inline(phys.condensation_coordinate.dx_dt, x="x_new", r_dr_dt="r_dr_dt")};
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
            
            auto x_old = {c_inline(phys.condensation_coordinate.x, volume="v[i]")};
            auto r_old = {c_inline(phys.trivia.radius, volume="v[i]")};
            auto x_insane = {c_inline(phys.condensation_coordinate.x, volume="vdry[i]/100")};
            auto rd3 = vdry[i] / {const.pi_4_3};
            auto sgm = {c_inline(phys.surface_tension.sigma, T="_T", v_wet="v[i]", v_dry="vdry[i]")};
            auto RH_eq = {c_inline(phys.hygroscopicity.RH_eq, r="r_old", T="_T", kp="kappa", rd3="rd3", sgm="sgm")};

            real_type Dr=0;
            real_type Kr=0; 
            real_type r_dr_dt_old=0;
            real_type dx_old=0;

            real_type x_new = 0;
            if ( ! {c_inline(phys.trivia.within_tolerance, return_type='bool', error_estimate="abs(_RH - RH_eq)", value="_RH", rtol="RH_rtol")}) {{
                Dr = {c_inline(phys.diffusion_kinetics.DK, DK="_DTp", r="r_old", lmbd="_lambdaD")};
                Kr = {c_inline(phys.diffusion_kinetics.DK, DK="const.K0", r="r_old", lmbd="_lambdaK")};
                r_dr_dt_old = {c_inline(phys.drop_growth.r_dr_dt, RH_eq="RH_eq", T="_T", RH="_RH", lv="_lv", pvs="_pvs", D="Dr", K="Kr")};
                dx_old = dt * {c_inline(phys.condensation_coordinate.dx_dt, x="x_old", r_dr_dt="r_dr_dt_old")};
            }}
            else {{
                dx_old = 0;
            }}
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
            v[i] = {c_inline(phys.condensation_coordinate.volume, x="x_new")};
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__pre_for = trtc.For(("dthd_dt_pred", "dqv_dt_pred", "rhod_mean", "pthd", "thd", "pqv", "qv", "prhod", "rhod", "dt", "RH_max"),
                              "i", f'''
            dthd_dt_pred[i] = (pthd[i] - thd[i]) / dt;
            dqv_dt_pred[i] = (pqv[i] - qv[i]) / dt;
            rhod_mean[i] = (prhod[i] + rhod[i]) / 2;
            RH_max[i] = 0;
        ''')

        self.__pre = trtc.For(
            (*CondensationMethods.keys, "dthd_dt_pred", "dqv_dt_pred", "rhod_mean", "thd", "qv", "rhod", "dt", "RH_max"),
            "i", f'''
            thd[i] += dt * dthd_dt_pred[i] / 2;
            qv[i] += dt * dqv_dt_pred[i] / 2;

            T[i] = {c_inline(phys.state_variable_triplet.T, rhod='rhod_mean[i]', thd='thd[i]')};
            p[i] = {c_inline(phys.state_variable_triplet.p, rhod='rhod_mean[i]', T='T[i]', qv='qv[i]')};
            pv[i] = {c_inline(phys.state_variable_triplet.pv, p='p[i]', qv='qv[i]')};
            lv[i] = {c_inline(phys.latent_heat.lv, T='T[i]')};
            pvs[i] = {c_inline(phys.saturation_vapour_pressure.pvs_Celsius, T='T[i] - const.T0')};
            RH[i] = pv[i] / pvs[i];
            RH_max[i] = max(RH_max[i], RH[i]);
            DTp[i] = {c_inline(phys.diffusion_thermics.D, T='T[i]', p='p[i]')};
            lambdaK[i] = {c_inline(phys.diffusion_kinetics.lambdaK, T='T[i]', p='p[i]')};
            lambdaD[i] = {c_inline(phys.diffusion_kinetics.lambdaD, D='DTp[i]', T='T[i]')};
        ''')

        self.__post = trtc.For(
            ("dthd_dt_pred", "dqv_dt_pred", "rhod_mean", "thd", "qv", "rhod", "dt", "ml_new", "ml_old", "dv_mean", "T",
             "lv"),
            "i", f'''
            auto dml_dt = (ml_new[i] - ml_old[i]) / dt;
            auto dqv_dt_corr = - dml_dt / (rhod_mean[i] * dv_mean);
            auto dthd_dt_corr = {c_inline(phys.state_variable_triplet.dthd_dt, rhod='rhod_mean[i]', thd='thd[i]', T='T[i]', dqv_dt='dqv_dt_corr', lv='lv[i]')};

            thd[i] += dt * (dthd_dt_pred[i] / 2 + dthd_dt_corr);
            qv[i] += dt * (dqv_dt_pred[i] / 2 + dqv_dt_corr);
            ml_old[i] = ml_new[i];
        ''')

    def calculate_m_l(self, ml, v, n, cell_id):
        ml[:] = 0
        self.__calculate_m_l.launch_n(len(n), (ml.data, v.data, n.data, cell_id.data))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def condensation(
        self,
        solver,
        n_cell, cell_start_arg,
        v, v_cr, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
        rtol_x, rtol_thd, dt, counters, cell_order, RH_max, success, cell_id
    ):
        assert solver is None

        if self.adaptive:
            counters['n_substeps'][:] = 1  # TODO #527

        n_substeps = counters['n_substeps'][0]
        dv_mean = dv

        success[:] = True  # TODO #528
        dvfloat = PrecisionResolver.get_floating_point

        self.__pre_for.launch_n(n_cell, (self.dthd_dt_pred.data, self.dqv_dt_pred.data, self.rhod_mean.data, pthd.data, thd.data, pqv.data, qv.data, prhod.data, rhod.data, dvfloat(dt), RH_max.data))

        dt /= n_substeps
        self.calculate_m_l(self.ml_old, v, n, cell_id)

        for _ in range(n_substeps):
            self.__pre.launch_n(n_cell, (*self.vars.values(),  self.dthd_dt_pred.data, self.dqv_dt_pred.data, self.rhod_mean.data, pthd.data, pqv.data, rhod.data, dvfloat(dt), RH_max.data))
            self.__update_volume.launch_n(len(n), (v.data, vdry.data, *self.vars.values(),
                                                   dvfloat(kappa),
                                                   dvfloat(dt), dvfloat(self.RH_rtol), dvfloat(rtol_x),
                                                   dvfloat(self.max_iters), cell_id.data)
                                          )
            self.calculate_m_l(self.ml_new, v, n, cell_id)
            self.__post.launch_n(n_cell, (self.dthd_dt_pred.data, self.dqv_dt_pred.data, self.rhod_mean.data, pthd.data, pqv.data, rhod.data, dvfloat(dt), self.ml_new.data, self.ml_old.data, dvfloat(dv_mean), self.vars['T'], self.vars['lv']))

    def make_condensation_solver(self, dt, n_cell, *, dt_range, adaptive, fuse, multiplier, RH_rtol, max_iters):
        self.adaptive = adaptive
        self.RH_rtol = RH_rtol
        self.max_iters = max_iters
        self.ml_old = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.ml_new = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.T = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.dthd_dt_pred = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.dqv_dt_pred = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.rhod_mean = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.vars = {key: Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype()).data
                     for key in CondensationMethods.keys}
        return None
