from PySDM.physics import constants as const
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl import nice_thrust, c_inline
from .precision_resolver import PrecisionResolver
from PySDM.backends.thrustRTC.bisection import BISECTION
from ..conf import trtc
from PySDM.backends.thrustRTC.storage import Storage


class CondensationMethods:
    def __init__(self):
        phys = self.formulae
        self.RH_rtol = None

        # TODO #509: precision for consts
        self.__calculate_m_l = trtc.For(("ml", "v", "n", "cell_id"), "i", f'''
            atomicAdd((real_type*) &ml[cell_id[i]], n[i] * v[i] * {const.rho_w}); 
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__update_volume = trtc.For(("v", "vdry", "T", "kappa", "DTp", "lambdaD", "lambdaK", "RH",
                                         "pvs", "lv", "dt", "p", "RH_rtol", "rtol_x", "max_iters"), "i",
            '''
            struct Args {
                real_type x_old, dt, p, kappa, rd3, T, RH, lv, pvs, Dr, Kr;
            };
            struct Minfun {
                static __device__ real_type value(const real_type &x_new, void* args_p)
                {
                '''
                f'''
                    auto args = static_cast<Args*>(args_p);
                    auto vol = {c_inline(phys.condensation_coordinate.volume, x="x_new")};
                    auto r_new = {c_inline(phys.trivia.radius, volume="vol")};
                    auto sgm = {c_inline(phys.surface_tension.sigma, T="args->T", v_wet="vol", v_dry="const.pi_4_3 * args->rd3")};
                    auto RH_eq = {c_inline(phys.hygroscopicity.RH_eq, r="r_new", T="args->T", kp="args->kappa", rd3="args->rd3", sgm="sgm")};
                    auto r_dr_dt = {c_inline(phys.drop_growth.r_dr_dt, RH_eq="RH_eq", T="args->T", RH="args->RH", lv="args->lv", pvs="args->pvs", D="args->Dr", K="args->Kr")};
                    return args->x_old - x_new + args->dt * {c_inline(phys.condensation_coordinate.dx_dt, x="x_new", r_dr_dt="r_dr_dt")};
                '''
                '''
                }
            };
            '''
            f'''
            {BISECTION}
            auto x_old = {c_inline(phys.condensation_coordinate.x, volume="v[i]")};
            auto r_old = {c_inline(phys.trivia.radius, volume="v[i]")};
            auto x_insane = {c_inline(phys.condensation_coordinate.x, volume="vdry[i]/100")};
            auto rd3 = vdry[i] / {const.pi_4_3};
            auto sgm = {c_inline(phys.surface_tension.sigma, T="T", v_wet="v[i]", v_dry="vdry[i]")};
            auto RH_eq = {c_inline(phys.hygroscopicity.RH_eq, r="r_old", T="T", kp="kappa", rd3="rd3", sgm="sgm")};

            real_type Dr, Kr, r_dr_dt_old, dx_old;
            Args args;
            real_type x_new;
            if ( ! {c_inline(phys.trivia.within_tolerance, error_estimate="abs(RH - RH_eq)", value="RH", rtol="RH_rtol")})
            {{
                Dr = {c_inline(phys.diffusion_kinetics.DK, DK="DTp", r="r_old", lmbd="lambdaD")};
                Kr = {c_inline(phys.diffusion_kinetics.DK, DK="const.K0", r="r_old", lmbd="lambdaK")};
                args = Args{{x_old, dt, p, kappa, rd3, T, RH, lv, pvs, Dr, Kr}};
                r_dr_dt_old = {c_inline(phys.drop_growth.r_dr_dt, RH_eq="RH_eq", T="T", RH="RH", lv="lv", pvs="pvs", D="Dr", K="Kr")};
                dx_old = dt * {c_inline(phys.condensation_coordinate.dx_dt, x="x_old", r_dr_dt="r_dr_dt_old")};
            }}
            else 
            {{
                dx_old = 0;
            }}
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
                    b = max(x_insane, a + ldexp(dx_old, counter));
                    fb = Minfun::value(b, &args);
                }}

                //if not success:
                //    x_new = np.nan
                //    break
                if (a != b) {{
                    if (a > b) {{
                        {{ 
                            auto tmp = a; 
                            a = b;
                            b = tmp;
                        }}
                        {{ 
                            auto tmp = fa; 
                            fa = fb;
                            fb = tmp;
                        }}
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
        phys = self.formulae

        # TODO #509: not here
        self.ml_old = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())
        self.ml_new = Storage.empty(shape=n_cell, dtype=PrecisionResolver.get_np_dtype())

        if n_cell != 1:
            raise NotImplementedError()
        cid = 0
        n_substeps = counters['n_substeps'][cell_id][0]
        dv_mean = dv
        dthd_dt_pred = (pthd[cid] - thd[cid]) / dt
        dqv_dt_pred = (pqv[cid] - qv[cid]) / dt
        rhod_mean = (prhod[cid] + rhod[cid]) / 2
        m_d = rhod_mean * dv_mean
        success[cid] = True
        # </TODO>

        dt /= n_substeps
        self.calculate_m_l(self.ml_old, v, n, cell_id)

        for i in range(n_substeps):
            thd[cid] += dt * dthd_dt_pred / 2  # TODO #48 example showing that it makes sense
            qv[cid] += dt * dqv_dt_pred / 2

            T = phys.state_variable_triplet.T(rhod_mean, thd[cid])
            p = phys.state_variable_triplet.p(rhod_mean, T, qv[cid])
            pv = phys.state_variable_triplet.pv(p, qv[cid])
            lv = phys.latent_heat.lv(T)
            pvs = phys.saturation_vapour_pressure.pvs_Celsius(T - const.T0)
            RH = pv / pvs
            DTp = phys.diffusion_thermics.D(T, p)
            lambdaK = phys.diffusion_kinetics.lambdaK(T, p)
            lambdaD = phys.diffusion_kinetics.lambdaD(DTp, T)

            dvfloat = PrecisionResolver.get_floating_point
            self.__update_volume.launch_n(len(n), (v.data, vdry.data,
                                                   dvfloat(T), dvfloat(kappa), dvfloat(DTp), dvfloat(lambdaD),
                                                   dvfloat(lambdaK), dvfloat(RH), dvfloat(pvs), dvfloat(lv),
                                                   dvfloat(dt), dvfloat(p), dvfloat(self.RH_rtol), dvfloat(rtol_x),
                                                   dvfloat(self.max_iters))
                                          )
            self.calculate_m_l(self.ml_new, v, n, cell_id)

            # ...
            dml_dt = (self.ml_new[cid] - self.ml_old[cid]) / dt
            dqv_dt_corr = - dml_dt / m_d
            dthd_dt_corr = phys.state_variable_triplet.dthd_dt(rhod=rhod_mean, thd=thd[cid], T=T, dqv_dt=dqv_dt_corr, lv=lv)

            thd[cid] += dt * (dthd_dt_pred / 2 + dthd_dt_corr)
            qv[cid] += dt * (dqv_dt_pred / 2 + dqv_dt_corr)
            self.ml_old[:] = self.ml_new[:]
            # ...

    def make_condensation_solver(self, dt, *, dt_range, adaptive, fuse, multiplier, RH_rtol, max_iters):
        if adaptive:
            raise NotImplementedError()
        self.RH_rtol = RH_rtol
        self.max_iters = max_iters
        return None
