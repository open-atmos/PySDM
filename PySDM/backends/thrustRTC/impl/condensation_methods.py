from PySDM.physics import constants as const
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl import nice_thrust, c_inline
from .precision_resolver import PrecisionResolver
from ..conf import trtc


class CondensationMethods:
    def __init__(self):
        phys = self.formulae

        # TODO #228: precision for consts
        self.__calculate_m_l = trtc.For(("ml", "v", "n", "cell_id"), "i", f'''
            atomicAdd((real_type*) &ml[cell_id[i]], n[i] * v[i] * {const.rho_w}); 
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__update_volume = trtc.For(("v", "vdry"), "i", f'''
            auto x_old = {c_inline(phys.condensation_coordinate.x, volume="v[i]")};
            auto r_old = {c_inline(phys.trivia.radius, volume="v[i]")};
            auto x_insane = {c_inline(phys.condensation_coordinate.x, volume="vdry[i]/100")};
            auto rd3 = vdry[i] / {const.pi_4_3};
            auto sgm = {c_inline(phys.surface_tension.sigma, T="T", v_wet="v[i]", v_dry="vdry[i]")};
            auto RH_eq = {c_inline(phys.hygroscopicity.RH_eq, r="r_old", T="T", kp="kappa", rd3="rd3", sgm="sgm")};

            v[i] = 0;
        '''.replace("real_type", PrecisionResolver.get_C_type()))

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
        self.ml_old = trtc.device_vector(PrecisionResolver.get_C_type(), n_cell)
        self.ml_new = trtc.device_vector(PrecisionResolver.get_C_type(), n_cell)

        if n_cell != 1:
            raise NotImplementedError()
        cid = 0
        dv_mean = dv
        dthd_dt = (pthd[cid] - thd[cid]) / dt
        dqv_dt = (pqv[cid] - qv[cid]) / dt
        rhod_mean = (prhod[cid] + rhod[cid]) / 2
        m_d = rhod_mean * dv_mean

        n_substeps = 10  # TODO #228!
        for i in range(n_substeps):
            T = phys.state_variable_triplet.T(rhod_mean, thd[cid])
            p = phys.state_variable_triplet.p(rhod_mean, T, qv[cid])
            pv = phys.state_variable_triplet.pv(p, qv[cid])
            lv = phys.latent_heat.lv(T)
            pvs = phys.saturation_vapour_pressure.pvs_Celsius(T - const.T0)
            RH = pv / pvs
            DTp = phys.diffusion_thermics.D(T, p)
            lambdaK = phys.diffusion_kinetics.lambdaK(T, p)
            lambdaD = phys.diffusion_kinetics.lambdaD(DTp, T)

            trtc.Fill(self.ml_old, 0)
            self.__calculate_m_l.launch_n(len(n), (self.ml_old, v.data, n.data, cell_id.data))
            self.__update_volume.launch_n(len(n), (v.data, vdry.data))
            self.__calculate_m_l.launch_n(len(n), (self.ml_new, v.data, n.data, cell_id.data))

    def make_condensation_solver(self, dt, dt_range, adaptive):
        # TODO #228: uncomment
        # if adaptive:
        #     raise NotImplementedError()
        return None
