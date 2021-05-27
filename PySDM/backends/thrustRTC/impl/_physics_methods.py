from ..conf import trtc
from PySDM.backends.thrustRTC.impl.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver


class PhysicsMethods:
    def __init__(self):
        phys = self.formulae

        self._temperature_pressure_RH_body = trtc.For(["rhod", "thd", "qv", "T", "p", "RH"], "i", f'''
            T[i] = {phys.state_variable_triplet.T.c_inline(rhod="rhod[i]", thd="thd[i]")};
            p[i] = {phys.state_variable_triplet.p.c_inline(rhod="rhod[i]", T="T[i]", qv="qv[i]")};
            RH[i] = {phys.state_variable_triplet.pv.c_inline(p="p[i]", qv="qv[i]")} / {phys.saturation_vapour_pressure.pvs_Celsius.c_inline(T="T[i] - const.T0")};
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__explicit_euler_body = trtc.For(("y", "dt", "dy_dt"), "i", f'''
            y[i] = {phys.trivia.explicit_euler.c_inline(y="y[i]", dt="dt", dy_dt="dy_dt")};
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__critical_volume_body = trtc.For(("v_cr", "kappa", "v_dry", "v_wet", "T", "cell"), "i", f'''
            auto sigma = {phys.surface_tension.sigma.c_inline(T="T[cell[i]]", v_wet="v_wet[i]", v_dry="v_dry[i]")};
            auto r_cr = {phys.hygroscopicity.r_cr.c_inline(
                kp="kappa",
                rd3="v_dry[i] / const.pi_4_3",
                T="T[cell[i]]",
                sgm="sigma"
            )};
            v_cr[i] = {phys.trivia.volume.c_inline(radius="r_cr")};
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__terminal_velocity_body = trtc.For(["values", "radius", "k1", "k2", "k3", "r1", "r2"], "i", '''
            if (radius[i] < r1) {
                values[i] = k1 * radius[i] * radius[i];
            }
            else {
                if (radius[i] < r2) {
                    values[i] = k2 * radius[i];
                }
                else {
                    values[i] = k3 * pow(radius[i], (real_type)(.5));
                }
            }
            '''.replace("real_type", PrecisionResolver.get_C_type()))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def critical_volume(self, v_cr, kappa, v_dry, v_wet, T, cell):
        kappa = PrecisionResolver.get_floating_point(kappa)
        self.__critical_volume_body.launch_n(
            v_cr.shape[0], (v_cr.data, kappa, v_dry.data, v_wet.data, T.data, cell.data)
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def temperature_pressure_RH(self, rhod, thd, qv, T, p, RH):
        self._temperature_pressure_RH_body.launch_n(
            T.shape[0], (rhod.data, thd.data, qv.data, T.data, p.data, RH.data))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def terminal_velocity(self, values, radius, k1, k2, k3, r1, r2):
        k1 = PrecisionResolver.get_floating_point(k1)
        k2 = PrecisionResolver.get_floating_point(k2)
        k3 = PrecisionResolver.get_floating_point(k3)
        r1 = PrecisionResolver.get_floating_point(r1)
        r2 = PrecisionResolver.get_floating_point(r2)
        self.__terminal_velocity_body.launch_n(values.size(), [values, radius, k1, k2, k3, r1, r2])

    @nice_thrust(**NICE_THRUST_FLAGS)
    def explicit_euler(self, y, dt, dy_dt):
        dt = PrecisionResolver.get_floating_point(dt)
        dy_dt = PrecisionResolver.get_floating_point(dy_dt)
        self.__explicit_euler_body.launch_n(y.shape[0], (y.data, dt, dy_dt))
