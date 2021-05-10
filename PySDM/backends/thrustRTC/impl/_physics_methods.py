"""
Created at 20.03.2020
"""

from ..conf import trtc
from PySDM.backends.thrustRTC.impl.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver
from PySDM.backends.thrustRTC.impl.c_inline import c_inline


class PhysicsMethods:
    def __init__(self):
        self._temperature_pressure_RH_body = trtc.For(["rhod", "thd", "qv", "T", "p", "RH"], "i", f'''
            T[i] = {c_inline(self.formulae.state_variable_triplet.T, rhod="rhod[i]", thd="thd[i]")};
            p[i] = {c_inline(self.formulae.state_variable_triplet.p, rhod="rhod[i]", T="T[i]", qv="qv[i]")};
            RH[i] = {c_inline(self.formulae.state_variable_triplet.pv, p="p[i]", qv="qv[i]")} / {c_inline(self.formulae.saturation_vapour_pressure.pvs_Celsius, T="T[i] - const.T0")};
        ''')

    @nice_thrust(**NICE_THRUST_FLAGS)
    def temperature_pressure_RH(self, rhod, thd, qv, T, p, RH):
        self._temperature_pressure_RH_body.launch_n(
            T.shape[0], (rhod.data, thd.data, qv.data, T.data, p.data, RH.data))

    __terminal_velocity_body = trtc.For(["values", "radius", "k1", "k2", "k3", "r1", "r2"], "i", '''
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

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def terminal_velocity(values, radius, k1, k2, k3, r1, r2):
        k1 = PrecisionResolver.get_floating_point(k1)
        k2 = PrecisionResolver.get_floating_point(k2)
        k3 = PrecisionResolver.get_floating_point(k3)
        r1 = PrecisionResolver.get_floating_point(r1)
        r2 = PrecisionResolver.get_floating_point(r2)
        PhysicsMethods.__terminal_velocity_body.launch_n(values.size(), [values, radius, k1, k2, k3, r1, r2])
