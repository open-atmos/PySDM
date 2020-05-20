"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .nice_thrust import nice_thrust
from .conf import NICE_THRUST_FLAGS


class PhysicsMethods:
    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def explicit_in_space(omega, c_l, c_r):
        return "c_l * (1 - omega) + c_r * omega;"

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def implicit_in_space(omega, c_l, c_r):
        """
        see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
        """
        result = "(omega * (c_r - c_l) + c_l) / (1 - (c_r - c_l));"
        return result

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def temperature_pressure_RH(rhod, thd, qv):
        return "temperature_pressure_RH;"

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def radius(volume):
        return ""

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def dr_dt_MM(r, T, p, RH, kp, rd):
        return ""

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def dr_dt_FF(r, T, p, qv, kp, rd, T_i):
        return ""

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def dthd_dt(rhod, thd, T, dqv_dt):
        return ""
