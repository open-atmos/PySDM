"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class PhysicsMethods:
    @staticmethod
    def explicit_in_space(omega, c_l, c_r):
        return "c_l * (1 - omega) + c_r * omega;"

    @staticmethod
    def implicit_in_space(omega, c_l, c_r):
        """
        see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
        """
        result = '''
            auto = dC = c_r - c_l;
            (omega * dC + c_l) / (1 - dC);
        '''
        return result

    @staticmethod
    def temperature_pressure_RH(rhod, thd, qv):
        return "temperature_pressure_RH;"

    @staticmethod
    def radius(volume):
        return ""

    @staticmethod
    def dr_dt_MM(r, T, p, RH, kp, rd):
        return ""

    @staticmethod
    def dr_dt_FF(r, T, p, qv, kp, rd, T_i):
        return ""

    @staticmethod
    def dthd_dt(rhod, thd, T, dqv_dt):
        return ""
