"""
Vapour pressure factors from Table V in [Van Hook 1968](https://10.1021/j100850a028)
"""

import numpy as np


class VanHook1968:
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_2H(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_2H_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_2H_B / T
            + const.VAN_HOOK_1968_ALPHA_L_2H_C
        )

    @staticmethod
    def alpha_i_2H(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_2H_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_2H_B / T
            + const.VAN_HOOK_1968_ALPHA_I_2H_C
        )

    @staticmethod
    def alpha_l_18O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_18O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_18O_B / T
            + const.VAN_HOOK_1968_ALPHA_L_18O_C
        )

    @staticmethod
    def alpha_i_18O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_18O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_18O_B / T
            + const.VAN_HOOK_1968_ALPHA_I_18O_C
        )

    @staticmethod
    def alpha_l_17O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_17O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_17O_B / T
            + const.VAN_HOOK_1968_ALPHA_L_17O_C
        )

    @staticmethod
    def alpha_i_17O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_17O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_17O_B / T
            + const.VAN_HOOK_1968_ALPHA_I_17O_C
        )

    @staticmethod
    def alpha_l_3H(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_3H_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_3H_B / T
            + const.VAN_HOOK_1968_ALPHA_L_3H_C
        )

    @staticmethod
    def alpha_i_3H(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_3H_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_3H_B / T
            + const.VAN_HOOK_1968_ALPHA_I_3H_C
        )

    @staticmethod
    def alpha_l_TOT(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_TOT_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_TOT_B / T
            + const.VAN_HOOK_1968_ALPHA_L_TOT_C
        )

    @staticmethod
    def alpha_i_TOT(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_TOT_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_TOT_B / T
            + const.VAN_HOOK_1968_ALPHA_I_TOT_C
        )

    @staticmethod
    def alpha_l_DOT(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_DOT_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_DOT_B / T
            + const.VAN_HOOK_1968_ALPHA_L_DOT_C
        )

    @staticmethod
    def alpha_i_DOT(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_DOT_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_DOT_B / T
            + const.VAN_HOOK_1968_ALPHA_I_DOT_C
        )

    @staticmethod
    def alpha_l_DOD(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_DOD_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_DOD_B / T
            + const.VAN_HOOK_1968_ALPHA_L_DOD_C
        )

    @staticmethod
    def alpha_i_DOD(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_DOD_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_DOD_B / T
            + const.VAN_HOOK_1968_ALPHA_I_DOD_C
        )
