"""
Vapour pressure factors from Table V in [Van Hook 1968](https://10.1021/j100850a028)
"""
import numpy as np


class VanHook1968:
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_HDO(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_HDO_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_HDO_B / T
            + const.VAN_HOOK_1968_ALPHA_L_HDO_C
        )

    @staticmethod
    def alpha_i_HDO(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_HDO_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_HDO_B / T
            + const.VAN_HOOK_1968_ALPHA_I_HDO_C
        )

    @staticmethod
    def alpha_l_H2_18O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_H218O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_H218O_B / T
            + const.VAN_HOOK_1968_ALPHA_L_H218O_C
        )

    @staticmethod
    def alpha_i_H2_18O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_H218O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_H218O_B / T
            + const.VAN_HOOK_1968_ALPHA_I_H218O_C
        )

    @staticmethod
    def alpha_l_H2_17O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_H217O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_H217O_B / T
            + const.VAN_HOOK_1968_ALPHA_L_H217O_C
        )

    @staticmethod
    def alpha_i_H2_17O(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_H217O_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_H217O_B / T
            + const.VAN_HOOK_1968_ALPHA_I_H217O_C
        )

    @staticmethod
    def alpha_l_HOT(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_L_HOT_A / T**2
            + const.VAN_HOOK_1968_ALPHA_L_HOT_B / T
            + const.VAN_HOOK_1968_ALPHA_L_HOT_C
        )

    @staticmethod
    def alpha_i_HOT(const, T):
        return np.exp(
            const.VAN_HOOK_1968_ALPHA_I_HOT_A / T**2
            + const.VAN_HOOK_1968_ALPHA_I_HOT_B / T
            + const.VAN_HOOK_1968_ALPHA_I_HOT_C
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
