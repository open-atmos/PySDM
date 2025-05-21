"""
Fickian diffusion only drop growth
"""


class Fick:
    def __init__(self, _):
        pass

    @staticmethod
    def Fk(const, T, K, lv):  # pylint: disable=unused-argument
        """thermodynamic term associated with heat conduction"""

    @staticmethod
    def Fd(const, T, D, pvs):
        """the term associated with vapour diffusion"""
        return const.rho_w * const.Rv * T / D / pvs

    @staticmethod
    def r_dr_dt(RH_eq, RH, Fk, Fd):  # pylint: disable=unused-argument
        return (RH - RH_eq) / Fd
