"""
Fickian diffusion only drop growth

The notation for terms associated with heat conduction and diffusion are from
[Rogers & Yau 1989](https://archive.org/details/shortcourseinclo0000roge_m3k2).
"""


class Fick:
    def __init__(self, _):
        pass

    @staticmethod
    def Fk(const, T, K, lv):  # pylint: disable=unused-argument
        """heat conduction not taken into account"""
        return 0

    @staticmethod
    def Fd(const, T, D, pvs):
        """the term associated with vapour diffusion"""
        return const.rho_w * const.Rv * T / D / pvs

    @staticmethod
    def r_dr_dt(RH_eq, RH, Fk, Fd):  # pylint: disable=unused-argument
        """Drop growth equation with radius r.

        Parameters
        ----------
        Fk
            Thermodynamic term associated with heat conduction from Rogers & Yau 1989.
        Fd
            Term associated with vapour diffusion.
        """
        return (RH - RH_eq) / Fd
