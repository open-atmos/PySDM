"""
single-equation approximation of the vapour and heat diffusion problem
as proposed in [Howell 1949](https://doi.org/10.1175/1520-0469(1949)006%3C0134:TGOCDI%3E2.0.CO;2)
same as in [Mason 1951](https://doi.org/10.1088/0370-1301/64/9/307)

The notation for terms associated with heat conduction and diffusion are from eq. 7.17
 in [Rogers & Yau 1971](https://archive.org/details/shortcourseinclo0000roge_m3k2).
"""

from .fick import Fick


class Howell1949(Fick):  # pylint: disable=too-few-public-methods

    @staticmethod
    def Fk(const, T, K, lv):
        """Thermodynamic term associated with heat conduction.

        Parameters
        ----------
        T
            Temperature.
        K
            Thermal diffusivity with heat ventilation factor.
        lv
            Latent heat of evaporation or sublimation.
        """
        return const.rho_w * lv / T / K * (lv / T / const.Rv)

    @staticmethod
    def r_dr_dt(RH_eq, RH, Fk, Fd):
        """Drop growth equation with radius r.

        Parameters
        ----------
        Fk
            Thermodynamic term associated with heat conduction from Rogers & Yau 1989.
        Fd
            Term associated with vapour diffusion.
        """
        return (RH - RH_eq) / (Fk + Fd)
