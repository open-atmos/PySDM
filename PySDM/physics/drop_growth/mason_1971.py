"""
single-equation approximation of the vapour and heat diffusion problem
as given in eq. 3.11 in [Mason 1971](https://archive.org/details/physicsofclouds0000maso)
(see also discussion of the ventilation effect on page 125).
The difference between Howell'49 and Mason'71 is `-1` in the `Fk` definition.

The notation for terms associated with heat conduction and diffusion are from
[Rogers & Yau 1989](https://archive.org/details/shortcourseinclo0000roge_m3k2).
"""

from .howell_1949 import Howell1949


class Mason1971(Howell1949):  # pylint: disable=too-few-public-methods

    @staticmethod
    def Fk(const, T, K, lv):
        """thermodynamic term associated with heat conduction"""
        return const.rho_w * lv / T / K * (lv / T / const.Rv - 1)
