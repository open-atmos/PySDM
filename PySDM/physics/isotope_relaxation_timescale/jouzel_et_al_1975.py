"""eq. 7 in [Jouzel et al. 1975](https://doi.org/10.1029/JC080i036p05015)
with assumptions that the supersaturation is absent (S=1)
and at constant vapour phase (R_liq = alpha * R_vap).
D_iso (diffusivity coefficient for heavy isotopes) is replaced by D in calculations of timescale
as stated to be very similar
ventilation coefficient from [Kinzer & Gunn 1951 (J. Meteor.)](https://doi.org/10.1175/1520-0469(1951)008%3C0071:TETATR%3E2.0.CO;2)
with empirical calculations of F-coefficient
"""  # pylint: disable=line-too-long


class JouzelEtAl1975:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(
        const, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments, unused-argument
        """relative growth of heavy isotope as a function of mass

        Parameters
        ----------
        D_iso
            diffusivity of the heavy isotope multiplied by ventilation coefficient
        """
        return (radius**2 * const.rho_w * alpha) / (3 * rho_s * D_iso)
