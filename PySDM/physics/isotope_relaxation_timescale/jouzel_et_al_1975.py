"""eq. 7 in [Jouzel et al. 1975](https://doi.org/10.1029/JC080i036p05015)
with assumptions that the supersaturation is absent (S=1)
and at constant vapour phase (R_liq = alpha * R_vap).
D_iso (diffusivity coefficient for heavy isotopes) is replaced by D in calculations of timescale
as stated to be very similar
"""


class JouzelEtAl1975:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(dm_dt_over_m):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        """
        return 1 / dm_dt_over_m

    @staticmethod
    def isotope_dm_dt_over_m(
        const, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments, unused-argument
        """relative growth of heavy isotope as a function of mass

        Parameters
        ----------
        D_iso
            diffusivity of the heavy isotope multiplied by ventilation coefficient (vent_coeff from
            [Kinzer & Gunn 1951 (J. Meteor.)](https://doi.org/10.1175/1520-0469(1951)008%3C0071:TETATR%3E2.0.CO;2)
            with empirical calculations of F-coefficient)
        """
        return 3 * rho_s / radius**2 / const.rho_w / alpha * D_iso
