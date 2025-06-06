"""eq. 7 in [Jouzel et al. 1975](https://doi.org/10.1029/JC080i036p05015)
with assumptions that the supersaturation is absent (S=1)
and at constant vapour phase (R_liq = alpha * R_vap)"""


class JouzelEtAl1975:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(m_dm_dt):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        """
        return 1 / m_dm_dt

    @staticmethod
    def isotope_m_dm_dt(
        const, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments, unused-argument
        """relative growth of heavy isotope as a function of mass"""
        return abs(-3 * rho_s / radius**2 / const.rho_w / alpha * D_iso)
