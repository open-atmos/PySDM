"""eq. 28 in [Miyake et al. 1968](https://doi.org/10.2467/mripapers1950.19.2_243)
theta = 1 + E * sqrt(Re) / (sqrt(4 pi) D eta - mass ventilation factor.
"""


class MiyakeEtAl1968:
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
        return abs(-3 * rho_s / radius**2 / const.rho_w / alpha * D_iso)

    @staticmethod
    # pylint: disable=too-many-arguments unused-argument
    def tau_of_rdrdt(const, radius, r_dr_dt, alpha):
        return -(radius**2) / 3 / r_dr_dt * alpha
