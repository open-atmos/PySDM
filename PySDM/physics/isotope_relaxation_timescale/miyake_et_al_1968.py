"""eq. 28 in [Miyake et al. 1968](https://doi.org/10.2467/mripapers1950.19.2_243)
theta is as discussed in [Kinzer & Gunn 1951 (J. Meteor.)](https://doi.org/10.1175/1520-0469(1951)008%3C0071:TETATR%3E2.0.CO;2)
"""  # pylint: disable=line-too-long


class MiyakeEtAl1968:  # pylint:disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(
        const, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments, unused-argument
        """
        relative growth of heavy isotope as a function of mass from eq. (28)

        Parameters
        ----------
        rho_s
            is equal to (e_s * M / R_str / T) in eq. (28)
        D
            diffusivity * theta, where theta from eq. (25) is ventilation_coefficient
        """
        return (radius**2 * alpha * const.rho_w) / (3 * rho_s * D)
