"""eq. 7 in [Jouzel et al 1975](https://doi.org/10.1029/JC080i036p05015)"""


class JouzelEtAl1975:  # pylint: disable=too-few-public-methods
    @staticmethod
    # pylint: disable=too-many-arguments
    def tau(const, e_s, D_iso, M_iso, vent_coeff_iso, radius, alpha, temperature):
        """relaxation time for stationary droplet??"""
        return (
            radius**2
            * alpha
            * const.rho_w
            * temperature
            / 3
            / vent_coeff_iso
            / D_iso
            / e_s
            / M_iso
        )
