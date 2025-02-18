"""eq. 7 in [Stewart 1975](https://doi.org/10.1029/JC080i009p01133)"""


class Stewart1975:
    @staticmethod
    # pylint: disable=too-many-arguments
    def tau(const, e_s, Dn, M_iso, vent_coeff, radius, alpha, temperature):
        """relaxation time for stationary droplet; Dn denotes D^n"""
        return (
            radius**2
            * alpha
            * const.rho_w
            * temperature
            / 3
            / vent_coeff
            / Dn
            / e_s
            / M_iso
        )
