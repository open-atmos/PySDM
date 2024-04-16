""" eq. 28 in [Miyake et al. 1968](https://doi.org/10.2467/mripapers1950.19.2_243) """


class MiyakeEtAl1968:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau(const, e_s, D, M, vent_coeff, radius, alpha, temperature):
        return (radius**2 * alpha * const.rho_w * const.R_str * temperature) / (
            3 * e_s * D * M * vent_coeff
        )
