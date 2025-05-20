"""
single-equation approximation of the vapour and heat diffusion problem
as given in eq. 3.11 in [Mason 1971](https://archive.org/details/physicsofclouds0000maso)
(see also discussion of the ventilation effect on page 125)
"""


class Mason1971:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def r_dr_dt(RH_eq, RH, Fk_factor_for_heat_conduction, Fd_factor_for_mass_diffusion):
        return (RH - RH_eq) / (
            Fk_factor_for_heat_conduction + Fd_factor_for_mass_diffusion
        )
