"""
kinetic fractionation factor in the framework of the Craig-Gordon model
as given in eq. 1.5 in
[Rozanski_et_al_2001 (UNESCO, ed. Mook)
](https://web.archive.org/web/20160322221332/https://hydrology.nl/images/docs/ihp/Mook_III.pdf)
and as used in [Pierchala et al. 2022](https://doi.org/10.1016/j.gca.2022.01.020)
"""


class CraigGordon:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_kinetic(*, relative_humidity, turbulence_parameter_n, delta_diff, theta):
        """delta_diff = 1 - heavy_to_light_diffusivity_ratio"""
        return 1 + theta * turbulence_parameter_n * delta_diff * (1 - relative_humidity)
