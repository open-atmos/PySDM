"""
kinetic fractionation factor from [Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749)
(as eq. 3e for n=1 in [Stewart 1975](https://doi.org/10.1029/JC080i009p01133))
"""


class JouzelAndMerlivat1984:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_kinetic(
        alpha_equilibrium, saturation_over_ice, diffusivity_ratio_heavy_to_light
    ):
        """eq. (11)"""
        return saturation_over_ice / (
            alpha_equilibrium
            / diffusivity_ratio_heavy_to_light
            * (saturation_over_ice - 1)
            + 1
        )
