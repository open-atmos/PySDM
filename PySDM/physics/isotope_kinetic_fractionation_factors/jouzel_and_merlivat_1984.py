"""
kinetic fractionation factor from [Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749)
(as eq. 3e for n=1 in [Stewart 1975](https://doi.org/10.1029/JC080i009p01133))
"""


class JouzelAndMerlivat1984:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_kinetic(
        alpha_equilibrium, relative_humidity, heavy_to_light_diffusivity_ratio
    ):
        """eq. (11) or eq. (14)"""
        return relative_humidity / (
            alpha_equilibrium
            / heavy_to_light_diffusivity_ratio
            * (relative_humidity - 1)
            + 1
        )
