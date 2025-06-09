"""
based on [Brutsaert 1982](https://doi.org/10.1007/978-94-017-1497-6) Springer Netherlands
statement about ventilation coefficient for heavy isotopes on pp. 92-93.
"""


class Brutsaert1982:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def ratio_heavy_to_light(ventilation_coefficient, diffusivity_ratio_heavy_to_light):
        """heavy to light isotope ventilation ratio"""
        return (
            1 - diffusivity_ratio_heavy_to_light ** (1 / 3)
        ) / ventilation_coefficient + diffusivity_ratio_heavy_to_light ** (1 / 3)
