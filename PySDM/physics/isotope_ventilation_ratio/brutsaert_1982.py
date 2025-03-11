"""
based on [Brutsaert 1982](https://doi.org/10.1007/978-94-017-1497-6) Springer Netherlands
statement about ventilation coefficient for heavy isotopes on pp. 92-93.
"""


class Brutsaert1982:  # pylint disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def isotope_ventilation_coefficient(
        const, sqrt_re_times_cbrt_sc, diffusivity_ratio
    ):
        return (
            const.FROESSLING_1938_A
            + const.FROESSLING_1938_B * sqrt_re_times_cbrt_sc * diffusivity_ratio
        )

    @staticmethod
    def isotope_ventilation_ratio(ventilation_coefficient, diffusivity_ratio):
        """heavy to light isotope ventilation ratio"""
        D_ratio_cbrt = diffusivity_ratio ** (1 / 3)
        return (1 - D_ratio_cbrt) / ventilation_coefficient + D_ratio_cbrt
