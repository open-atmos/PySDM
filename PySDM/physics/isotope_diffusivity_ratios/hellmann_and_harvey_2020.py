"""
Ratios of diffusivity in air of heavy vs. light isotope using fits provided in
[Hellmann & Harvey 2020](https://doi.org/10.1029/2020GL089999)
"""


class HellmannAndHarvey2020:
    def __init__(self, _):
        pass

    @staticmethod
    def ratio_2H(const, temperature):
        return (
            const.HELLMANN_HARVEY_EQ6_COEFF0
            + const.HELLMANN_HARVEY_EQ6_COEFF1
            / (temperature / const.HELLMANN_HARVEY_T_UNIT)
            + const.HELLMANN_HARVEY_EQ6_COEFF2
            / (temperature / const.HELLMANN_HARVEY_T_UNIT) ** (const.TWO_AND_A_HALF)
        )

    @staticmethod
    def ratio_17O(const, temperature):
        return (
            const.HELLMANN_HARVEY_EQ7_COEFF0
            + const.HELLMANN_HARVEY_EQ7_COEFF1
            / (temperature / const.HELLMANN_HARVEY_T_UNIT) ** (const.ONE_HALF)
            + const.HELLMANN_HARVEY_EQ7_COEFF2
            / (temperature / const.HELLMANN_HARVEY_T_UNIT) ** (const.TWO_AND_A_HALF)
        )

    @staticmethod
    def ratio_18O(const, temperature):
        return (
            const.HELLMANN_HARVEY_EQ8_COEFF0
            + const.HELLMANN_HARVEY_EQ8_COEFF1
            / (temperature / const.HELLMANN_HARVEY_T_UNIT) ** (const.ONE_HALF)
            + const.HELLMANN_HARVEY_EQ8_COEFF2
            / (temperature / const.HELLMANN_HARVEY_T_UNIT) ** (const.THREE)
        )
