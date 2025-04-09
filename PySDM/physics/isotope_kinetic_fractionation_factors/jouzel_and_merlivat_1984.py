"""
kinetic fractionation factor from [Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749)
"""


class JouzelAndMerlivat1984:  # pylint: disable=too-few-public-methods
    @staticmethod
    def alpha_kinetic(alpha_equilibrium, relative_humidity, D_heavy2D_light):
        return (
            alpha_equilibrium
            * relative_humidity
            / (alpha_equilibrium / D_heavy2D_light * (relative_humidity - 1) + 1)
        )
