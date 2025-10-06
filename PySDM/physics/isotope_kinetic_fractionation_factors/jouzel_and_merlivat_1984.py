"""
kinetic fractionation factor [Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749),
as eq. 3e for n=1 in [Stewart 1975](https://doi.org/10.1029/JC080i009p01133)
and eq. 1 in [Bolot 2013](https://doi.org/10.5194/acp-13-7903-2013),
where alpha_kinetic is multiplied by alpha equilibrium (eq. 1 defines effective alpha).
"""


class JouzelAndMerlivat1984:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_kinetic(alpha_equilibrium, saturation, D_ratio_heavy_to_light):
        """eq. (11)

        Parameters
        ----------
        alpha_equilibrium
            Equilibrium fractionation factor.
        saturation
            Over liquid water or ice.
        D_ratio_heavy_to_light
            Diffusivity ratio for heavy to light isotope.

        Returns
        ----------
        alpha_kinetic
            Kinetic fractionation factor for liquid water or ice."""
        return saturation / (
            alpha_equilibrium / D_ratio_heavy_to_light * (saturation - 1) + 1
        )
