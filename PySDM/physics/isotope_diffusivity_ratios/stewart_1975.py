"""
Temperature-independent ratio of vapour diffusivity in air for heavy vs. light isotopes
assuming same collision diameters for different isotopic water molecules, see:
eq. (8) in [Stewart 1975](https://doi.org/10.1029/JC080i009p01133),
eq. (1) in [Merlivat 1978](https://doi.org/10.1063/1.436884),
eq. (6) in [Cappa et al. 2003](https://doi.org/10.1029/2003JD003597),
eq. (22) in [Horita et al. 2008](https://doi.org/10.1080/10256010801887174),
eq. (3) in [Hellmann and Harvey 2020](https://doi.org/10.1029/2020GL089999).

All functions return constants, so there is a potential overhead in computing them on each call,
but this variant is provided for historical reference only, hence leaving like that.
"""


class Stewart1975:
    def __init__(self, _):
        pass

    @staticmethod
    def ratio_2H(const, temperature):  # pylint: disable=unused-argument
        return (
            (
                (2 * const.M_1H + const.M_16O)
                * (const.Md + const.M_2H + const.M_1H + const.M_16O)
            )
            / (
                (const.M_2H + const.M_1H + const.M_16O)
                * (const.Md + (2 * const.M_1H + const.M_16O))
            )
        ) ** const.ONE_HALF

    @staticmethod
    def ratio_18O(const, temperature):  # pylint: disable=unused-argument
        return (
            ((2 * const.M_1H + const.M_16O) * (const.Md + 2 * const.M_1H + const.M_18O))
            / (
                (2 * const.M_1H + const.M_18O)
                * (const.Md + (2 * const.M_1H + const.M_16O))
            )
        ) ** const.ONE_HALF
