"""
[Bohren 1987](https://doi.org/10.1119/1.15109) Eq. 14
"""


class Bohren1987:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def albedo(const, g, tau):
        return ((const.ONE - g) * tau) / (const.TWO + (const.ONE - g) * tau)
