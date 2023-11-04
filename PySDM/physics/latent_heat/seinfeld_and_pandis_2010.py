"""
temperature-dependent latent heat of vaporization from Seinfeld and Pandis
"""


class SeinfeldPandis:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def lv(const, T):
        return const.l_tri * (const.T_tri / T) ** (const.l_l19_a + const.l_l19_b * T)
