"""
temperature-independent latent heat of vaporization
"""


class Constant:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def lv(const, T):  # pylint: disable=unused-argument
        return const.l_tri
