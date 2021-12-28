"""
temperature-independent latent heat of vaporization
"""


class Constant:
    def __init__(self, const):
        pass

    @staticmethod
    def lv(const, T):  # pylint: disable=unused-argument
        return const.l_tri
