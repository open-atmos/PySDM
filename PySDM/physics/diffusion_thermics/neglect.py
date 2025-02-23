"""
constant diffusion coefficient formulation
"""


class Neglect:
    def __init__(self, _):
        pass

    @staticmethod
    def D(const, T, p):  # pylint: disable=unused-argument
        return const.D0

    @staticmethod
    def K(const, T, p):  # pylint: disable=unused-argument
        return const.K0
