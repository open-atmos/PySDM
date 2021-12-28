"""
no transition-regime corrections formulation
"""


class Neglect:
    def __init__(self, const):
        pass

    @staticmethod
    def lambdaD(const, D, T):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def lambdaK(const, T, p):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def DK(const, DK, r, lmbd):  # pylint: disable=unused-argument
        return DK
