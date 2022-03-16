"""
no transition-regime corrections formulation
"""


class Neglect:
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(_, D, T):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def lambdaK(_, T, p):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def D(_, D, r, lmbd):  # pylint: disable=unused-argument
        return D

    @staticmethod
    def K(_, K, r, lmbd):  # pylint: disable=unused-argument
        return K
