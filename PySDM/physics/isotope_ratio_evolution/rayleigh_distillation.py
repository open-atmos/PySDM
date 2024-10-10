""" Fractional distillation assuming continuous removal under equilibrium """


class RayleighDistillation:  # pylint:disable=too-few-public-methods
    """https://en.wikipedia.org/wiki/Rayleigh_fractionation"""

    def __init__(self, _):
        pass

    @staticmethod
    def R_over_R0(_, X_over_X0, a):
        return X_over_X0 ** (a - 1)
