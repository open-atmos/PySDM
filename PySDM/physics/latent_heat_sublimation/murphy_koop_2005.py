"""
temperature-dependent latent heat of sublimation from Murphy and Koop (2005) Eq. (5)
"""


class MurphyKoop:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def ls(const, T):  # pylint: disable=unused-argument
        return (const.MK05_SUB_C1 + const.MK05_SUB_C2 * T - const.MK05_SUB_C3 * T**2. + const.MK05_SUB_C4 * np.exp(-(T/const.MK05_SUB_C5)**2.)) / const.Mv
