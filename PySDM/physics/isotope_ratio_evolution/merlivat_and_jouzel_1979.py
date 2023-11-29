"""
see derivation of eq. (12) in [Merlivat and Jouzel 1979](TODO-doi)
"""


class MerlivatAndJouzel1979:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def d_Rv_over_Rv(_, alpha, d_alpha, n_vapour, d_n_vapour, n_liquid):
        return ((alpha - 1) * d_n_vapour - n_liquid * d_alpha) / (
            n_vapour + alpha * n_liquid
        )
