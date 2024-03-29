"""
see derivation of eq. (12) in [Merlivat and Jouzel 1979](https://doi.org/10.1029/JC084iC08p05029)
(for constant alpha leads to eq. (13) in
[Gedzelman & Arnold 1994](https://doi.org/10.1029/93jd03518))
"""


class MerlivatAndJouzel1979:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def d_Rv_over_Rv(_, alpha, d_alpha, n_vapour, d_n_vapour, n_liquid):
        return ((alpha - 1) * d_n_vapour - n_liquid * d_alpha) / (
            n_vapour + alpha * n_liquid
        )
