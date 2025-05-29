"""based on [Picciotto et al. 1960](https://doi.org/10.1038/187857a0)
where delta(2H)= slope_coeff * delta(18O) + intercept_coeff formulae is given
with swapped 2H and 18O in a paper
"""


class PicciottoEtAl1960:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def d18O_of_d2H(const, delta_2H):
        return (
            delta_2H - const.PICCIOTTO_18O_TO_2H_INTERCEPT_COEFF
        ) / const.PICCIOTTO_18O_TO_2H_SLOPE_COEFF
