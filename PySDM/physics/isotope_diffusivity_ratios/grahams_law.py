""" [Graham's law](https://en.wikipedia.org/wiki/Graham%27s_law)
    see also eq. (21) in [Horita et al. 2008](https://doi.org/10.1080/10256010801887174)
"""


class GrahamsLaw:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def ratio_2H(const, temperature):  # pylint: disable=unused-argument
        return (
            (2 * const.M_1H + const.M_16O) / (const.M_2H + const.M_1H + const.M_16O)
        ) ** const.ONE_HALF
