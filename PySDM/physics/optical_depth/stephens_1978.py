"""
[Stephens 1978](https://doi.org/10.1175/1520-0469(1978)035%3C2123:RPIEWC%3E2.0.CO;2)
Eq. 7 for optical depth, where LWP is in g/m^2 and reff is in um.
"""


class Stephens1978:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(const, LWP, reff):
        return (const.ONE_AND_A_HALF * LWP) / (const.rho_w * reff)
