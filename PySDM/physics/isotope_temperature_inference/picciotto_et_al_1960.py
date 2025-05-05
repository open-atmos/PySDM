"""based on [Picciotto et al. 1960](https://doi.org/10.1038/187857a0)
where delta(T)=-(a*T + b) formulae given, here cast as T(delta)=(-delta-b)/a"""


class PicciottoEtAl1960:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def temperature_from_delta_18O(const, delta_18O):
        return const.T0 + (-delta_18O - const.PICCIOTTO_18O_B) / const.PICCIOTTO_18O_A

    @staticmethod
    def temperature_from_delta_2H(const, delta_2H):
        return const.T0 + (-delta_2H - const.PICCIOTTO_2H_B) / const.PICCIOTTO_2H_A
