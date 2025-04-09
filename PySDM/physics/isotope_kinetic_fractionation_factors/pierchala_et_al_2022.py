"""
kinetic fractionation factor and epsilon kinetic
from [Pierchala et al. 2022](https://doi.org/10.1016/j.gca.2022.01.020)
"""


class PierchalaEtAl2022:  # pylint: disable=too-few-public-methods
    @staticmethod
    def alpha_kinetic(eps_kinetic):
        return 1 - eps_kinetic

    @staticmethod
    def eps_kinetic(*, theta, n, eps_diff, relative_humidity):
        assert 0 <= n <= 1
        return theta * n * eps_diff * (1 - relative_humidity)
