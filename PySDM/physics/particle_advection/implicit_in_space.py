"""
eqs. 14-16 in [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015)
"""


class ImplicitInSpace:
    def __init__(self, _):
        pass

    @staticmethod
    def displacement(_, omega, c_l, c_r):
        return (omega * c_r + c_l * (1 - omega)) / (1 - c_r + c_l)
