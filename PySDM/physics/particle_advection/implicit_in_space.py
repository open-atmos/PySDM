"""
eqs. 14-16 in [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015)
"""


class ImplicitInSpace:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def displacement(_, position_in_cell, c_l, c_r):
        return (c_l * (1 - position_in_cell) + c_r * position_in_cell) / (1 - c_r + c_l)
