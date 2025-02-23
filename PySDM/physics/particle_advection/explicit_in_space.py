"""
basic explicit-in-space Euler scheme
"""


class ExplicitInSpace:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def displacement(_, position_in_cell, c_l, c_r):
        return c_l * (1 - position_in_cell) + c_r * position_in_cell
