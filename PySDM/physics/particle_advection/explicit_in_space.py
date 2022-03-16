"""
basic explicit-in-space Euler scheme
"""


class ExplicitInSpace:
    def __init__(self, _):
        pass

    @staticmethod
    def displacement(_, omega, c_l, c_r):
        return c_l * (1 - omega) + c_r * omega
