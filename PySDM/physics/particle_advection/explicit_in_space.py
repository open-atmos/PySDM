"""
basic explicit-in-space Euler scheme
"""


class ExplicitInSpace:
    def __init__(self, const):
        pass

    @staticmethod
    def displacement(const, omega, c_l, c_r):
        return c_l * (1 - omega) + c_r * omega
