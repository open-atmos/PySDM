"""
particle water mass as condensation coordinate (i.e. no transformation)
"""


class WaterMass:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(const, x, dm_dt):
        return dm_dt

    @staticmethod
    def mass(_, x):
        return x

    @staticmethod
    def x(_, mass):
        return mass
