"""
particle water mass as condensation coordinate (i.e. no transformation)
"""


class WaterMass:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(x, dm_dt):  # pylint: disable=unused-argument
        return dm_dt

    @staticmethod
    def mass(x):
        return x

    @staticmethod
    def x(mass):
        return mass
