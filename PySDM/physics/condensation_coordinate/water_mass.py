"""
particle water mass as condensation coordinate (i.e. no transformation)
"""


class WaterMass:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(m, dm_dt):  # pylint: disable=unused-argument
        return dm_dt

    @staticmethod
    def mass(x):
        return x

    @staticmethod
    def x(mass):
        return mass

    @staticmethod
    def x_max(const):
        """1 kg droplet!"""
        return const.ONE
