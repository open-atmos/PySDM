"""
spherical particles with constant density of water
"""


class LiquidSphere:
    def __init__(self, _):
        pass

    @staticmethod
    def mass_to_volume(const, mass):
        return mass / const.rho_w

    @staticmethod
    def mass_to_surface(const, mass):
        pass

    @staticmethod
    def volume_to_mass(const, volume):
        return const.rho_w * volume
