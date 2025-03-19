"""
spherical particles with constant density of ice
"""
from PySDM.physics.trivia import Trivia

class LiquidSpheres:
    def __init__(self, const, _):
        pass

    @staticmethod
    def mass_to_volume(const, mass):
        return (
            Trivia.volume_of_density_mass(const, const.rho_w, mass)
        )

    @staticmethod
    def volume_to_mass(const, volume):
        return (
            Trivia.mass_of_density_volume(const.rho_w, volume)
        )

    @staticmethod
    def volume_to_radius(const, volume):
        return (
            Trivia.radius(const, volume)
        )

    @staticmethod
    def radius_to_volume(const, radius):
        return (
            Trivia.sphere_radius_to_volume(const, radius)
        )

    @staticmethod
    def radius_to_mass(const, radius, mass_density):
        return (
                Trivia.sphere_radius_to_volume(const, radius) * const.rho_w
        )

    @staticmethod
    def radius_to_area(const, radius):
        return (
            Trivia.area(const, radius)
        )

    @staticmethod
    def mass_to_capacity(const, mass):
        return (
            Trivia.sphere_mass_to_radius(const, mass, const.rho_w, )
        )

    @staticmethod
    def maximum_diameter(const, mass):
        return (
                Trivia.sphere_mass_to_radius(const, mass, const.rho_w) * 2
        )