"""
spherical particles with constant density
"""
from PySDM.physics.trivia import Trivia

class Spheres:
    def __init__(self, _):
        pass

    @staticmethod
    def mass_to_volume(const, mass):
        return (
            Trivia.volume_of_density_mass(self.mass_density, mass)
        )

    @staticmethod
    def volume_to_mass(const, volume):
        return (
            Trivia.mass_of_density_volume(self.mass_density, volume)
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
    def radius_to_mass(const, radius):
        return (
            self.sphere_radius_to_volumeconst, radius) * self.mass_density
        )

    @staticmethod
    def radius_to_area(const, radius):
        return(
            Trivia.area(const, radius)
        )

    @staticmethod
    def mass_to_capacity(const, mass):
        return(
            Trivia.sphere_mass_to_radius(const, mass, self.mass_density)
        )

    @staticmethod
    def maximum_diameter(const, mass):
        return(
            Trivia.sphere_mass_to_radius(const, mass, self.mass_density) * 2.
        )
