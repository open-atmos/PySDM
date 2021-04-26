from PySDM.physics import constants as const
from numpy import abs


class Trivia:
    @staticmethod
    def volume_of_density_mass(rho, m):
        return m / rho

    @staticmethod
    def radius(volume):
        return (volume / const.pi_4_3)**(1/3)

    @staticmethod
    def volume(radius):
        return const.pi_4_3 * radius**3

    @staticmethod
    def explicit_euler(y, dt, dy_dt):
        y += dt * dy_dt

    @staticmethod
    def within_tolerance(error_estimate, value, rtol):
        return error_estimate < rtol * abs(value)
