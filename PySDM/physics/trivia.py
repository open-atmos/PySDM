from PySDM.physics import constants as const


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
