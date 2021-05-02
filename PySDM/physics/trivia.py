from PySDM.physics import constants as const
from numpy import abs, log10, exp


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

    @staticmethod
    def H2pH(H):
        return -log10(H * 1e-3)

    @staticmethod
    def pH2H(pH):
        return 10**(-pH) * 1e3

    @staticmethod
    def vant_hoff(K, dH, T, *, T_0):
        return K * exp(-dH / const.R_str * (1 / T - 1 / T_0))

    @staticmethod
    def tdep2enthalpy(tdep):
        return -tdep * const.R_str

    @staticmethod
    def arrhenius(A, Ea, T):
        return A * exp(-Ea / (const.R_str * T))
