"""
Various (hopefully) undebatable formulae
"""
from PySDM.physics import constants as const
from numpy import abs, log10, exp, power


class Trivia:
    @staticmethod
    def volume_of_density_mass(rho, m):
        return m / rho

    @staticmethod
    def radius(volume):
        return power(volume / const.pi_4_3, const.one_third)

    @staticmethod
    def volume(radius):
        return const.pi_4_3 * power(radius, const.three)

    @staticmethod
    def explicit_euler(y, dt, dy_dt):
        return y + dt * dy_dt

    @staticmethod
    def within_tolerance(error_estimate, value, rtol):
        return error_estimate < rtol * abs(value)

    @staticmethod
    def H2pH(H):
        return -log10(H * 1e-3)

    @staticmethod
    def pH2H(pH):
        return power(10, -pH) * 1e3

    @staticmethod
    def vant_hoff(K, dH, T, *, T_0):
        return K * exp(-dH / const.R_str * (1 / T - 1 / T_0))

    @staticmethod
    def tdep2enthalpy(tdep):
        return -tdep * const.R_str

    @staticmethod
    def arrhenius(A, Ea, T):
        return A * exp(-Ea / (const.R_str * T))

    @staticmethod
    def mole_fraction_2_mixing_ratio(mole_fraction, specific_gravity):
        return specific_gravity * mole_fraction / (1 - mole_fraction)

    @staticmethod
    def mixing_ratio_2_mole_fraction(mixing_ratio, specific_gravity):
        return mixing_ratio / (specific_gravity + mixing_ratio)

    @staticmethod
    def p_d(p, qv):
        return p * (1 - 1 / (1 + const.eps / qv))

    @staticmethod
    def th_std(p, T):
        return T * power(const.p1000 / p, const.Rd_over_c_pd)
