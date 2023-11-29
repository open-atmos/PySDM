"""
Various (hopefully) undebatable formulae

`erfinv` approximation based on eqs. 11-12 from Vedder 1987, https://doi.org/10.1119/1.15018
"""
import numpy as np


class Trivia:
    def __init__(self, _):
        pass

    @staticmethod
    def volume_of_density_mass(rho, m):
        return m / rho

    @staticmethod
    def radius(const, volume):
        return np.power(volume / const.PI_4_3, const.ONE_THIRD)

    @staticmethod
    def area(const, radius):
        return const.PI * const.FOUR * np.power(radius, const.TWO)

    @staticmethod
    def volume(const, radius):
        return const.PI_4_3 * np.power(radius, const.THREE)

    @staticmethod
    def sphere_surface(const, diameter):
        return const.PI * diameter**2

    @staticmethod
    def explicit_euler(y, dt, dy_dt):
        return y + dt * dy_dt

    @staticmethod
    def within_tolerance(error_estimate, value, rtol):
        return error_estimate < rtol * np.abs(value)

    @staticmethod
    def H2pH(H):
        return -np.log10(H * 1e-3)

    @staticmethod
    def pH2H(pH):
        return np.power(10, -pH) * 1e3

    @staticmethod
    def vant_hoff(const, K, dH, T, *, T_0):
        return K * np.exp(-dH / const.R_str * (1 / T - 1 / T_0))

    @staticmethod
    def tdep2enthalpy(const, tdep):
        return -tdep * const.R_str

    @staticmethod
    def arrhenius(const, A, Ea, T):
        return A * np.exp(-Ea / (const.R_str * T))

    @staticmethod
    def mole_fraction_2_mixing_ratio(mole_fraction, specific_gravity):
        return specific_gravity * mole_fraction / (1 - mole_fraction)

    @staticmethod
    def mixing_ratio_2_mole_fraction(mixing_ratio, specific_gravity):
        return mixing_ratio / (specific_gravity + mixing_ratio)

    @staticmethod
    def p_d(const, p, water_vapour_mixing_ratio):
        return p * (1 - 1 / (1 + const.eps / water_vapour_mixing_ratio))

    @staticmethod
    def th_std(const, p, T):
        return T * np.power(const.p1000 / p, const.Rd_over_c_pd)

    @staticmethod
    def unfrozen_and_saturated(_, water_mass, relative_humidity):
        return water_mass > 0 and relative_humidity > 1

    @staticmethod
    def frozen_and_above_freezing_point(const, water_mass, temperature):
        return water_mass < 0 and temperature > const.T0

    @staticmethod
    def erfinv_approx(const, c):
        return (
            2
            * np.sqrt(const.VEDDER_1987_A)
            * np.sinh(
                np.arcsinh(
                    np.arctanh(c)
                    / 2
                    / const.VEDDER_1987_b
                    / np.power(const.VEDDER_1987_A, const.ONE_AND_A_HALF)
                )
                / 3
            )
        )
