"""
spherical particles with constant density of water
Reynolds number calculation assumes length scale is particle diameter
"""

import numpy as np


class LiquidSpheres:
    def __init__(self, _):
        pass

    @staticmethod
    def supports_mixed_phase(_=None):
        return False

    @staticmethod
    def mass_to_volume(const, mass):
        return mass / const.rho_w

    @staticmethod
    def volume_to_mass(const, volume):
        return const.rho_w * volume

    @staticmethod
    def radius_to_mass(const, radius):
        return const.rho_w * const.PI_4_3 * np.power(radius, const.THREE)

    @staticmethod
    def reynolds_number(_, radius, velocity_wrt_air, dynamic_viscosity, density):
        return 2 * radius * velocity_wrt_air * density / dynamic_viscosity

    @staticmethod
    def dm_dt(const, r, r_dr_dt):
        """
        dm_dt = d(4/3 pi r^3 rho_w) / dt
              = 4 pi r^2 rho_w dr/dt
              = 4 pi rho_w r(mass) * r_dr_dt
              = 4 pi rho_w cbrt(mass/rho_w/pi/(4/3)) r_dr_dt
        """
        return 4 * const.PI * const.rho_w * r * r_dr_dt

    @staticmethod
    def dm_dt_over_m(r, r_dr_dt):
        return 3 / r**2 * r_dr_dt

    @staticmethod
    def r_dr_dt(r, dm_dt_over_m):
        return r**2 / 3 * dm_dt_over_m
