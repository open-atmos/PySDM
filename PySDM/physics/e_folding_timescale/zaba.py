"""
e-folding timescale for droplet evaporation tau = [1/m dm/dt]^-1
"""


class Zaba:
    def __init__(self):
        pass

    @staticmethod
    def e_folding_tau_of_mass(m, dm_dt):
        return m / dm_dt

    @staticmethod
    def e_folding_tau_of_radius(r, r_dr_dt):
        return r**2 / 3 / r_dr_dt
