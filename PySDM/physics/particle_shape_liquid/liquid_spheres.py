"""
spherical particles with constant density of ice
"""
from PySDM.physics.particle_shape_common import Spheres

class LiquidSpheres(Spheres):
    def __init__(self, const, _):
        self.mass_density = const.rho_w

    @staticmethod
    def reynolds_number(_, radius, velocity_wrt_air, dynamic_viscosity, density):
        return 2 * radius * velocity_wrt_air * density / dynamic_viscosity