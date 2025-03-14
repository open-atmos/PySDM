"""
spherical particles with constant density of ice
"""
from PySDM.physics.particle_shape_common import Spheres

class IceSpheres(Spheres):
    def __init__(self, const, _):
        self.mass_density = const.rho_i

    @staticmethod
    def supports_mixed_phase(_=None):
        return True


