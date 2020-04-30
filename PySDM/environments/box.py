from PySDM.mesh import Mesh
from PySDM.particles_builder import ParticlesBuilder


class Box:
    def __init__(self, _: ParticlesBuilder, dv=None):
        self.mesh = Mesh.mesh_0d(dv)
        pass

    def ante_step(self): pass
    def post_step(self): pass
