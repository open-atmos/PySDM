"""basic water vapor deposition on ice"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class VapourDepositionOnIce:
    def __init__(self):
        """called by the user while building a particulator"""
        self.particulator = None

    def register(self, *, builder):
        """called by the builder"""
        self.particulator = builder.particulator
        assert builder.formulae.particle_shape_and_density.supports_mixed_phase()
        builder.request_attribute("Reynolds number")

    def __call__(self):
        """called by the particulator during simulation"""
        self.particulator.deposition()
