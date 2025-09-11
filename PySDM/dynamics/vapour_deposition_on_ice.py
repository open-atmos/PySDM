"""basic water vapor deposition on ice"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class VapourDepositionOnIce:
    def __init__(self, adaptive: bool = True):
        """called by the user while building a particulator"""
        self.particulator = None
        self.adaptive = adaptive

    def register(self, *, builder):
        """called by the builder"""
        self.particulator = builder.particulator
        assert builder.formulae.particle_shape_and_density.supports_mixed_phase()
        for var in ("water vapour mixing ratio", "dry air potential temperature"):
            builder.request_attribute(f"dropwise {var}")
            builder.request_attribute(f"dropwise {var} tendency")

    def __call__(self):
        """called by the particulator during simulation"""
        self.particulator.deposition(adaptive=self.adaptive)
