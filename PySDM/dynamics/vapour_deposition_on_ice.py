from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class VapourDepositionOnIce:
    def __init__(self):
        """called by the user while building a particulator"""
        self.particulator = None

    def register(self, *, builder):
        """called by the builder"""
        self.particulator = builder.particulator

    def __call__(self):
        """called by the particulator during simulation"""
        self.particulator.deposition()
