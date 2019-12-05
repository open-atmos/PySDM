from ._moist import _Moist


class _MoistEulerian(_Moist):
    def __init__(self, particles, variables):
        super().__init__(particles, variables)
        self._dv = self.particles.mesh.dv

    @property
    def dv(self):
        return self._dv
