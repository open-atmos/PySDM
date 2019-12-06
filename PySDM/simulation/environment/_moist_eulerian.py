from ._moist import _Moist
import numpy as np


class _MoistEulerian(_Moist):
    def __init__(self, particles, variables):
        super().__init__(particles, variables)
        self._dv = self.particles.mesh.dv

    @property
    def dv(self):
        return self._dv

    def get_qv(self) -> np.ndarray:
        return self._get_qv()

    def get_thd(self) -> np.ndarray:
        return self._get_thd()

    @property
    def eulerian_fields(self):
        raise NotImplementedError()
