from ._moist import _Moist
import numpy as np


class _MoistEulerian(_Moist):
    def __init__(self, particles, dt, mesh, variables):
        super().__init__(particles, dt, mesh, variables)

    @property
    def dv(self):
        return self.mesh.dv

    def get_qv(self) -> np.ndarray:
        return self._get_qv()

    def get_thd(self) -> np.ndarray:
        return self._get_thd()

    def step(self):
        raise NotImplementedError()
