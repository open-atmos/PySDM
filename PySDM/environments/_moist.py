"""
"""

import numpy as np


class _Moist:

    def __init__(self, dt, mesh, variables):
        variables += ['qv', 'thd', 'T', 'p', 'RH']
        self.core = None
        self.dt = dt
        self.mesh = mesh
        self.variables = variables
        self._values = None
        self._tmp = None

    def register(self, builder):
        self.core = builder.core
        self.core.observers.append(self)
        self._values = {
            "predicted": None,
            "current": self._allocate(self.variables)
        }
        self._tmp = self._allocate(self.variables)

    def _allocate(self, variables):
        result = {}
        for var in variables:
            result[var] = self.core.Storage.empty((self.mesh.n_cell,), float)
        return result

    def __getitem__(self, index):
        return self._values['current'][index]

    def get_predicted(self, index):
        if self._values['predicted'] is None:
            raise AssertionError("Environment is not synchronized.")
        return self._values['predicted'][index]

    def sync(self):
        target = self._tmp
        target['qv'].ravel(self.get_qv())
        target['thd'].ravel(self.get_thd())

        self.core.backend.temperature_pressure_RH(
            target['rhod'], target['thd'], target['qv'],
            target['T'], target['p'], target['RH']
        )
        self._values["predicted"] = target

    def get_qv(self) -> np.ndarray: raise NotImplementedError()
    def get_thd(self) -> np.ndarray: raise NotImplementedError()

    def notify(self):
        if self._values["predicted"] is None:
            return

        self._tmp = self._values["current"]
        self._values["current"] = self._values["predicted"]
        self._values["predicted"] = None
