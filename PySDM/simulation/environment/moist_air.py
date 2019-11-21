"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np


class MoistAir:
    def __init__(self, simulation, thd_xzt_lambda, qv_xzt_lambda, rhod):
        self.simulation = simulation

        self.thd_lambda = thd_xzt_lambda
        self.qv_lambda = qv_xzt_lambda

        self.rhod = simulation.backend.from_ndarray(rhod.ravel())

        self._values = {
            "new": None,
            "old": self._allocate()
        }
        self._tmp = self._allocate()

        self.sync()
        self.signal_next_timestep()

    def _allocate(self):
        result = {}
        for var in ['qv', 'thd', 'RH', 'p', 'T']:
            result[var] = self.simulation.backend.array((self.simulation.n_cell,), float)
        return result

    def sync(self):
        target = self._tmp
        self.simulation.backend.upload(self.qv_lambda().ravel(), target['qv'])
        self.simulation.backend.upload(self.thd_lambda().ravel(), target['thd'])

        self.simulation.backend.apply(
             function=self.simulation.backend.temperature_pressure_RH,
             args=(self.rhod, target['thd'], target['qv']),
             output=(target['T'], target['p'], target['RH'])
        )
        self._values["new"] = target

    # TODO: this is only used from within PySDM, examples always use ["old"] - awkward
    def __getitem__(self, index):
        values = self._values[index]
        if values is None:
            raise Exception("condensation not called.")
        return values

    # TODO: rename?
    def signal_next_timestep(self):
        self.simulation.backend.download(self._values["new"]["qv"].reshape(self.simulation.grid), self.qv_lambda())
        self.simulation.backend.download(self._values["new"]["thd"].reshape(self.simulation.grid), self.thd_lambda())

        self._tmp = self._values["old"]
        self._values["old"] = self._values["new"]
        self._values["new"] = None


