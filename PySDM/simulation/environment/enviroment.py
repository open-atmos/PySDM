"""
Created at 28.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Environment:
    def __init__(self, particles, variables):
        self.particles = particles
        self._values = {
            "new": None,
            "old": self._allocate(variables)
        }
        self._tmp = self._allocate(variables)

    def _allocate(self, variables):
        result = {}
        for var in variables:
            result[var] = self.particles.backend.array((self.particles.mesh.n_cell,), float)
        return result

    def sync(self):
        target = self._tmp
        self.particles.backend.upload(self.qv_lambda().ravel(), target['qv'])
        self.particles.backend.upload(self.thd_lambda().ravel(), target['thd'])

        self.particles.backend.apply(
            function=self.particles.backend.temperature_pressure_RH,
            args=(self.rhod, target['thd'], target['qv']),
            output=(target['T'], target['p'], target['RH'])
        )
        self._values["new"] = target

    def _swap(self):
        self._tmp = self._values["old"]
        self._values["old"] = self._values["new"]
        self._values["new"] = None
