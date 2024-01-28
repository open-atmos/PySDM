"""
rates of activation, deactivation and ripening events (take into account substeps,
 fetching a value resets the given counter)
"""

import numpy as np

from PySDM.products.impl.product import Product


class EventRate(Product):
    def __init__(self, what, name=None, unit=None):
        super().__init__(name=name, unit=unit)
        self.condensation = None
        self.timestep_count = 0
        self.what = what
        self.event_count = None

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.condensation = self.particulator.dynamics["Condensation"]
        self.event_count = np.zeros_like(self.buffer)

    def notify(self):
        self.timestep_count += 1
        self._download_to_buffer(self.condensation.counters["n_" + self.what])
        self.event_count[:] += self.buffer[:]

    def _impl(self, **kwargs):
        if self.timestep_count == 0:
            return self.event_count

        self.event_count[:] /= (
            self.timestep_count * self.particulator.dt * self.particulator.mesh.dv
        )
        self._download_to_buffer(self.particulator.environment["rhod"])
        self.event_count[:] /= self.buffer[:]
        self.buffer[:] = self.event_count[:]
        self.timestep_count = 0
        self.event_count[:] = 0
        return self.buffer


class RipeningRate(EventRate):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__("ripening", name=name, unit=unit)


class ActivatingRate(EventRate):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__("activating", name=name, unit=unit)


class DeactivatingRate(EventRate):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__("deactivating", name=name, unit=unit)
