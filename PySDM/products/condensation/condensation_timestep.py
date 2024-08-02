"""
minimum and maximum condensation timestep (fetching a value resets the counter)
"""

import numpy as np

from PySDM.products.impl import Product, register_product


@register_product()
class _CondensationTimestep(Product):
    def __init__(self, name, unit, extremum, reset_value):
        super().__init__(
            name=name,
            unit=unit,
        )
        self.extremum = extremum
        self.reset_value = reset_value
        self.value = None
        self.condensation = None
        self.range = None

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.condensation = self.particulator.dynamics["Condensation"]
        self.range = self.condensation.dt_cond_range
        self.value = np.full_like(self.buffer, np.nan)

    def notify(self):
        self._download_to_buffer(self.condensation.counters["n_substeps"])
        self.buffer[:] = self.condensation.particulator.dt / self.buffer
        self.value = self.extremum(self.buffer, self.value)

    def _impl(self, **kwargs):
        self.buffer[:] = self.value[:]
        self.value[:] = self.reset_value
        return self.buffer


@register_product()
class CondensationTimestepMin(_CondensationTimestep):
    def __init__(self, name=None, unit="s"):
        super().__init__(name=name, unit=unit, extremum=np.minimum, reset_value=np.inf)


@register_product()
class CondensationTimestepMax(_CondensationTimestep):
    def __init__(self, name=None, unit="s"):
        super().__init__(name=name, unit=unit, extremum=np.maximum, reset_value=-np.inf)
