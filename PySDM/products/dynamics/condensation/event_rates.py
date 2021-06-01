from PySDM.products.product import Product
import numpy as np
from PySDM.physics.constants import convert_to, si


class EventRate(Product):

    def __init__(self, what):
        super().__init__(
            name=what+'_rate',
            description=what+' rate',
            unit='s-1 mg-1'
        )
        self.condensation = None
        self.timestep_count = 0
        self.what = what
        self.event_count = None

    def register(self, builder):
        super().register(builder)
        self.core.observers.append(self)
        self.condensation = self.core.dynamics['Condensation']
        self.event_count = np.zeros_like(self.buffer)

    def notify(self):
        self.timestep_count += 1
        self.download_to_buffer(self.condensation.counters['n_'+self.what])
        self.event_count[:] += self.buffer[:]

    def get(self):
        if self.timestep_count == 0:
            return self.event_count
        else:
            self.event_count[:] /= (self.timestep_count * self.core.dt * self.core.mesh.dv)
            self.download_to_buffer(self.core.environment['rhod'])
            self.event_count[:] /= self.buffer[:]
            self.buffer[:] = self.event_count[:]
            self.timestep_count = 0
            self.event_count[:] = 0
            convert_to(self.buffer, 1/si.mg)
            return self.buffer


class RipeningRate(EventRate):
    def __init__(self):
        super().__init__('ripening')


class ActivatingRate(EventRate):
    def __init__(self):
        super().__init__('activating')


class DeactivatingRate(EventRate):
    def __init__(self):
        super().__init__('deactivating')
