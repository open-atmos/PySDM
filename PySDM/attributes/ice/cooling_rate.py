"""
cooling rate estimated as the difference in current and previous grid-cell
 temperatures divided by the timestep (i.e. equals zero if particle has not
 moved to a different cell since the last timestep)
"""

import numpy as np
from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class CoolingRate(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute("cell id")
        super().__init__(
            builder=builder, name="cooling rate", dependencies=(self.cell_id,)
        )
        self.prev_T = builder.particulator.backend.Storage.from_ndarray(
            np.full(builder.particulator.n_sd, np.nan)
        )
        builder.particulator.observers.append(self)

    def notify(self):
        cell_id = self.particulator.attributes["cell id"]
        self.prev_T[:] = self.particulator.environment["T"][cell_id]

    def recalculate(self):
        cell_id = self.particulator.attributes["cell id"]
        env_T = self.particulator.environment["T"]
        self.data[:] = env_T[cell_id]
        self.data -= self.prev_T
        self.data /= -self.particulator.environment.dt
