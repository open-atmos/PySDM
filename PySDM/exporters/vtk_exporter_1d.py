"""
vtk exporter of particle attributes for 1d simulations
"""

import os

import numpy as np
from pyevtk.hl import pointsToVTK


class VTKExporter_1d:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        data,
        settings,
        path="./sd_attributes/",
        exclude_particle_reservoir=True,
    ):
        self.data = data
        self.settings = settings
        self.path = path
        if not os.path.isdir(self.path) and len(settings.save_spec_and_attr_times) > 0:
            os.mkdir(self.path)

        self.exclude_particle_reservoir = exclude_particle_reservoir

        self.num_len = len(str(settings.t_max))

    def _export_attributes(self, time_index_and_value):
        time_index = time_index_and_value[0]
        time = time_index_and_value[1]
        path = self.path + "time" + self._add_leading_zeros(time)

        payload = {}
        for k in self.data.keys():
            if len(self.data[k][time_index].shape) == 1:
                payload[k] = self.data[k][time_index]
            elif len(self.data[k][time_index].shape) == 2:
                assert self.data[k][time_index].shape[0] == 1
                payload[k] = self.data[k][time_index][0]
            else:
                raise NotImplementedError("Shape of data array is not recognized.")

        z = (
            self.settings.dz * (payload["cell origin"] + payload["position in cell"])
            - self.settings.particle_reservoir_depth
        )

        if self.exclude_particle_reservoir:
            reservoir_particles_indexes = np.where(z < 0)
            z = np.delete(z, reservoir_particles_indexes)
            keys = payload.keys()
            for k in keys:
                payload[k] = np.delete(payload[k], reservoir_particles_indexes)

        x = np.full_like(z, 0)
        y = np.full_like(z, 0)

        pointsToVTK(path, x, y, z, data=payload)

    def _add_leading_zeros(self, a):
        return "".join(["0" for i in range(self.num_len - len(str(a)))]) + str(a)

    def run(self):
        for time_index_and_value in enumerate(self.settings.save_spec_and_attr_times):
            self._export_attributes(time_index_and_value)
