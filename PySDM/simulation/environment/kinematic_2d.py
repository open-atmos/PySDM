"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory, z_vec_coord, x_vec_coord


class Kinematic2D:
    def __init__(self, particles, stream_function, field_values, rhod_lambda):
        self.particles = particles
        # TODO: rename
        self.rhod_lambda = rhod_lambda

        rhod = np.repeat(
            rhod_lambda(
                (np.arange(particles.grid[1]) + 1 / 2) / particles.grid[1]
            ).reshape((1, particles.grid[1])),
            particles.grid[0],
            axis=0
        )

        self.GC, self.eulerian_fields = MPDATAFactory.kinematic_2d(
            grid=particles.grid, size=particles.size, dt=particles.dt,
            stream_function=stream_function,
            field_values=field_values,
            g_factor=rhod
        )

        self.thd_lambda = lambda: self.eulerian_fields.mpdatas["th"].curr.get()
        self.qv_lambda = lambda: self.eulerian_fields.mpdatas["qv"].curr.get()

        self.rhod = particles.backend.from_ndarray(rhod.ravel())

        self._values = {
            "new": None,
            "old": self._allocate()
        }
        self._tmp = self._allocate()

        # TODO
        self.sync()
        self.post_step()

    def _allocate(self):
        result = {}
        for var in ['qv', 'thd', 'RH', 'p', 'T']:
            result[var] = self.particles.backend.array((self.particles.n_cell,), float)
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

    # TODO: this is only used from within PySDM, examples always use ["old"] - awkward
    def __getitem__(self, index):
        values = self._values[index]
        if values is None:
            raise Exception("condensation not called.")
        return values

    # TODO: rename?
    def ante_step(self):
        self.eulerian_fields.step()

    def post_step(self):
        self.particles.backend.download(self._values["new"]["qv"].reshape(self.particles.grid), self.qv_lambda())
        self.particles.backend.download(self._values["new"]["thd"].reshape(self.particles.grid), self.thd_lambda())

        self._tmp = self._values["old"]
        self._values["old"] = self._values["new"]
        self._values["new"] = None
        
    def get_courant_field_data(self):
        result = [  # TODO: test it!!!!
            self.GC.data(0) / self.rhod_lambda(
                x_vec_coord(self.particles.grid, self.particles.size)[1]),
            self.GC.data(1) / self.rhod_lambda(
                z_vec_coord(self.particles.grid, self.particles.size)[1])
        ]
        return result


