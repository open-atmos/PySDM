"""
particle displacement due to advection by the flow & sedimentation

adaptive time-stepping controlled by comparing implicit-Euler (I)
and explicit-Euler (E) maximal displacements with:
rtol > |(I - E) / E|
(see eqs 13-16 in [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015))
"""

from collections import namedtuple

import numpy as np

DEFAULTS = namedtuple("_", ("rtol", "adaptive"))(rtol=1e-2, adaptive=True)


class Displacement:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        enable_sedimentation=False,
        precipitation_counting_level_index: int = 0,
        adaptive=DEFAULTS.adaptive,
        rtol=DEFAULTS.rtol,
    ):  # pylint: disable=too-many-arguments
        self.particulator = None
        self.enable_sedimentation = enable_sedimentation
        self.dimension = None
        self.grid = None
        self.courant = None
        self.displacement = None
        self.temp = None
        self.precipitation_in_last_step = 0
        self.precipitation_counting_level_index = precipitation_counting_level_index

        self.adaptive = adaptive
        self.rtol = rtol
        self._n_substeps = 1

    def register(self, builder):
        builder.request_attribute("relative fall velocity")
        self.particulator = builder.particulator
        self.dimension = len(builder.particulator.environment.mesh.grid)
        self.grid = self.particulator.Storage.from_ndarray(
            np.array(builder.particulator.environment.mesh.grid, dtype=np.int64)
        )
        if self.dimension == 1:
            courant_field = (np.full(self.grid[0] + 1, np.nan),)
        elif self.dimension == 2:
            courant_field = (
                np.full((self.grid[0] + 1, self.grid[1]), np.nan),
                np.full((self.grid[0], self.grid[1] + 1), np.nan),
            )
        elif self.dimension == 3:
            courant_field = (
                np.full((self.grid[0] + 1, self.grid[1], self.grid[2]), np.nan),
                np.full((self.grid[0], self.grid[1] + 1, self.grid[2]), np.nan),
                np.full((self.grid[0], self.grid[1], self.grid[2] + 1), np.nan),
            )
        else:
            raise NotImplementedError()
        self.courant = tuple(
            self.particulator.Storage.from_ndarray(courant_field[i])
            for i in range(self.dimension)
        )
        self.displacement = self.particulator.Storage.from_ndarray(
            np.zeros((self.dimension, self.particulator.n_sd))
        )
        self.temp = self.particulator.Storage.from_ndarray(
            np.zeros((self.dimension, self.particulator.n_sd), dtype=np.int64)
        )

    def upload_courant_field(self, courant_field):
        for i, component in enumerate(courant_field):
            self.courant[i].upload(component)

        # note: to be improved, should make n_substeps variable in space as in cond/coal
        if self.adaptive:
            error_estimate = self.rtol
            self._n_substeps = 0.5
            while error_estimate >= self.rtol:
                self._n_substeps = int(self._n_substeps * 2)
                error_estimate = 0
                for i, courant_component in enumerate(courant_field):
                    max_abs_delta_courant = np.amax(
                        np.abs(np.diff(courant_component, axis=i))
                    )
                    max_abs_delta_courant /= self._n_substeps
                    error_estimate = max(
                        error_estimate,
                        (
                            0
                            if max_abs_delta_courant == 0
                            else 1 / (1 / max_abs_delta_courant - 1)
                        ),
                    )

    def __call__(self):
        # TIP: not need all array only [idx[:sd_num]]
        cell_origin = self.particulator.attributes["cell origin"]
        position_in_cell = self.particulator.attributes["position in cell"]

        self.precipitation_in_last_step = 0.0
        for _ in range(self._n_substeps):
            self.calculate_displacement(
                self.displacement, self.courant, cell_origin, position_in_cell
            )
            self.update_position(position_in_cell, self.displacement)
            if self.enable_sedimentation:
                self.precipitation_in_last_step += self.particulator.remove_precipitated(
                    displacement=self.displacement,
                    precipitation_counting_level_index=self.precipitation_counting_level_index,
                )
            self.particulator.flag_out_of_column()
            self.update_cell_origin(cell_origin, position_in_cell)
            self.boundary_condition(cell_origin)
            self.particulator.recalculate_cell_id()

        for key in ("position in cell", "cell origin", "cell id"):
            self.particulator.attributes.mark_updated(key)

    def calculate_displacement(
        self, displacement, courant, cell_origin, position_in_cell
    ):
        self.particulator.calculate_displacement(
            displacement=displacement,
            courant=courant,
            cell_origin=cell_origin,
            position_in_cell=position_in_cell,
            n_substeps=self._n_substeps,
        )
        if self.enable_sedimentation:
            displacement_z = displacement[self.dimension - 1, :]
            dt = self.particulator.dt / self._n_substeps
            dt_over_dz = dt / self.particulator.mesh.dz
            displacement_z *= 1 / dt_over_dz
            displacement_z -= self.particulator.attributes["relative fall velocity"]
            displacement_z *= dt_over_dz

    @staticmethod
    def update_position(position_in_cell, displacement):
        position_in_cell += displacement

    def update_cell_origin(self, cell_origin, position_in_cell):
        floor_of_position = self.temp
        floor_of_position.floor(position_in_cell)
        cell_origin += floor_of_position
        position_in_cell -= floor_of_position

    def boundary_condition(self, cell_origin):
        cell_origin %= self.grid
