import inspect
from threading import Thread

import numpy as np
from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PySDM_examples.Szumowski_et_al_1998.fields import (
    nondivergent_vector_field_2d,
    x_vec_coord,
    z_vec_coord,
)

from PySDM.backends.impl_numba import conf
from PySDM.impl.arakawa_c import make_rhod


class MPDATA_2D:
    def __init__(
        self,
        *,
        advectees,
        stream_function,
        rhod_of_zZ,
        dt,
        grid,
        size,
        displacement,
        n_iters=2,
        infinite_gauge=True,
        nonoscillatory=True,
        third_order_terms=False
    ):
        self.grid = grid
        self.size = size
        self.dt = dt
        self.stream_function = stream_function
        self.stream_function_time_dependent = (
            "t" in inspect.signature(stream_function).parameters
        )
        self.asynchronous = False
        self.thread: (Thread, None) = None
        self.displacement = displacement
        self.t = 0

        options = Options(
            n_iters=n_iters,
            infinite_gauge=infinite_gauge,
            nonoscillatory=nonoscillatory,
            third_order_terms=third_order_terms,
        )
        disable_threads_if_needed = {}
        if not conf.JIT_FLAGS["parallel"]:
            disable_threads_if_needed["n_threads"] = 1

        stepper = Stepper(
            options=options,
            grid=self.grid,
            non_unit_g_factor=True,
            **disable_threads_if_needed
        )

        advector_impl = VectorField(
            (
                np.full((grid[0] + 1, grid[1]), np.nan),
                np.full((grid[0], grid[1] + 1), np.nan),
            ),
            halo=options.n_halo,
            boundary_conditions=(Periodic(), Periodic()),
        )

        g_factor = make_rhod(self.grid, rhod_of_zZ)
        g_factor_impl = ScalarField(
            g_factor.astype(dtype=options.dtype),
            halo=options.n_halo,
            boundary_conditions=(Periodic(), Periodic()),
        )

        self.g_factor_vec = (
            rhod_of_zZ(zZ=x_vec_coord(self.grid)[-1]),
            rhod_of_zZ(zZ=z_vec_coord(self.grid)[-1]),
        )
        self.mpdatas = {}
        for k, v in advectees.items():
            advectee_impl = ScalarField(
                np.asarray(v, dtype=options.dtype),
                halo=options.n_halo,
                boundary_conditions=(Periodic(), Periodic()),
            )
            self.mpdatas[k] = Solver(
                stepper=stepper,
                advectee=advectee_impl,
                advector=advector_impl,
                g_factor=g_factor_impl,
            )

    def __getitem__(self, item):
        return self.mpdatas[item]

    def __call__(self):
        if self.asynchronous:
            self.thread = Thread(target=self.step, args=())
            self.thread.start()
        else:
            self.step()

    def wait(self):
        if self.asynchronous:
            if self.thread is not None:
                self.thread.join()

    def refresh_advector(self):
        for mpdata in self.mpdatas.values():
            advector = nondivergent_vector_field_2d(
                self.grid, self.size, self.dt, self.stream_function, t=self.t
            )
            for d in range(len(self.grid)):
                np.testing.assert_array_less(np.abs(advector[d]), 1)
                mpdata.advector.get_component(d)[:] = advector[d]
            if self.displacement is not None:
                for d in range(len(self.grid)):
                    advector[d] /= self.g_factor_vec[d]
                self.displacement.upload_courant_field(advector)
            break  # the advector field is shared

    def step(self):
        if not self.stream_function_time_dependent and self.t == 0:
            self.refresh_advector()

        self.t += 0.5 * self.dt
        if self.stream_function_time_dependent:
            self.refresh_advector()
        for mpdata in self.mpdatas.values():
            mpdata.advance(1)
        self.t += 0.5 * self.dt
