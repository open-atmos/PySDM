import inspect
from functools import cached_property
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
        n_iters=2,
        infinite_gauge=True,
        nonoscillatory=True,
        third_order_terms=False,
    ):
        self._grid = grid
        self.size = size
        self.dt = dt
        self.stream_function = stream_function
        self.stream_function_time_dependent = (
            "t" in inspect.signature(stream_function).parameters
        )
        self.asynchronous = False
        self.thread: (Thread, None) = None
        self.t = 0
        self.advectees = advectees

        self._options = Options(
            n_iters=n_iters,
            infinite_gauge=infinite_gauge,
            nonoscillatory=nonoscillatory,
            third_order_terms=third_order_terms,
        )

        self.g_factor = make_rhod(grid, rhod_of_zZ)
        self.g_factor_vec = (
            rhod_of_zZ(zZ=x_vec_coord(grid)[-1]),
            rhod_of_zZ(zZ=z_vec_coord(grid)[-1]),
        )

    @cached_property
    def mpdatas(self):
        disable_threads_if_needed = {}
        if not conf.JIT_FLAGS["parallel"]:
            disable_threads_if_needed["n_threads"] = 1

        stepper = Stepper(
            options=self._options,
            grid=self._grid,
            non_unit_g_factor=True,
            **disable_threads_if_needed,
        )

        advector_impl = VectorField(
            (
                np.full((self._grid[0] + 1, self._grid[1]), np.nan),
                np.full((self._grid[0], self._grid[1] + 1), np.nan),
            ),
            halo=self._options.n_halo,
            boundary_conditions=(Periodic(), Periodic()),
        )

        g_factor_impl = ScalarField(
            self.g_factor.astype(dtype=self._options.dtype),
            halo=self._options.n_halo,
            boundary_conditions=(Periodic(), Periodic()),
        )

        mpdatas = {}
        for k, v in self.advectees.items():
            advectee_impl = ScalarField(
                np.asarray(v, dtype=self._options.dtype),
                halo=self._options.n_halo,
                boundary_conditions=(Periodic(), Periodic()),
            )
            mpdatas[k] = Solver(
                stepper=stepper,
                advectee=advectee_impl,
                advector=advector_impl,
                g_factor=g_factor_impl,
            )
        return mpdatas

    def __getitem__(self, key: str):
        if "mpdatas" in self.__dict__:
            return self.mpdatas[key].advectee.get()
        return self.advectees[key]

    def __call__(self, displacement):
        if self.asynchronous:
            self.thread = Thread(target=self.step, args=())
            self.thread.start()
        else:
            self.step(displacement)

    def wait(self):
        if self.asynchronous:
            if self.thread is not None:
                self.thread.join()

    def refresh_advector(self, displacement):
        for mpdata in self.mpdatas.values():
            advector = nondivergent_vector_field_2d(
                self._grid, self.size, self.dt, self.stream_function, t=self.t
            )
            for d in range(len(self._grid)):
                np.testing.assert_array_less(np.abs(advector[d]), 1)
                mpdata.advector.get_component(d)[:] = advector[d]
            if displacement is not None:
                for d in range(len(self._grid)):
                    advector[d] /= self.g_factor_vec[d]
                displacement.upload_courant_field(advector)
            break  # the advector field is shared

    def step(self, displacement):
        if not self.stream_function_time_dependent and self.t == 0:
            self.refresh_advector(displacement)

        self.t += 0.5 * self.dt
        if self.stream_function_time_dependent:
            self.refresh_advector(displacement)
        for mpdata in self.mpdatas.values():
            mpdata.advance(1)
        self.t += 0.5 * self.dt
