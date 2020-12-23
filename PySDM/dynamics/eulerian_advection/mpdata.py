"""
Created at 04.09.2020
"""

import numpy as np
from threading import Thread
from PyMPDATA import Options, Stepper, VectorField, ScalarField, Solver
from PyMPDATA.arakawa_c.boundary_condition.periodic_boundary_condition import PeriodicBoundaryCondition
from ...backends.numba import conf
from numba.core.errors import NumbaExperimentalFeatureWarning


class MPDATA:
    def __init__(self, *, fields,
                 n_iters=2, infinite_gauge=True,
                 flux_corrected_transport=True, third_order_terms=False):
        self.grid = fields.g_factor.shape
        self.asynchronous = False
        self.thread: (Thread, None) = None

        options = Options(
            n_iters=n_iters,
            infinite_gauge=infinite_gauge,
            flux_corrected_transport=flux_corrected_transport,
            third_order_terms=third_order_terms
        )
        disable_threads_if_needed = {}
        if not conf.JIT_FLAGS['parallel']:
            disable_threads_if_needed['n_threads'] = 1

        stepper = Stepper(options=options, grid=self.grid, non_unit_g_factor=True, **disable_threads_if_needed)

        # CFL condition
        for d in range(len(fields.advector)):
            np.testing.assert_array_less(np.abs(fields.advector[d]), 1)

        self.advector = fields.advector
        advector_impl = VectorField(fields.advector, halo=options.n_halo,
                                    boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))

        # nondivergence (of velocity field, hence dt)  # TODO: move to better place
        # assert np.amax(abs(advector_impl.div((dt, dt)).get())) < 5e-9

        self.g_factor = fields.g_factor
        g_factor_impl = ScalarField(fields.g_factor.astype(dtype=options.dtype), halo=options.n_halo,
                               boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        self.mpdatas = {}
        for k, v in fields.advectees.items():
            advectee = ScalarField(np.full(self.grid, v, dtype=options.dtype), halo=options.n_halo,
                                   boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
            self.mpdatas[k] = Solver(stepper=stepper, advectee=advectee, advector=advector_impl, g_factor=g_factor_impl)

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

    def step(self):
        try:
            for mpdata in self.mpdatas.values():
                mpdata.advance(1)
        except NumbaExperimentalFeatureWarning:
            pass
