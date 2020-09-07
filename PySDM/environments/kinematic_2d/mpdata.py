"""
Created at 04.09.2020
"""

import numpy as np
from threading import Thread
from MPyDATA import Options, Stepper, VectorField, ScalarField, Solver
from MPyDATA.arakawa_c.boundary_condition.periodic_boundary_condition import PeriodicBoundaryCondition


class MPDATA:
    def __init__(self, *, grid, dt, field_values, g_factor: np.ndarray, advector,
                 mpdata_iters, mpdata_infinite_gauge, mpdata_flux_corrected_transport,
                 mpdata_third_order_terms):

        self.asynchronous = True
        self.thread: (Thread, None) = None

        options = Options(
            n_iters=mpdata_iters,
            infinite_gauge=mpdata_infinite_gauge,
            flux_corrected_transport=mpdata_flux_corrected_transport,
            third_order_terms=mpdata_third_order_terms
        )
        stepper = Stepper(options=options, grid=grid, non_unit_g_factor=True)

        # CFL condition
        for d in range(len(advector)):
            np.testing.assert_array_less(np.abs(advector[d]), 1)

        self.advector = advector
        advector_impl = VectorField(advector, halo=options.n_halo,
                                    boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))

        # nondivergence (of velocity field, hence dt)
        assert np.amax(abs(advector_impl.div((dt, dt)).get())) < 5e-9

        self.g_factor = g_factor
        g_factor_impl = ScalarField(g_factor.astype(dtype=options.dtype), halo=options.n_halo,
                               boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        self.mpdatas = {}
        for k, v in field_values.items():
            advectee = ScalarField(np.full(grid, v, dtype=options.dtype), halo=options.n_halo,
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
        for mpdata in self.mpdatas.values():
            mpdata.advance(1)
