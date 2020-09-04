"""
Created at 04.09.2020
"""
import numpy as np
from MPyDATA import Options, Stepper, VectorField, ScalarField, Solver
from MPyDATA.arakawa_c.boundary_condition.periodic_boundary_condition import PeriodicBoundaryCondition


def make_advection_solver(*, grid, dt, field_values, g_factor: np.ndarray, rho_times_courant,
                          mpdata_iters, mpdata_infinite_gauge, mpdata_flux_corrected_transport,
                          mpdata_third_order_terms, mpdata_n_threads):
    options = Options(
        n_iters=mpdata_iters,
        infinite_gauge=mpdata_infinite_gauge,
        flux_corrected_transport=mpdata_flux_corrected_transport,
        third_order_terms=mpdata_third_order_terms
    )
    stepper = Stepper(options=options, grid=grid, non_unit_g_factor=True, n_threads=mpdata_n_threads)

    # CFL condition
    for d in range(len(rho_times_courant)):
        np.testing.assert_array_less(np.abs(rho_times_courant[d]), 1)

    advector = VectorField(rho_times_courant, halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(advector.div((dt, dt)).get())) < 5e-9

    g_factor = ScalarField(g_factor.astype(dtype=options.dtype), halo=options.n_halo,
                           boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
    mpdatas = {}
    for k, v in field_values.items():
        advectee = ScalarField(np.full(grid, v, dtype=options.dtype), halo=options.n_halo,
                               boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        mpdatas[k] = Solver(stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor)

    return advector, mpdatas