"""
Created at 18.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.initialisation import spatial_discretisation, spectral_discretisation
from PySDM.simulation.initialisation.r_wet_init import r_wet_init
from PySDM import utils
from PySDM.utils import Physics

from examples.ICMW_2012_case_1.setup import Setup
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory
from PySDM.simulation.physics.constants import si


def test():
    setup = Setup()

    _, eulerian_fields = MPDATAFactory.kinematic_2d(
        grid=setup.grid, size=setup.size, dt=setup.dt,
        stream_function=setup.stream_function,
        field_values=setup.field_values)

    ambient_air = setup.ambient_air(
        grid=setup.grid,
        backend=setup.backend,
        thd_xzt_lambda=lambda: eulerian_fields.mpdatas["th"].curr.get(),
        qv_xzt_lambda=lambda: eulerian_fields.mpdatas["qv"].curr.get(),
        rhod_z_lambda=setup.rhod
    )

    r_dry, n = spectral_discretisation.constant_multiplicity(
        setup.n_sd, setup.spectrum, (setup.r_min, setup.r_max)
    )
    positions = spatial_discretisation.pseudorandom(setup.grid, setup.n_sd)

    # <TEMP>
    cell_origin = positions.astype(dtype=int)
    strides = utils.strides(setup.grid)
    cell_id = np.dot(strides, cell_origin.T).ravel()
    # </TEMP>

    r_wet = r_wet_init(r_dry, ambient_air, cell_id, setup.kappa)
    state = StateFactory.state_2d(n=n, grid=setup.grid,
                                  # TODO: rename x -> ...
                                  extensive={'x': utils.Physics.r2x(r_wet), 'dry volume': Physics.r2x(r_dry)},
                                  intensive={},
                                  positions=positions,
                                  backend=setup.backend)

    # Act (moments)
    
    # Asset (TODO: turn plotting into asserts)
    from matplotlib import pyplot

    x_bins = np.logspace(
        (np.log10(Physics.r2x(setup.r_min))),
        (np.log10(Physics.r2x(setup.r_max))),
        num=64,
        endpoint=True
    )
    r_bins = Physics.x2r(x_bins)

    vals = np.empty((len(r_bins) - 1, setup.grid[1]))

    n_moments = 1
    moment_0 = state.backend.array(state.n_cell, dtype=int)
    moments = state.backend.array((n_moments, state.n_cell), dtype=float)
    tmp = np.empty(state.n_cell)
    for i in range(len(vals)):
        state.moments(moment_0, moments, specs={}, attr_name='dry volume', attr_range=(x_bins[i], x_bins[i + 1]))
        state.backend.download(moment_0, tmp)
        #vals[i] *= setup.rho / setup.dv
        #vals[i] /= (np.log(r_bins[i + 1]) - np.log(r_bins[i]))
        vals[i, :] = tmp.reshape(setup.grid).sum(axis=0)

    for lev in range(0, setup.grid[1], 5):
        pyplot.step(
            r_bins[:-1] * si.metres / si.micrometres,
            vals[:, lev] * si.kilograms / si.grams,
            where='post'
        )
    pyplot.grid()
    pyplot.xscale('log')
    pyplot.xlabel('particle radius [µm]')
    pyplot.ylabel('dm/dlnr [g/m^3/(unit dr/r)]')
    pyplot.legend()
    pyplot.show()

