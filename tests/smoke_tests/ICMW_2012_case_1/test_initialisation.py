"""
Created at 18.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from examples.ICMW_2012_case_1.setup import Setup
from examples.ICMW_2012_case_1.example import Simulation
from PySDM.simulation.physics.constants import si
from PySDM.utils import Physics


def test():
    setup = Setup()
    simulation = Simulation(setup, None)

    # Act (moments)
    simulation.init()
    state = simulation.state
    
    # Asset (TODO: turn plotting into asserts)
    from matplotlib import pyplot

    x_bins = np.logspace(
        (np.log10(Physics.r2x(setup.r_min))),
        (np.log10(Physics.r2x(setup.r_max))),
        num=10,
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
        vals[i, :] = tmp.reshape(setup.grid).sum() / (setup.dv * setup.grid[0] * setup.grid[1])
#        vals[i, :] = tmp.reshape(setup.grid).sum(axis=0) / (setup.dv * setup.grid[0])
        #vals[i, :] /= (np.log(r_bins[i + 1]) - np.log(r_bins[i]))

    for level in range(0, setup.grid[1], 5):
        pyplot.step(
            r_bins[:-1] * si.metres / si.micrometres,
            vals[:, level] * si.metre**3 / si.centimetre**3,
            where='post'
        )
    pyplot.grid()
    pyplot.xscale('log')
    pyplot.xlabel('particle radius [µm]')
    pyplot.ylabel('concentration [cm^{-3}/(unit dr/r)]')
    pyplot.legend()
    pyplot.show()

