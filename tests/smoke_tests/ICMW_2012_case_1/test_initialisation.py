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

from matplotlib import pyplot


def test_dry_radius(plot=False):
    setup = Setup()
    setup.grid = (10, 10)
    setup.n_sd_per_gridbox = 5000

    simulation = Simulation(setup, None)

    n_bins = 32
    n_levels = setup.grid[1]
    x_bins = np.logspace(
        (np.log10(Physics.r2x(setup.r_min))),
        (np.log10(Physics.r2x(setup.r_max))),
        num=n_bins,
        endpoint=True
    )
    r_bins = Physics.x2r(x_bins)

    vals = np.empty((len(r_bins) - 1, setup.grid[1]))

    n_moments = 1
    n_cell = np.prod(np.array(setup.grid))
    moment_0 = setup.backend.array(n_cell, dtype=int)
    moments = setup.backend.array((n_moments, n_cell), dtype=float)
    tmp = np.empty(n_cell)

    # Act (moments)
    simulation.init()
    state = simulation.state
    rhod = setup.backend.to_ndarray(simulation.ambient_air.rhod).reshape(setup.grid).mean(axis=0)

    for i in range(len(vals)):
        state.moments(moment_0, moments, specs={}, attr_name='dry volume', attr_range=(x_bins[i], x_bins[i + 1]))
        state.backend.download(moment_0, tmp)
        vals[i, :] = tmp.reshape(setup.grid).sum(axis=0) / (setup.dv * setup.grid[0])
    # vals[i, :] /= (np.log(r_bins[i + 1]) - np.log(r_bins[i]))

    # Plot
    if plot:
        for level in range(0, n_levels):
            pyplot.step(
                r_bins[:-1] * si.metres / si.micrometres,
                vals[:, level] * si.metre ** 3 / si.centimetre ** 3,
                where='post'
            )
        pyplot.grid()
        pyplot.xscale('log')
        pyplot.xlabel('particle radius [µm]')
        pyplot.ylabel('concentration [cm^{-3}/(unit dr/r)]')
        pyplot.legend()
        pyplot.show()

    # Assert - location of maximum
    for level in range(n_levels):
        real_max = setup.spectrum_per_mass_of_dry_air.distribution_params[2]
        idx_max = np.argmax(vals[:, level])
        assert r_bins[idx_max] < real_max < r_bins[idx_max+1]

    # Assert - total number
    for level in reversed(range(n_levels)):
        mass_conc = np.sum(vals[:, level]) / rhod[level]
        mass_conc_STP = setup.spectrum_per_mass_of_dry_air.norm_factor
        assert .5 * mass_conc_STP < mass_conc < 1.5 * mass_conc_STP

    # Assert - decreasing number density
    total_above = 0
    for level in reversed(range(n_levels)):
        total_below = np.sum(vals[:, level])
        assert total_below > total_above
        total_above = total_below

